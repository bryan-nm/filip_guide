"""End-to-end best-of-n smoke test with the REAL models, small enough to run on a
laptop. Complements smoke_best_of_n.py (which stubs the models): this one drives
the actual ESM3 generator + FILIP classifier + tokenizer bridge through
`guided_generate`, then independently re-derives each candidate's path
log-likelihood from the real ESM3 forwards, and finally round-trips a shard
through `merge`.

Local weights (~/Documents/models) + the in-repo retrieval checkpoint:

    FILIPG_MODELS_DIR=~/Documents/models \\
      python smoke_generate_local.py --ckpt best_hardneg_epoch49.pt --device cpu

Needs the `esm` package (ESM3 + ESM-C). Deliberately tiny — length 30 (the
pipeline's min_len, and the shortest length at which the composition thresholds
are meaningful), n 4 — so it exercises every code path in ~2 min on CPU without
approximating a real generation run. Set FILIPG_PROTEIN_ENCODER_TYPE=amplify to
run it in the Aurora classifier configuration instead.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import types

import torch

from config import default_cfg
from src import encoders, prompt_bank
from src.bridge import TokenizerBridge
from src.esm3_generative import build_generator
from src.filip_predictor import build_predictor_from_checkpoint
from src.generate_dataset import _candidate_records, _score, _select, merge
from src.guide import DEGGuidance
from src.sample import guided_generate

# A few throwaway captions so softmax_bank has a partition to normalize over.
BANK_CAPTIONS = [
    "FUNCTION: Catalyzes the hydrolysis of ATP to ADP and phosphate.",
    "FUNCTION: Binds DNA in a sequence-specific manner and represses transcription.",
    "FUNCTION: Transports protons across the inner mitochondrial membrane.",
    "FUNCTION: Serine protease that cleaves peptide bonds C-terminal to arginine.",
    "FUNCTION: Structural component of the bacterial flagellar hook.",
    "FUNCTION: Reduces oxidized glutathione using NADPH as a cofactor.",
    "FUNCTION: Chaperone that assists the folding of nascent polypeptides.",
    "FUNCTION: Ribosomal protein that stabilizes the 30S subunit.",
]
TARGET_PROMPT = "FUNCTION: Catalyzes the hydrolysis of ATP to ADP and phosphate."


def replay_path_logp(generator, seqs, length):
    """Recompute each candidate's sum_pos log p_ESM3(x_pos | committed context)
    from the final sequences alone, replaying the ascending decode order with the
    real generator. Must match what guided_generate accumulated for free."""
    n = len(seqs)
    x, attn = generator.init_sequence(length, n)
    ids = torch.tensor([[int(generator.tok.convert_tokens_to_ids(c)) for c in s]
                        for s in seqs], device=x.device)
    total = torch.zeros(n, device=x.device)
    for i, pos in enumerate(generator.content_positions(length)):
        lp = generator.log_probs(x, attn)[:, pos, :]
        total += lp.gather(1, ids[:, i:i + 1]).squeeze(1)
        x[:, pos] = ids[:, i]
    return total.tolist()


def run_arm(tag, guidance, predictor, generator, args):
    seqs, path_lp = guided_generate(
        guidance, args.length, n=args.n, temperature=args.temperature,
        order="ascending", generator_seed=args.seed, return_path_logp=True)
    assert len(seqs) == args.n and all(len(s) == args.length for s in seqs), \
        [len(s) for s in seqs]

    ref = replay_path_logp(generator, seqs, args.length)
    err = max(abs(a - b) for a, b in zip(path_lp, ref))
    assert err < 1e-2, f"{tag}: path_logp disagrees with replay by {err}"

    filip_lp = _score(predictor, seqs, generator.device)
    cands = _candidate_records(seqs, path_lp, filip_lp, args.max_aa_frac, args.max_run)
    w, waived = _select(cands, args.select_by, guidance.gamma)

    print(f"  [{tag}] gamma={guidance.gamma:g} replay_err={err:.2e} "
          f"winner=cand{w} degenerate={sum(c['degenerate'] for c in cands)}/{args.n} "
          f"waived={waived}")
    for i, c in enumerate(cands):
        mark = "*" if i == w else " "
        print(f"   {mark} c{i} {c['seq']}  path={c['path_logp']:+8.2f} "
              f"({c['path_logp_per_res']:+.3f}/res) filip={c['filip_logp']:+7.3f} "
              f"maxAA={c['max_aa_frac']:.2f} run={c['max_run']}"
              + ("  DEGENERATE" if c["degenerate"] else ""))
    return cands, w, waived


def main():
    cfg = default_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="best_hardneg_epoch49.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--length", type=int, default=30)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=10.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--select-by", default="path_logp")
    ap.add_argument("--max-aa-frac", type=float, default=cfg.guidance.max_aa_frac)
    ap.add_argument("--max-run", type=int, default=cfg.guidance.max_homopolymer_run)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"== load models (device={device}) ==")
    print(f"  generator     {cfg.guidance.generator_type} <- {cfg.guidance.esm3_path}")
    print(f"  classifier    {cfg.model.protein_encoder_type} <- "
          f"{cfg.model.protein_encoder_path}")
    print(f"  retrieval ckpt {args.ckpt}")
    generator = build_generator(cfg, device)
    predictor = build_predictor_from_checkpoint(args.ckpt, cfg, device, mode="softmax_bank")
    tmodel, ttok = encoders.load_text_encoder(cfg.model.text_encoder_path, device,
                                              cfg.data.caption_field_labels)
    print(f"  row_chunk={predictor.row_chunk} bank_chunk={predictor.bank_chunk} "
          f"scale={predictor.scale:.3f}")

    print("== prompt bank + target ==")
    bank = prompt_bank.build_prompt_bank(predictor.filip, tmodel, ttok, BANK_CAPTIONS,
                                         device, cfg.data.max_text_tokens)
    predictor.set_bank(bank["z_t"], bank["mask_t"])
    predictor.set_target(TARGET_PROMPT, tmodel, ttok, cfg.data.max_text_tokens)
    print(f"  bank {tuple(bank['z_t'].shape)}  target {TARGET_PROMPT[:50]}...")

    bridge = TokenizerBridge(generator.tok, predictor.ptok).to(device)
    guided = DEGGuidance(generator, predictor, bridge, args.gamma)
    uncond = DEGGuidance(generator, predictor, bridge, 0.0)

    print(f"== best-of-{args.n}, length {args.length} ==")
    rec = {"accession": "SMOKE1", "prompt": TARGET_PROMPT, "length": args.length,
           "true_seq": "MKTAYIAKQR"[:args.length],
           "n_candidates": args.n, "select_by": args.select_by, "gamma": args.gamma}
    rec["true_logp"] = _score(predictor, [rec["true_seq"]], device)[0]
    for tag, g in (("gen", guided), ("uncond", uncond)):
        cands, w, waived = run_arm(tag, g, predictor, generator, args)
        win = cands[w]
        rec[f"{tag}_seq"] = win["seq"]
        rec[f"{tag}_logp"] = win["filip_logp"]
        rec[f"{tag}_path_logp"] = win["path_logp"]
        rec[f"{tag}_path_logp_per_res"] = win["path_logp_per_res"]
        rec[f"{tag}_sel_idx"] = w
        rec[f"{tag}_n_degenerate"] = sum(c["degenerate"] for c in cands)
        rec[f"{tag}_filter_waived"] = waived
        rec[f"{tag}_candidates"] = cands

    # The arms share a seed, so an identical RNG stream must give identical
    # candidates whenever gamma matches — the guided/unconditional pairing.
    a = guided_generate(DEGGuidance(generator, predictor, bridge, 0.0), args.length,
                        n=args.n, temperature=args.temperature, generator_seed=args.seed)
    assert a == [c["seq"] for c in rec["uncond_candidates"]], "seed pairing broken"
    print("  seed pairing: identical candidate set on re-draw")

    print("== JSONL + merge round-trip ==")
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "shard_000.jsonl"), "w") as f:
            f.write(json.dumps(rec) + "\n")
        merge(types.SimpleNamespace(out_dir=d))
        files = sorted(os.listdir(d))
        assert "generated_all.fasta" in files and "unconditional_all.fasta" in files, files
        n_all = sum(1 for l in open(os.path.join(d, "generated_all.fasta")) if l[0] == ">")
        assert n_all == args.n, n_all
        print(f"  files: {files}")
        print(f"  generated_all.fasta holds all {n_all} candidates")

    print("\nGENERATE_LOCAL_SMOKE_OK")


if __name__ == "__main__":
    main()
