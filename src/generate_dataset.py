"""Batch prompt-guided generation over held-out SwissProt entries, for structural
evaluation (ESMfold + TM-align of generated vs. true structure).

For each of N entries we take its caption as the prompt and generate a sequence of
the SAME length as the true sequence (so the structures are directly comparable),
emitting the true sequence, the guided sample, and (optionally) an unconditional
(gamma=0) baseline. Embarrassingly parallel: entries are sharded across MPI ranks
(no process group needed); each rank writes a JSONL shard. A separate `--merge`
pass (single process) concatenates the shards and writes paired FASTAs
(true/generated/unconditional, matching headers) for folding + alignment.

    # under mpiexec:
    python -m src.generate_dataset --ckpt <clf.pt> --out-dir runs/gen100 \
        --n-entries 100 --gamma 1.0 --likelihood sigmoid --device xpu
    # then, single process:
    python -m src.generate_dataset --merge --out-dir runs/gen100
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import List

import torch

from config import default_cfg
from src import encoders, prompt_bank
from src.bridge import TokenizerBridge
from src.data import build_splits, load_pairs
from src.dist import barrier, init_distributed
from src.esm3_generative import build_generator
from src.filip_predictor import build_predictor_from_checkpoint
from src.guide import DEGGuidance
from src.model import load_retrieval
from src.sample import guided_generate


def select_entries(cfg, split: str, n: int, min_len: int, max_len: int):
    """Deterministic held-out entry list: (accession, prompt, true_seq), identical
    on every rank. Filtered to [min_len, max_len] residues, deduped by accession."""
    pairs = load_pairs(cfg.data.csv_path, id_col=cfg.data.csv_id_col,
                       protein_col=cfg.data.csv_protein_col, text_col=cfg.data.csv_text_col,
                       pfam_col=cfg.data.csv_pfam_col)
    splits = build_splits(pairs, cfg.data.splits, cfg.data.seed)
    seen, out = set(), []
    for i in splits[split]:
        p = pairs[i]
        if not (min_len <= len(p.protein) <= max_len) or p.uid in seen:
            continue
        seen.add(p.uid)
        out.append((p.uid, p.text, p.protein))
        if len(out) >= n:
            break
    return out


@torch.no_grad()
def _pick_device(name: str):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_bank(args) -> None:
    """Single-process: encode a reference prompt bank (train captions) with the
    classifier's text head and cache it, for softmax_bank guidance. Run ONCE
    before the sharded generation."""
    cfg = default_cfg()
    device = _pick_device(args.device)
    filip_model = load_retrieval(args.ckpt, device, cfg, freeze=True)
    tmodel, ttok = encoders.load_text_encoder(cfg.model.text_encoder_path, device,
                                              cfg.data.caption_field_labels)
    pairs = load_pairs(cfg.data.csv_path, id_col=cfg.data.csv_id_col,
                       protein_col=cfg.data.csv_protein_col, text_col=cfg.data.csv_text_col,
                       pfam_col=cfg.data.csv_pfam_col)
    splits = build_splits(pairs, cfg.data.splits, cfg.data.seed)
    caps = [pairs[i].text for i in splits["train"]]      # train captions as the reference set
    bank = prompt_bank.build_prompt_bank(filip_model, tmodel, ttok, caps[:args.bank_size * 3],
                                         device, cfg.data.max_text_tokens)
    # build_prompt_bank dedups; keep at most bank_size.
    if bank["z_t"].size(0) > args.bank_size:
        bank = {"z_t": bank["z_t"][:args.bank_size], "mask_t": bank["mask_t"][:args.bank_size],
                "captions": bank["captions"][:args.bank_size]}
    os.makedirs(os.path.dirname(args.prompt_bank) or ".", exist_ok=True)
    prompt_bank.save_prompt_bank(bank, args.prompt_bank)
    print(f"[bank] {bank['z_t'].size(0)} reference prompts -> {args.prompt_bank}", flush=True)


@torch.no_grad()
def _score(predictor, seqs: List[str], device) -> List[float]:
    if not seqs:
        return []
    enc = predictor.ptok(seqs, padding=True, truncation=True, max_length=1024, return_tensors="pt")
    return predictor.log_prob(enc["input_ids"].to(device), enc["attention_mask"].to(device)).tolist()


def generate(args) -> None:
    cfg = default_cfg()
    env = init_distributed(args.device, init_pg=False)   # rank/world only, no process group
    device = env.device
    if env.is_main:
        print(f"[gen] world={env.world_size} device={device} generator={cfg.guidance.generator_type} "
              f"gamma={args.gamma} likelihood={args.likelihood} n={args.n_entries} "
              f"uncond={args.uncond}", flush=True)
        os.makedirs(args.out_dir, exist_ok=True)
    barrier()

    entries = select_entries(cfg, args.split, args.n_entries, args.min_len, args.max_len)
    mine = entries[env.rank::env.world_size]
    if env.is_main:
        print(f"[gen] {len(entries)} entries selected from split '{args.split}' "
              f"(len {args.min_len}-{args.max_len}); ~{len(mine)}/rank", flush=True)

    generator = build_generator(cfg, device)
    predictor = build_predictor_from_checkpoint(args.ckpt, cfg, device, mode=args.likelihood)
    tmodel, ttok = encoders.load_text_encoder(cfg.model.text_encoder_path, device,
                                              cfg.data.caption_field_labels)
    if args.likelihood == "softmax_bank":
        predictor.load_bank(args.prompt_bank)
    bridge = TokenizerBridge(generator.tok, predictor.ptok).to(device)

    guided = DEGGuidance(generator, predictor, bridge, args.gamma)
    uncond = DEGGuidance(generator, predictor, bridge, 0.0) if args.uncond else None

    shard_path = os.path.join(args.out_dir, f"shard_{env.rank:03d}.jsonl")
    with open(shard_path, "w") as f:
        for k, (acc, prompt, true_seq) in enumerate(mine):
            L = len(true_seq)
            predictor.set_target(prompt, tmodel, ttok, cfg.data.max_text_tokens)
            seed = args.seed + env.rank * 100000 + k
            gen_seq = guided_generate(guided, L, n=1, temperature=args.temperature,
                                      order=args.order, generator_seed=seed)[0]
            rec = {"accession": acc, "prompt": prompt, "length": L,
                   "true_seq": true_seq, "gen_seq": gen_seq}
            score_seqs = [true_seq, gen_seq]
            if uncond is not None:
                rec["uncond_seq"] = guided_generate(uncond, L, n=1, temperature=args.temperature,
                                                    order=args.order, generator_seed=seed)[0]
                score_seqs.append(rec["uncond_seq"])
            lp = _score(predictor, score_seqs, device)
            rec["true_logp"], rec["gen_logp"] = lp[0], lp[1]
            if uncond is not None:
                rec["uncond_logp"] = lp[2]
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[rank {env.rank}] {acc} L={L} "
                  f"true_lp={rec['true_logp']:+.2f} gen_lp={rec['gen_logp']:+.2f}"
                  + (f" uncond_lp={rec['uncond_logp']:+.2f}" if uncond else ""), flush=True)
    barrier()
    if env.is_main:
        print(f"[gen] done; shards in {args.out_dir}. Merge with --merge.", flush=True)


def _fasta(records, key, path):
    with open(path, "w") as f:
        for r in records:
            if key in r and r[key]:
                f.write(f">{r['accession']}\n{r[key]}\n")


def merge(args) -> None:
    records = []
    for sp in sorted(glob.glob(os.path.join(args.out_dir, "shard_*.jsonl"))):
        with open(sp) as f:
            records += [json.loads(line) for line in f if line.strip()]
    records.sort(key=lambda r: r["accession"])
    with open(os.path.join(args.out_dir, "results.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    _fasta(records, "true_seq", os.path.join(args.out_dir, "true.fasta"))
    _fasta(records, "gen_seq", os.path.join(args.out_dir, "generated.fasta"))
    if any("uncond_seq" in r for r in records):
        _fasta(records, "uncond_seq", os.path.join(args.out_dir, "unconditional.fasta"))

    import statistics as st
    gl = [r["gen_logp"] for r in records]
    tl = [r["true_logp"] for r in records]
    print(f"[merge] {len(records)} entries -> results.jsonl + true.fasta + generated.fasta"
          + (" + unconditional.fasta" if any('uncond_seq' in r for r in records) else ""))
    print(f"[merge] mean logp(prompt|seq): true={st.mean(tl):+.3f} gen={st.mean(gl):+.3f}"
          + (f" uncond={st.mean([r['uncond_logp'] for r in records]):+.3f}"
             if any('uncond_seq' in r for r in records) else ""))


def main() -> None:
    cfg = default_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true", help="single-process: merge shards -> fastas")
    ap.add_argument("--build-bank", action="store_true",
                    help="single-process: build the softmax_bank reference prompt bank")
    ap.add_argument("--bank-size", type=int, default=2000)
    ap.add_argument("--out-dir", default=None, help="required for generation/merge")
    ap.add_argument("--ckpt", help="trained FILIP classifier checkpoint")
    ap.add_argument("--n-entries", type=int, default=100)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--min-len", type=int, default=30)
    ap.add_argument("--max-len", type=int, default=400)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--likelihood", default="sigmoid", choices=["sigmoid", "softmax_bank"])
    ap.add_argument("--prompt-bank", default=cfg.guidance.prompt_bank_path)
    ap.add_argument("--temperature", type=float, default=cfg.guidance.sample_temperature)
    ap.add_argument("--order", default="ascending", choices=["ascending", "random"])
    ap.add_argument("--no-uncond", dest="uncond", action="store_false",
                    help="skip the gamma=0 baseline")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.merge:
        if not args.out_dir:
            ap.error("--out-dir is required for --merge")
        merge(args)
    elif args.build_bank:
        if not args.ckpt:
            ap.error("--ckpt is required for --build-bank")
        build_bank(args)
    else:
        if not args.ckpt or not args.out_dir:
            ap.error("--ckpt and --out-dir are required for generation")
        generate(args)


if __name__ == "__main__":
    main()
