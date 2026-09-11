"""Batch prompt-guided generation over held-out SwissProt entries, for structural
evaluation (ESMfold + TM-align of generated vs. true structure).

For each of N entries we take its caption as the prompt and generate a sequence of
the SAME length as the true sequence (so the structures are directly comparable),
emitting the true sequence, the guided sample, and (optionally) an unconditional
(gamma=0) baseline. Embarrassingly parallel: entries are sharded across MPI ranks
(no process group needed); each rank writes a JSONL shard. A separate `--merge`
pass (single process) concatenates the shards and writes paired FASTAs
(true/generated/unconditional, matching headers) for folding + alignment.

Best-of-n (`--n-candidates`): the n candidates for an entry are drawn as ONE
batched decode, and the winner is picked by `--select-by` (default `path_logp`,
the generator's own AO-ARM log-likelihood, which comes free out of the guided
score). Selection is applied SYMMETRICALLY to the guided and unconditional arms
with identical n, temperature and seed, so the arms stay matched; every candidate
and all of its metrics are kept in the JSONL, so the run can be re-selected
offline under a different metric without regenerating.

    # under mpiexec:
    python -m src.generate_dataset --ckpt <clf.pt> --out-dir runs/gen100 \
        --n-entries 100 --gamma 1.0 --likelihood sigmoid --device xpu \
        --n-candidates 10 --select-by path_logp
    # then, single process:
    python -m src.generate_dataset --merge --out-dir runs/gen100
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from typing import Dict, List, Tuple

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


def _degeneracy(seq: str) -> Tuple[float, int]:
    """(largest single-residue fraction, longest homopolymer run) for `seq`.

    Likelihood-based best-of-n rewards low-complexity sequences — ESM3 at
    temperature 1.0 occasionally emits poly-A / poly-K stretches, and those score
    high — so these two cheap statistics gate candidates before selection.
    """
    if not seq:
        return 1.0, 0
    frac = max(Counter(seq).values()) / len(seq)
    run = best = 1
    for a, b in zip(seq, seq[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return frac, best


def _candidate_records(seqs: List[str], path_logps: List[float], filip_logps: List[float],
                       max_aa_frac: float, max_run: int) -> List[Dict]:
    """One record per candidate, with every metric we might later select on."""
    out = []
    for seq, plp, flp in zip(seqs, path_logps, filip_logps):
        frac, run = _degeneracy(seq)
        out.append({
            "seq": seq,
            "path_logp": plp,
            "path_logp_per_res": plp / max(len(seq), 1),
            "filip_logp": flp,
            "max_aa_frac": frac,
            "max_run": run,
            "degenerate": bool(frac > max_aa_frac or run > max_run),
        })
    return out


def _selector(select_by: str, gamma: float):
    """The scalar a candidate is ranked by. Shared by selection and the merge
    report, so the two can never drift apart."""
    if select_by == "path_logp":
        return lambda c: c["path_logp"]
    if select_by == "filip_logp":
        return lambda c: c["filip_logp"]
    if select_by == "joint_logp":     # the sampler's own objective; == path at gamma=0
        return lambda c: c["path_logp"] + gamma * c["filip_logp"]
    if select_by == "first":
        return None
    raise ValueError(f"unknown --select-by {select_by!r}")


def _select(cands: List[Dict], select_by: str, gamma: float) -> Tuple[int, bool]:
    """Winning candidate index, and whether the degeneracy filter had to be waived.

    Degenerate candidates are excluded; if EVERY candidate is degenerate the filter
    is waived rather than dropping the entry (the record flags it, so those entries
    can be excluded from the analysis instead of silently biasing it).
    """
    keep = [i for i, c in enumerate(cands) if not c["degenerate"]]
    waived = not keep
    if waived:
        keep = list(range(len(cands)))
    key = _selector(select_by, gamma)
    if key is None:               # "first"
        return keep[0], waived
    return max(keep, key=lambda i: key(cands[i])), waived


def generate(args) -> None:
    cfg = default_cfg()
    env = init_distributed(args.device, init_pg=False)   # rank/world only, no process group
    device = env.device
    if env.is_main:
        print(f"[gen] world={env.world_size} device={device} generator={cfg.guidance.generator_type} "
              f"gamma={args.gamma} likelihood={args.likelihood} n={args.n_entries} "
              f"uncond={args.uncond}", flush=True)
        print(f"[gen] best-of-n: n_candidates={args.n_candidates} select_by={args.select_by} "
              f"filter=(max_aa_frac={args.max_aa_frac}, max_run={args.max_run}) "
              f"row_chunk={cfg.guidance.classifier_row_chunk} "
              f"bank_chunk={cfg.guidance.bank_chunk}", flush=True)
        if args.select_by == "filip_logp":
            print("[gen] WARNING --select-by filip_logp is circular: it is the guidance "
                  "objective, and applying it to the gamma=0 arm (needed to keep the arms "
                  "matched) conditions the control. Diagnostic use only.", flush=True)
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
            rec = {"accession": acc, "prompt": prompt, "length": L, "true_seq": true_seq,
                   "n_candidates": args.n_candidates, "select_by": args.select_by,
                   "gamma": args.gamma,
                   "true_logp": _score(predictor, [true_seq], device)[0]}

            # Both arms: same n, same temperature, same seed (so the candidate sets
            # are paired), same selector. gamma is the ONLY difference.
            for tag, g in (("gen", guided), ("uncond", uncond)):
                if g is None:
                    continue
                seqs, path_lp = guided_generate(
                    g, L, n=args.n_candidates, temperature=args.temperature,
                    order=args.order, generator_seed=seed, return_path_logp=True)
                filip_lp = _score(predictor, seqs, device)
                cands = _candidate_records(seqs, path_lp, filip_lp,
                                           args.max_aa_frac, args.max_run)
                w, waived = _select(cands, args.select_by, g.gamma)
                win = cands[w]
                rec[f"{tag}_seq"] = win["seq"]
                rec[f"{tag}_logp"] = win["filip_logp"]          # FILIP score of the winner
                rec[f"{tag}_path_logp"] = win["path_logp"]
                rec[f"{tag}_path_logp_per_res"] = win["path_logp_per_res"]
                rec[f"{tag}_sel_idx"] = w
                rec[f"{tag}_n_degenerate"] = sum(c["degenerate"] for c in cands)
                rec[f"{tag}_filter_waived"] = waived
                rec[f"{tag}_candidates"] = cands                # keep ALL of them

            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[rank {env.rank}] {acc} L={L} true_lp={rec['true_logp']:+.2f} "
                  f"gen_lp={rec['gen_logp']:+.2f} gen_path={rec['gen_path_logp_per_res']:+.3f}/res "
                  f"(cand {rec['gen_sel_idx']}/{args.n_candidates}, "
                  f"{rec['gen_n_degenerate']} degen)"
                  + (f" uncond_lp={rec['uncond_logp']:+.2f} "
                     f"uncond_path={rec['uncond_path_logp_per_res']:+.3f}/res"
                     if uncond else ""), flush=True)
    barrier()
    if env.is_main:
        print(f"[gen] done; shards in {args.out_dir}. Merge with --merge.", flush=True)


def _fasta(records, key, path):
    with open(path, "w") as f:
        for r in records:
            if key in r and r[key]:
                f.write(f">{r['accession']}\n{r[key]}\n")


def _fasta_all_candidates(records, tag, path):
    """Every candidate, headers `<accession>__c<i>`, so the full set can be folded
    later (to check the selector against pLDDT / TM-score) without regenerating."""
    n = 0
    with open(path, "w") as f:
        for r in records:
            for i, c in enumerate(r.get(f"{tag}_candidates", [])):
                f.write(f">{r['accession']}__c{i}\n{c['seq']}\n")
                n += 1
    return n


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
    has_uncond = any("uncond_seq" in r for r in records)
    if has_uncond:
        _fasta(records, "uncond_seq", os.path.join(args.out_dir, "unconditional.fasta"))

    import statistics as st
    tl = [r["true_logp"] for r in records]
    arms = ["gen"] + (["uncond"] if has_uncond else [])
    print(f"[merge] {len(records)} entries -> results.jsonl + true.fasta + generated.fasta"
          + (" + unconditional.fasta" if has_uncond else ""))

    # Winners (what gets folded).
    print(f"[merge] mean logp(prompt|seq): true={st.mean(tl):+.3f} "
          + " ".join(f"{a}={st.mean([r[f'{a}_logp'] for r in records]):+.3f}" for a in arms))

    # Best-of-n accounting: candidate 0 is the n=1 run this replaces, so the delta
    # isolates what selection bought. Reported for the SELECTOR (path logp, per
    # residue) and for FILIP, which selection is deliberately blind to.
    n_cand = max((r.get("n_candidates", 1) for r in records), default=1)
    if n_cand > 1 and all(f"{arms[0]}_candidates" in r for r in records):
        for a in arms:
            sel_p = st.mean([r[f"{a}_path_logp_per_res"] for r in records])
            first_p = st.mean([r[f"{a}_candidates"][0]["path_logp"] / max(r["length"], 1)
                               for r in records])
            sel_f = st.mean([r[f"{a}_logp"] for r in records])
            first_f = st.mean([r[f"{a}_candidates"][0]["filip_logp"] for r in records])
            degen = sum(r[f"{a}_n_degenerate"] for r in records)
            waived = sum(r[f"{a}_filter_waived"] for r in records)
            total = sum(len(r[f"{a}_candidates"]) for r in records)
            # How often the degeneracy filter vetoed the candidate the selector
            # ranked first. Without this the delta above is ambiguous: a negative
            # delta means the filter did its job, not that selection failed.
            over = 0
            for r in records:
                key = _selector(r.get("select_by", "path_logp"),
                                r.get("gamma", 0.0) if a == "gen" else 0.0)
                if key is None:
                    continue
                cs = r[f"{a}_candidates"]
                if max(range(len(cs)), key=lambda i: key(cs[i])) != r[f"{a}_sel_idx"]:
                    over += 1
            print(f"[merge] {a}: path_logp/res {first_p:+.3f} (cand 0) -> {sel_p:+.3f} "
                  f"(best-of-{n_cand}, delta {sel_p - first_p:+.3f}); "
                  f"filip_logp {first_f:+.3f} -> {sel_f:+.3f}; "
                  f"degenerate {degen}/{total} candidates, filter vetoed the top-ranked "
                  f"candidate on {over}/{len(records)} entries, waived on {waived}")
            stem = {"gen": "generated", "uncond": "unconditional"}[a]
            path = os.path.join(args.out_dir, f"{stem}_all.fasta")
            print(f"[merge] {a}: {_fasta_all_candidates(records, a, path)} candidates -> "
                  f"{os.path.basename(path)}")


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
    ap.add_argument("--n-candidates", type=int, default=cfg.guidance.n_candidates,
                    help="best-of-n: candidates per entry, drawn as one batched decode")
    ap.add_argument("--select-by", default=cfg.guidance.select_by,
                    choices=["path_logp", "filip_logp", "joint_logp", "first"],
                    help="winner metric; path_logp (ESM3 AO-ARM log-likelihood) is free "
                         "and independent of both the guidance objective and the fold eval")
    ap.add_argument("--max-aa-frac", type=float, default=cfg.guidance.max_aa_frac,
                    help="degeneracy filter: reject candidates whose most common residue "
                         "exceeds this fraction")
    ap.add_argument("--max-run", type=int, default=cfg.guidance.max_homopolymer_run,
                    help="degeneracy filter: reject candidates with a longer homopolymer run")
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
        if args.n_candidates < 1:
            ap.error("--n-candidates must be >= 1")
        generate(args)


if __name__ == "__main__":
    main()
