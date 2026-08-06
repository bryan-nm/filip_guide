"""Two-phase mask-aware FILIP retrieval training — distributed-capable.

    # Aurora (16 nodes x 12 ranks): see train_retrieval.pbs
    # Local single-device smoke:
    python -m src.train_retrieval --subset-size 256 --overfit --batch-size 8 \
        --phase2-epochs 4

Frozen ESM-C (protein) and BioLinkBERT (text) encoders run under no_grad each
step; only the projection/expansion heads (+ temperature) train. The protein side
is masked at t ~ Unif[mask_t_min, mask_t_max] every step, and `<mask>` positions
drop out of the FILIP valid mask (Decision #2), so the classifier learns to score
partially-revealed sequences.

Distribution mirrors mini-embed-filip: NOT DDP — the contrastive loss uses
logit_scale and subgroup-gathered embeddings outside the module forward, so we
broadcast initial weights and average gradients manually (see dist.py). Phase R2
gathers negatives within a subgroup of `group_size` ranks (grouped_all_gather);
at world_size 1 this reduces exactly to in-batch InfoNCE, so the same file runs
locally on MPS/CPU. ESM-C is encoded LIVE every step (masking makes its output
uncacheable), which is the dominant cost.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config import Cfg, default_cfg
from src import encoders, evaluate, losses, masking
from src.data import Pair, RawPairsDataset, build_splits, load_pairs, raw_collate
from src.dist import (average_gradients, barrier, broadcast_parameters, cleanup,
                      grouped_all_gather, grouped_all_gather_ids, init_distributed,
                      all_reduce_sum)
from src.model import build_retrieval


# ---------------------------------------------------------------------------
def autocast_ctx(device: torch.device, enabled: bool):
    if enabled and device.type in ("cuda", "xpu"):
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


def cosine_warmup_lr(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


# ---------------------------------------------------------------------------
# Encode (frozen encoders under no_grad; heads run with grad in the caller)
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_frozen(pmodel, ptok, tmodel, ttok, proteins, texts, device, cfg: Cfg,
                  *, reveal: Optional[torch.Tensor], generator=None):
    h_t, mask_t = encoders.encode_text_batch(tmodel, ttok, texts, device,
                                             cfg.data.max_text_tokens)
    enc = ptok(proteins, padding=True, truncation=True,
               max_length=cfg.data.max_protein_tokens, return_tensors="pt")
    ids, attn = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    if reveal is not None:
        ids, _, _ = masking.mask_protein_batch(ids, attn, ptok, t=reveal, generator=generator)
    h_p, mask_p = encoders.encode_protein_tokens(
        pmodel, ptok, ids, attn, include_masked=cfg.mask.include_masked_in_score)
    return h_p, mask_p, h_t, mask_t


@torch.no_grad()
def compute_feature_means(pmodel, ptok, tmodel, ttok, pairs: List[Pair], env, cfg: Cfg):
    """Corpus-mean token vectors for centering, computed UNDER the training mask
    schedule.

    Critical for our masked setup: ESM-C's dominant direction shifts sharply as a
    sequence is masked (raw pairwise-cosine ~0.61 unmasked -> ~0.85 at 70% masked),
    so a mean estimated on unmasked sequences under-corrects the masked features
    the projection head actually sees at train time (leaves ~0.52 residual
    anisotropy -> FILIP collapse). We therefore mask each sample with the SAME
    t ~ Unif[mask_t_min, mask_t_max] schedule as training and average over the
    revealed positions (the ones that feed FILIP). Sharded across ranks then
    SUM-all-reduced.
    """
    sub = pairs[:cfg.retrieval.means_sample][env.rank::max(env.world_size, 1)]
    device = env.device
    reveal_range = (cfg.mask.mask_t_min, cfg.mask.mask_t_max) if cfg.mask.enabled else None
    sum_p = torch.zeros(cfg.model.protein_hidden, device=device)
    sum_t = torch.zeros(cfg.model.text_hidden, device=device)
    cnt = torch.zeros(2, device=device)
    for s in range(0, len(sub), 16):
        chunk = sub[s:s + 16]
        reveal = (masking.sample_mask_fractions(len(chunk), *reveal_range, device)
                  if reveal_range else None)
        h_p, mask_p, h_t, mask_t = encode_frozen(
            pmodel, ptok, tmodel, ttok, [p.protein for p in chunk],
            [p.text for p in chunk], device, cfg, reveal=reveal)
        sum_p += h_p[mask_p].sum(0).float(); cnt[0] += int(mask_p.sum())
        sum_t += h_t[mask_t].sum(0).float(); cnt[1] += int(mask_t.sum())
    all_reduce_sum(sum_p); all_reduce_sum(sum_t); all_reduce_sum(cnt)
    return sum_p / cnt[0].clamp_min(1), sum_t / cnt[1].clamp_min(1)


def train(cfg: Cfg, args) -> None:
    torch.manual_seed(args.seed)
    env = init_distributed(args.device, group_size=cfg.retrieval.group_size)
    device = env.device
    if env.is_main:
        print(f"[setup] world={env.world_size} group_size={env.group_size} device={device}",
              flush=True)

    # Every rank loads the CSV and builds identical (seeded, deterministic) splits,
    # so no split file / broadcast is needed to stay in sync.
    pairs = load_pairs(cfg.data.csv_path,
                       id_col=cfg.data.csv_id_col, protein_col=cfg.data.csv_protein_col,
                       text_col=cfg.data.csv_text_col, pfam_col=cfg.data.csv_pfam_col,
                       subset_size=args.subset_size)
    splits = build_splits(pairs, cfg.data.splits, cfg.data.seed)
    if env.is_main:
        print(f"[data] pairs={len(pairs)} train={len(splits['train'])} "
              f"val={len(splits['val'])} test={len(splits['test'])}", flush=True)

    train_ds = RawPairsDataset(pairs, splits["train"])
    if env.distributed:
        sampler = DistributedSampler(train_ds, num_replicas=env.world_size, rank=env.rank,
                                     shuffle=True, seed=cfg.data.seed, drop_last=True)
        train_loader = DataLoader(train_ds, batch_size=cfg.retrieval.batch_size,
                                  sampler=sampler, num_workers=cfg.retrieval.num_workers,
                                  collate_fn=raw_collate, drop_last=True)
    else:
        sampler = None
        train_loader = DataLoader(train_ds, batch_size=cfg.retrieval.batch_size, shuffle=True,
                                  num_workers=cfg.retrieval.num_workers, collate_fn=raw_collate,
                                  drop_last=len(train_ds) >= 2 * cfg.retrieval.batch_size)
    if len(train_loader) == 0:
        raise RuntimeError(f"train_loader empty (per-rank train≈"
                           f"{len(train_ds)//max(env.world_size,1)}, bs={cfg.retrieval.batch_size}).")

    # Eval set (rank 0 only): val split, or the train split for the overfit smoke.
    eval_idx = (splits["train"] if args.overfit else splits["val"])[:cfg.retrieval.val_subset]
    eval_ds = RawPairsDataset(pairs, eval_idx)
    eval_pack = raw_collate([eval_ds[i] for i in range(len(eval_ds))])

    if env.is_main:
        print("[load] encoders (frozen)...", flush=True)
    tmodel, ttok = encoders.load_text_encoder(cfg.model.text_encoder_path, device,
                                              cfg.data.caption_field_labels)
    pmodel, ptok = encoders.load_protein_encoder(cfg.model.protein_encoder_path, device)
    model = build_retrieval(cfg).to(device)
    core = model
    if env.distributed:
        broadcast_parameters(model)
    if env.is_main:
        print(f"[model] trainable params: "
              f"{sum(p.numel() for p in model.parameters()):,}", flush=True)

    ckpt_dir = Path(cfg.retrieval.ckpt_dir)
    total_epochs = cfg.retrieval.phase1_epochs + cfg.retrieval.phase2_epochs
    steps_per_epoch = len(train_loader)
    total_steps = total_epochs * steps_per_epoch
    warmup = max(int(cfg.retrieval.warmup_frac * total_steps), 1)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.retrieval.lr,
                            weight_decay=cfg.retrieval.weight_decay)

    log: List[dict] = []
    gstep = 0
    start_epoch = 0
    resume_path = args.resume
    if resume_path == "auto":                       # pick latest epochNN.pt, if any
        found = sorted(ckpt_dir.glob("epoch*.pt")) if ckpt_dir.is_dir() else []
        resume_path = str(found[-1]) if found else None
        if env.is_main:
            print(f"[resume] auto -> {resume_path or 'no checkpoint, fresh start'}", flush=True)
    if resume_path:
        ck = torch.load(resume_path, map_location=device)
        core.load_state_dict(ck["model_state"])
        if "optimizer_state" in ck:
            opt.load_state_dict(ck["optimizer_state"])
        start_epoch = ck["epoch"] + 1
        gstep = ck.get("global_step", start_epoch * steps_per_epoch)
        log = ck.get("train_log", [])
        if env.is_main:
            print(f"[resume] {resume_path} -> epoch {start_epoch} (gstep={gstep})", flush=True)
    elif args.center:
        if env.is_main:
            print("[center] computing feature means (sharded)...", flush=True)
        mean_p, mean_t = compute_feature_means(
            pmodel, ptok, tmodel, ttok, [pairs[i] for i in splits["train"]], env, cfg)
        core.set_feature_means(mean_p=mean_p, mean_t=mean_t)
    elif env.is_main:
        print("[center] WARNING: --no-center; features un-centered, FILIP may collapse.",
              flush=True)

    reveal_range = (cfg.mask.mask_t_min, cfg.mask.mask_t_max) if cfg.mask.enabled else None

    for epoch in range(start_epoch, total_epochs):
        in_phase1 = epoch < cfg.retrieval.phase1_epochs
        phase = "R1-warm" if in_phase1 else "R2-NCE"
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        for it, batch in enumerate(train_loader):
            lr = cosine_warmup_lr(gstep, total_steps, warmup, cfg.retrieval.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)

            reveal = (masking.sample_mask_fractions(
                len(batch["proteins"]), *reveal_range, device) if reveal_range else None)
            with autocast_ctx(device, cfg.retrieval.use_autocast):
                h_p, mask_p, h_t, mask_t = encode_frozen(
                    pmodel, ptok, tmodel, ttok, batch["proteins"], batch["texts"],
                    device, cfg, reveal=reveal)
                out = model(h_p, h_t)                       # heads WITH grad
                h_p_c, h_t_c = core.center(h_p, h_t)        # centered recon targets
                if in_phase1:
                    res = losses.phase_r1_loss(
                        out, h_p_c, h_t_c, mask_p, mask_t,
                        uniformity_weight=cfg.retrieval.phase1_uniformity_weight,
                        uniformity_t=cfg.retrieval.uniformity_t,
                        recon_weight=cfg.retrieval.recon_weight)
                else:
                    z_p_all = grouped_all_gather(out["z_p"], env)
                    z_t_all = grouped_all_gather(out["z_t"], env)
                    mp_all = grouped_all_gather(mask_p.to(out["z_p"].dtype), env) > 0.5
                    mt_all = grouped_all_gather(mask_t.to(out["z_t"].dtype), env) > 0.5
                    local_offset = env.group_rank * out["z_p"].size(0)
                    groups_local = batch["acc_ids"].to(device)
                    text_local = batch["txt_ids"].to(device)
                    res = losses.phase_r2_loss_grouped(
                        out, h_p_c, h_t_c, mask_p, mask_t, core.logit_scale,
                        z_p_all=z_p_all, z_t_all=z_t_all, mask_p_all=mp_all, mask_t_all=mt_all,
                        local_offset=local_offset,
                        align_aux_weight=cfg.retrieval.align_aux_weight,
                        recon_weight=cfg.retrieval.recon_weight,
                        uniformity_weight=cfg.retrieval.r2_uniformity_weight,
                        uniformity_t=cfg.retrieval.uniformity_t,
                        chunk_rows=cfg.retrieval.filip_chunk_rows,
                        groups=groups_local, groups_all=grouped_all_gather_ids(groups_local, env),
                        text_groups=text_local, text_groups_all=grouped_all_gather_ids(text_local, env))

            res["loss"].backward()
            average_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.retrieval.grad_clip)
            opt.step()
            core.clamp_temperature()
            gstep += 1

            if env.is_main and ((it + 1) % cfg.retrieval.log_every == 0 or it == 0):
                print(f"[{phase}] ep={epoch} step={it+1}/{steps_per_epoch} lr={lr:.2e} "
                      f"loss={res['loss'].item():.4f} nce={res['nce'].item():.4f} "
                      f"acc={res['acc'].item():.3f} "
                      f"filip[p={res['filip_pos'].item():.3f} n={res['filip_neg'].item():.3f}] "
                      f"cos[p={res['cos_pos'].item():.3f} n={res['cos_neg'].item():.3f}] "
                      f"unif={res['unif'].item():.3f} "
                      f"recon={res['recon'].item():.4f} tau={core.logit_scale.exp().item():.2f}",
                      flush=True)

        if env.is_main:
            print(f"[{phase}] ep={epoch} done in {time.time()-t0:.1f}s", flush=True)
            model.eval()
            m = evaluate.evaluate_loader(
                pmodel, ptok, tmodel, ttok, core, eval_pack["proteins"], eval_pack["texts"],
                eval_pack["acc_ids"], eval_pack["txt_ids"], device,
                cfg.data.max_protein_tokens, cfg.data.max_text_tokens,
                chunk=cfg.retrieval.batch_size, reveal=None,
                include_masked_in_score=cfg.mask.include_masked_in_score)
            print(f"[val] ep={epoch} p2t R@1={m['p2t_R@1']:.3f} R@5={m['p2t_R@5']:.3f} "
                  f"mAP={m['p2t_mAP']:.3f} t2p R@1={m['t2p_R@1']:.3f} (n={m['n']})", flush=True)
            print(f"[geo] ep={epoch} filip[p={m['pos']:.3f} n={m['neg']:.3f} gap={m['gap']:.3f}] "
                  f"cos[p={m['cos_pos']:.3f} n={m['cos_neg']:.3f}] "
                  f"aniso[p={m['aniso_p']:.3f} t={m['aniso_t']:.3f}] "
                  f"mgap={m['modality_gap']:.3f} unif[p={m['unif_p']:.2f} t={m['unif_t']:.2f}]",
                  flush=True)
            log.append({"epoch": epoch, "phase": phase, **m})
            if args.save:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"epoch": epoch, "global_step": gstep, "model_state": core.state_dict(),
                            "optimizer_state": opt.state_dict(), "train_log": log},
                           ckpt_dir / f"epoch{epoch:02d}.pt")
                with open(ckpt_dir / "train_log.json", "w") as f:
                    json.dump(log, f, indent=2)
                print(f"[ckpt] saved {ckpt_dir / f'epoch{epoch:02d}.pt'}", flush=True)
        barrier()

    if env.is_main:
        print("[eval] recall vs reveal level t:", flush=True)
        for row in evaluate.recall_vs_mask(
                pmodel, ptok, tmodel, ttok, core, eval_pack, device,
                cfg.data.max_protein_tokens, cfg.data.max_text_tokens,
                include_masked_in_score=cfg.mask.include_masked_in_score):
            print(f"    t={row['reveal_t']:.2f} p2t R@1={row['p2t_R@1']:.3f} "
                  f"mAP={row['p2t_mAP']:.3f}", flush=True)
        print("[train] done", flush=True)
    cleanup()


def main() -> None:
    cfg = default_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ckpt-dir", default=cfg.retrieval.ckpt_dir)
    ap.add_argument("--subset-size", type=int, default=0, help="0 = full corpus")
    ap.add_argument("--seed", type=int, default=cfg.data.seed)
    ap.add_argument("--overfit", action="store_true", help="eval on the train split")
    ap.add_argument("--center", dest="center", action="store_true", default=True)
    ap.add_argument("--no-center", dest="center", action="store_false")
    ap.add_argument("--no-save", dest="save", action="store_false")
    ap.add_argument("--resume", default=None, help="path to an epochNN.pt to resume from")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--group-size", type=int, default=None)
    ap.add_argument("--phase1-epochs", type=int, default=None)
    ap.add_argument("--phase2-epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--val-subset", type=int, default=None)
    ap.add_argument("--filip-chunk-rows", type=int, default=None)
    ap.add_argument("--phase1-uniformity-weight", type=float, default=None)
    ap.add_argument("--r2-uniformity-weight", type=float, default=None)
    ap.add_argument("--align-aux-weight", type=float, default=None)
    args = ap.parse_args()

    r = cfg.retrieval
    for src, dst in [("batch_size", "batch_size"), ("group_size", "group_size"),
                     ("phase1_epochs", "phase1_epochs"), ("phase2_epochs", "phase2_epochs"),
                     ("lr", "lr"), ("val_subset", "val_subset"),
                     ("filip_chunk_rows", "filip_chunk_rows"),
                     ("phase1_uniformity_weight", "phase1_uniformity_weight"),
                     ("r2_uniformity_weight", "r2_uniformity_weight"),
                     ("align_aux_weight", "align_aux_weight")]:
        v = getattr(args, src)
        if v is not None:
            setattr(r, dst, v)
    r.ckpt_dir = args.ckpt_dir
    train(cfg, args)


if __name__ == "__main__":
    main()
