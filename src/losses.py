"""FILIP late-interaction losses + per-token uniformity + reconstruction.

Trimmed from mini-embed-filip to the single-device path this project needs (no
distributed subgroup gather, no hard-decoy mining). The load-bearing function —
`filip_score_matrix` — is unchanged: mask-awareness (Decision #2) lives entirely
in the caller's masks. Passing a protein valid-mask that excludes `<mask>`
positions (which encoders._protein_valid_mask does by default) makes the score
ignore not-yet-decoded positions, with no change to the arithmetic here.

Inputs are L2-normalized per token where applicable (projection-head outputs are;
expansion-head outputs are not, by design).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


# Finite "−inf" for masked entries: safe across fp32/bf16/autocast (true -inf can
# poison reductions), and well below scale*FILIP (|logit| <= ~100).
_NEG = -1e4


# ---------------------------------------------------------------------------
# FILIP score (per-pair scalar similarity in [-1, 1])
# ---------------------------------------------------------------------------
def filip_score_matrix(
    z_p: torch.Tensor,        # [B_p, L_p, D] L2-normalized per token
    z_t: torch.Tensor,        # [B_t, L_t, D] L2-normalized per token
    mask_p: torch.Tensor,     # [B_p, L_p] bool; True = valid (revealed, non-special)
    mask_t: torch.Tensor,     # [B_t, L_t] bool; True = valid token
    neg_inf: float = _NEG,
) -> torch.Tensor:
    """Returns [B_p, B_t] FILIP score: 0.5 * (p2t max-sim + t2p max-sim).

    A protein whose valid mask is all-False (fully masked sequence) yields a score
    of 0 in both directions here — guidance is only meaningful once >=1 position
    is revealed, matching the decoding process.
    """
    sim = torch.einsum("bld,cmd->bclm", z_p, z_t)  # [B_p, B_t, L_p, L_t]

    mp = mask_p[:, None, :, None]                  # [B_p, 1, L_p, 1]
    mt = mask_t[None, :, None, :]                  # [1, B_t, 1, L_t]
    sim = sim.masked_fill(~mp, neg_inf)
    sim = sim.masked_fill(~mt, neg_inf)

    # p -> t: for each protein position, best matching text token
    max_per_p = sim.max(dim=3).values              # [B_p, B_t, L_p]
    max_per_p = max_per_p.masked_fill(~mp.squeeze(3), 0.0)
    n_valid_p = mask_p.sum(dim=1).clamp_min(1).to(max_per_p.dtype)  # [B_p]
    score_p2t = max_per_p.sum(dim=2) / n_valid_p[:, None]           # [B_p, B_t]

    # t -> p: for each text token, best matching protein position
    max_per_t = sim.max(dim=2).values              # [B_p, B_t, L_t]
    max_per_t = max_per_t.masked_fill(~mt.squeeze(2), 0.0)
    n_valid_t = mask_t.sum(dim=1).clamp_min(1).to(max_per_t.dtype)  # [B_t]
    score_t2p = max_per_t.sum(dim=2) / n_valid_t[None, :]           # [B_p, B_t]

    return 0.5 * (score_p2t + score_t2p)


def filip_score_matrix_chunked(
    z_p: torch.Tensor, z_t: torch.Tensor,
    mask_p: torch.Tensor, mask_t: torch.Tensor,
    chunk_rows: int = 16,
) -> torch.Tensor:
    """Chunked over the protein axis: bounds the [chunk, B_t, L_p, L_t] transient."""
    rows = []
    for s in range(0, z_p.size(0), chunk_rows):
        e = min(s + chunk_rows, z_p.size(0))
        rows.append(filip_score_matrix(z_p[s:e], z_t, mask_p[s:e], mask_t))
    return torch.cat(rows, dim=0)


def mask_false_negatives(
    logits: torch.Tensor,            # [R, C] scaled similarity logits
    target: torch.Tensor,           # [R] index of each row's positive column
    groups_row: Optional[torch.Tensor] = None,   # [R] accession id per row
    groups_col: Optional[torch.Tensor] = None,   # [C] accession id per column
    text_row: Optional[torch.Tensor] = None,     # [R] caption id per row
    text_col: Optional[torch.Tensor] = None,     # [C] caption id per column
) -> torch.Tensor:
    """Mask same-protein or same-caption non-target columns out of the denominator.

    A byte-identical caption repeated across a different protein, or (on a
    multi-caption corpus) another caption of the same protein, is a valid positive
    the InfoNCE must not push away. No-op when neither pair of ids is supplied.
    """
    same = None
    if groups_row is not None and groups_col is not None:
        same = groups_row[:, None] == groups_col[None, :]
    if text_row is not None and text_col is not None:
        same_text = text_row[:, None] == text_col[None, :]
        same = same_text if same is None else (same | same_text)
    if same is None:
        return logits
    is_target = torch.zeros_like(same)
    is_target[torch.arange(logits.size(0), device=logits.device), target] = True
    return logits.masked_fill(same & ~is_target, _NEG)


# ---------------------------------------------------------------------------
# Token-level uniformity (Wang & Isola 2020) + reconstruction
# ---------------------------------------------------------------------------
def token_uniformity_loss(
    z: torch.Tensor, mask: torch.Tensor, t: float = 2.0, max_tokens: int = 4096,
) -> torch.Tensor:
    """log E_{i!=j} exp(-t ||z_i - z_j||^2) over valid tokens (subsampled)."""
    flat = z[mask]
    n = flat.size(0)
    if n < 2:
        return torch.zeros((), device=z.device, dtype=z.dtype)
    if n > max_tokens:
        idx = torch.randperm(n, device=z.device)[:max_tokens]
        flat = flat[idx]
        n = max_tokens
    sim = flat @ flat.t()
    sq_dists = 2.0 - 2.0 * sim
    off_diag = ~torch.eye(n, device=z.device, dtype=torch.bool)
    return torch.logsumexp(-t * sq_dists[off_diag], dim=0) - math.log(n * (n - 1))


def reconstruction_loss(h_hat: torch.Tensor, h: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
    """Masked per-token MSE between expand(project(h)) and h."""
    diff = ((h_hat - h) ** 2).mean(dim=-1)         # [B, L]
    valid = mask.to(diff.dtype)
    return (diff * valid).sum() / valid.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# Phase-R1 / Phase-R2 composites
# ---------------------------------------------------------------------------
def phase_r1_loss(
    out: dict, h_p, h_t, mask_p, mask_t, *,
    uniformity_weight: float, uniformity_t: float, recon_weight: float,
) -> dict:
    """Phase R1 warmup: maximize positive-pair FILIP + token uniformity + recon."""
    z_p, z_t = out["z_p"], out["z_t"]
    filip_pos = filip_score_matrix(z_p, z_t, mask_p, mask_t).diagonal().mean()
    l_align = 1.0 - filip_pos
    l_unif = 0.5 * (token_uniformity_loss(z_p, mask_p, uniformity_t)
                    + token_uniformity_loss(z_t, mask_t, uniformity_t))
    l_recon = 0.5 * (reconstruction_loss(out["h_p_hat"], h_p, mask_p)
                     + reconstruction_loss(out["h_t_hat"], h_t, mask_t))
    total = l_align + uniformity_weight * l_unif + recon_weight * l_recon
    return {"loss": total, "align": l_align, "unif": l_unif, "recon": l_recon,
            "nce": torch.zeros((), device=z_p.device),
            "acc": torch.zeros((), device=z_p.device), "filip_pos": filip_pos}


def phase_r2_loss(
    out: dict, h_p, h_t, mask_p, mask_t, logit_scale, *,
    align_aux_weight: float, recon_weight: float,
    uniformity_weight: float = 0.0, uniformity_t: float = 2.0, chunk_rows: int = 0,
    groups: Optional[torch.Tensor] = None, text_groups: Optional[torch.Tensor] = None,
) -> dict:
    """Phase R2: FILIP symmetric InfoNCE + small align aux + recon (+ optional unif)."""
    z_p, z_t = out["z_p"], out["z_t"]
    if chunk_rows > 0:
        filip_mat = filip_score_matrix_chunked(z_p, z_t, mask_p, mask_t, chunk_rows)
    else:
        filip_mat = filip_score_matrix(z_p, z_t, mask_p, mask_t)

    scale = logit_scale.exp()
    target = torch.arange(filip_mat.size(0), device=filip_mat.device)
    logits_pt = mask_false_negatives(scale * filip_mat, target, groups, groups,
                                     text_groups, text_groups)
    logits_tp = mask_false_negatives((scale * filip_mat).t(), target, groups, groups,
                                     text_groups, text_groups)
    l_nce = 0.5 * (F.cross_entropy(logits_pt, target) + F.cross_entropy(logits_tp, target))

    with torch.no_grad():
        acc = 0.5 * ((logits_pt.argmax(-1) == target).float().mean()
                     + (logits_tp.argmax(-1) == target).float().mean())

    filip_pos = filip_mat.diagonal().mean()
    l_align = 1.0 - filip_pos
    l_recon = 0.5 * (reconstruction_loss(out["h_p_hat"], h_p, mask_p)
                     + reconstruction_loss(out["h_t_hat"], h_t, mask_t))
    if uniformity_weight > 0:
        l_unif = 0.5 * (token_uniformity_loss(z_p, mask_p, uniformity_t)
                        + token_uniformity_loss(z_t, mask_t, uniformity_t))
    else:
        l_unif = torch.zeros((), device=z_p.device)

    total = (l_nce + align_aux_weight * l_align + recon_weight * l_recon
             + uniformity_weight * l_unif)
    return {"loss": total, "align": l_align, "unif": l_unif, "recon": l_recon,
            "nce": l_nce, "acc": acc, "filip_pos": filip_pos}


def phase_r2_loss_grouped(
    out: dict, h_p, h_t, mask_p, mask_t, logit_scale, *,
    z_p_all: torch.Tensor,        # [G, L_p, D] gathered across the subgroup
    z_t_all: torch.Tensor,        # [G, L_t, D]
    mask_p_all: torch.Tensor,     # [G, L_p]
    mask_t_all: torch.Tensor,     # [G, L_t]
    local_offset: int,            # row index of this rank's first sample within [0, G)
    align_aux_weight: float, recon_weight: float,
    uniformity_weight: float = 0.0, uniformity_t: float = 2.0, chunk_rows: int = 0,
    groups: Optional[torch.Tensor] = None,       # [B] accession id per local anchor
    groups_all: Optional[torch.Tensor] = None,   # [G] accession id per gathered column
    text_groups: Optional[torch.Tensor] = None,      # [B] caption id per local anchor
    text_groups_all: Optional[torch.Tensor] = None,  # [G] caption id per gathered column
) -> dict:
    """Distributed Phase-R2 InfoNCE over a bounded subgroup of negatives.

    Each rank scores its local B anchors against the gathered G columns; targets
    are `local_offset .. local_offset+B`. Only the local slice of `z_*_all` carries
    gradient (see dist.grouped_all_gather), so this is the standard gather-with-
    grad contrastive loss. At world/group size 1 it reduces exactly to in-batch
    InfoNCE over B — the single-device path uses it unchanged.
    """
    z_p, z_t = out["z_p"], out["z_t"]
    h_p_hat, h_t_hat = out["h_p_hat"], out["h_t_hat"]
    B = z_p.size(0)
    scale = logit_scale.exp()

    builder = (lambda a, b, ma, mb: filip_score_matrix_chunked(a, b, ma, mb, chunk_rows)) \
        if chunk_rows > 0 else filip_score_matrix

    mat_p2t = builder(z_p, z_t_all, mask_p, mask_t_all)            # [B, G]
    mat_t2p = builder(z_p_all, z_t, mask_p_all, mask_t).t()        # [B, G]

    target = torch.arange(B, device=z_p.device) + local_offset
    logits_pt = mask_false_negatives(scale * mat_p2t, target, groups, groups_all,
                                     text_groups, text_groups_all)
    logits_tp = mask_false_negatives(scale * mat_t2p, target, groups, groups_all,
                                     text_groups, text_groups_all)
    l_nce = 0.5 * (F.cross_entropy(logits_pt, target) + F.cross_entropy(logits_tp, target))

    with torch.no_grad():
        acc = 0.5 * ((logits_pt.argmax(-1) == target).float().mean()
                     + (logits_tp.argmax(-1) == target).float().mean())

    filip_pos = mat_p2t.gather(1, target[:, None]).squeeze(1).mean()
    l_align = 1.0 - filip_pos
    l_recon = 0.5 * (reconstruction_loss(h_p_hat, h_p, mask_p)
                     + reconstruction_loss(h_t_hat, h_t, mask_t))
    if uniformity_weight > 0:
        l_unif = 0.5 * (token_uniformity_loss(z_p, mask_p, uniformity_t)
                        + token_uniformity_loss(z_t, mask_t, uniformity_t))
    else:
        l_unif = torch.zeros((), device=z_p.device)

    total = (l_nce + align_aux_weight * l_align + recon_weight * l_recon
             + uniformity_weight * l_unif)
    return {"loss": total, "align": l_align, "unif": l_unif, "recon": l_recon,
            "nce": l_nce, "acc": acc, "filip_pos": filip_pos}
