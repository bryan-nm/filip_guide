"""Retrieval evaluation: encode helper, recall@k / mAP, recall-vs-mask curve.

`encode_pair_batch` is the single encode path shared by training and eval, so the
masking applied at train time is reproduced exactly at eval. The recall-vs-mask
sweep is the metric specific to this project: it checks the classifier degrades
gracefully as more positions are masked (its guidance-time operating regime).
"""
from __future__ import annotations

from typing import List, Optional, Union

import torch

from . import encoders, masking
from .losses import filip_score_matrix_chunked


@torch.no_grad()
def encode_pair_batch(
    pmodel, ptok, tmodel, ttok, filip_model,
    proteins: List[str], texts: List[str], device,
    max_protein_tokens: int, max_text_tokens: int,
    *,
    reveal: Union[None, float, torch.Tensor] = None,   # None => unmasked; else t reveal level
    include_masked_in_score: bool = False,
    generator: Optional[torch.Generator] = None,
):
    """Encode a batch to projected per-token embeddings + valid masks.

    Returns z_p [B,Lp,D], mask_p [B,Lp], z_t [B,Lt,D], mask_t [B,Lt] (+ the reveal
    levels t used). Encoders run frozen under no_grad; only projection is applied
    (the caller keeps grads if training).
    """
    # Text (never masked)
    h_t, mask_t = encoders.encode_text_batch(tmodel, ttok, texts, device, max_text_tokens)

    # Protein: tokenize, optionally mask, encode live
    enc = ptok(proteins, padding=True, truncation=True,
               max_length=max_protein_tokens, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    t_used = None
    if reveal is not None:
        B = ids.size(0)
        if isinstance(reveal, torch.Tensor):
            t = reveal.to(device)
        else:
            t = torch.full((B,), float(reveal), device=device)
        ids, _, t_used = masking.mask_protein_batch(ids, attn, ptok, t=t, generator=generator)
    h_p, mask_p = encoders.encode_protein_tokens(
        pmodel, ptok, ids, attn, include_masked=include_masked_in_score)

    z_p, z_t = filip_model.project(h_p, h_t)
    return z_p, mask_p, z_t, mask_t, t_used


def _relevance(acc_ids: torch.Tensor, txt_ids: torch.Tensor) -> torch.Tensor:
    """[N, N] bool: column j relevant to row i if same protein OR same caption."""
    same_acc = acc_ids[:, None] == acc_ids[None, :]
    same_txt = txt_ids[:, None] == txt_ids[None, :]
    return same_acc | same_txt


def _recall_at_k(scores: torch.Tensor, rel: torch.Tensor, ks=(1, 5, 10)) -> dict:
    """Mean recall@k: fraction of rows with >=1 relevant item in the top k."""
    order = scores.argsort(dim=1, descending=True)
    rel_sorted = torch.gather(rel, 1, order)
    out = {}
    for k in ks:
        kk = min(k, scores.size(1))
        out[k] = (rel_sorted[:, :kk].any(dim=1)).float().mean().item()
    return out


def _map(scores: torch.Tensor, rel: torch.Tensor) -> float:
    """Mean average precision over rows (ProTrek-style)."""
    order = scores.argsort(dim=1, descending=True)
    rel_sorted = torch.gather(rel, 1, order).float()
    csum = rel_sorted.cumsum(dim=1)
    ranks = torch.arange(1, scores.size(1) + 1, device=scores.device).float()[None, :]
    precision_at = csum / ranks
    n_rel = rel_sorted.sum(dim=1).clamp_min(1)
    ap = (precision_at * rel_sorted).sum(dim=1) / n_rel
    return ap.mean().item()


@torch.no_grad()
def retrieval_metrics(z_p, mask_p, z_t, mask_t, acc_ids, txt_ids,
                      chunk_rows: int = 16) -> dict:
    """R@1/5/10 + mAP in both directions, plus collapse diagnostics.

    Collapse tells: `pos`/`neg` are the mean FILIP similarity of true pairs vs
    true-negative pairs, and `gap = pos - neg` is the discriminability the whole
    model rests on (gap -> 0 with pos high == the space collapsed). `unif_p`/
    `unif_t` are Wang-Isola token uniformity per modality (more negative = more
    spread; near 0 = crowded).
    """
    from .losses import token_uniformity_loss, pool_normalize
    S = filip_score_matrix_chunked(z_p, z_t, mask_p, mask_t, chunk_rows)   # [N_p, N_t]
    rel = _relevance(acc_ids, txt_ids)
    p2t = _recall_at_k(S, rel)
    t2p = _recall_at_k(S.t(), rel.t())
    pos = S.diagonal().mean().item()
    neg = S[~rel].mean().item() if (~rel).any() else 0.0

    # Position-pooled geometry: matched/mismatched cross-modal cosine, and the
    # intra-modal anisotropy (mean pairwise cosine within a modality — the direct
    # "is the cone collapsing" read; near 1.0 = collapsed) + modality-gap L2.
    up, ut = pool_normalize(z_p, mask_p), pool_normalize(z_t, mask_t)      # [N, D]
    Cx = up @ ut.t()
    N = up.size(0)
    off = ~torch.eye(N, dtype=torch.bool, device=up.device)
    return {
        "p2t_R@1": p2t[1], "p2t_R@5": p2t[5], "p2t_R@10": p2t[10], "p2t_mAP": _map(S, rel),
        "t2p_R@1": t2p[1], "t2p_R@5": t2p[5], "t2p_R@10": t2p[10], "t2p_mAP": _map(S.t(), rel.t()),
        "pos": pos, "neg": neg, "gap": pos - neg,
        "cos_pos": Cx.diagonal().mean().item(),
        "cos_neg": Cx[~rel].mean().item() if (~rel).any() else 0.0,
        "aniso_p": (up @ up.t())[off].mean().item() if N > 1 else 0.0,
        "aniso_t": (ut @ ut.t())[off].mean().item() if N > 1 else 0.0,
        "modality_gap": (up.mean(0) - ut.mean(0)).norm().item(),
        "unif_p": token_uniformity_loss(z_p, mask_p).item(),
        "unif_t": token_uniformity_loss(z_t, mask_t).item(),
        "n": S.size(0),
    }


@torch.no_grad()
def evaluate_pairs(pmodel, ptok, tmodel, ttok, filip_model, batch, device,
                   max_protein_tokens, max_text_tokens, *,
                   reveal=None, include_masked_in_score=False,
                   generator: Optional[torch.Generator] = None) -> dict:
    """Encode one evaluation batch (whole split fits in memory here) + metrics."""
    z_p, mask_p, z_t, mask_t, _ = encode_pair_batch(
        pmodel, ptok, tmodel, ttok, filip_model, batch["proteins"], batch["texts"],
        device, max_protein_tokens, max_text_tokens,
        reveal=reveal, include_masked_in_score=include_masked_in_score, generator=generator)
    return retrieval_metrics(z_p, mask_p, z_t, mask_t, batch["acc_ids"].to(device),
                             batch["txt_ids"].to(device))


def _pad_cat(tensors: List[torch.Tensor], masks: List[torch.Tensor]):
    """Pad a list of [b, L, D] (and [b, L] masks) to global max L and concat."""
    L = max(t.size(1) for t in tensors)
    D = tensors[0].size(2)
    zt, mt = [], []
    for t, m in zip(tensors, masks):
        if t.size(1) < L:
            pad_t = t.new_zeros(t.size(0), L - t.size(1), D)
            pad_m = m.new_zeros(m.size(0), L - m.size(1))
            t = torch.cat([t, pad_t], dim=1)
            m = torch.cat([m, pad_m], dim=1)
        zt.append(t); mt.append(m)
    return torch.cat(zt, 0), torch.cat(mt, 0)


@torch.no_grad()
def evaluate_loader(pmodel, ptok, tmodel, ttok, filip_model,
                    proteins: List[str], texts: List[str],
                    acc_ids: torch.Tensor, txt_ids: torch.Tensor, device,
                    max_protein_tokens, max_text_tokens, *,
                    chunk: int = 16, reveal=None,
                    include_masked_in_score: bool = False) -> dict:
    """Encode a (possibly large) eval set in chunks, then compute metrics once.

    Used for the per-epoch rank-0 eval on a val subset that is too large to encode
    in a single batch.
    """
    zp_list, mp_list, zt_list, mt_list = [], [], [], []
    for s in range(0, len(proteins), chunk):
        z_p, mask_p, z_t, mask_t, _ = encode_pair_batch(
            pmodel, ptok, tmodel, ttok, filip_model,
            proteins[s:s + chunk], texts[s:s + chunk], device,
            max_protein_tokens, max_text_tokens, reveal=reveal,
            include_masked_in_score=include_masked_in_score)
        zp_list.append(z_p); mp_list.append(mask_p)
        zt_list.append(z_t); mt_list.append(mask_t)
    z_p, mask_p = _pad_cat(zp_list, mp_list)
    z_t, mask_t = _pad_cat(zt_list, mt_list)
    return retrieval_metrics(z_p, mask_p, z_t, mask_t, acc_ids.to(device), txt_ids.to(device))


@torch.no_grad()
def recall_vs_mask(pmodel, ptok, tmodel, ttok, filip_model, batch, device,
                   max_protein_tokens, max_text_tokens, *,
                   reveal_grid=(1.0, 0.75, 0.5, 0.25, 0.0),
                   include_masked_in_score=False, chunk: int = 16,
                   max_eval: int = 256) -> List[dict]:
    """p2t R@1 / mAP as a function of reveal level t (1.0 = fully revealed).

    The guidance-time sanity check: retrieval should weaken smoothly as t -> 0,
    not collapse. Encoded in chunks (capped at `max_eval` pairs) so it stays cheap
    on a large val subset.
    """
    n = min(len(batch["proteins"]), max_eval)
    proteins, texts = batch["proteins"][:n], batch["texts"][:n]
    acc_ids, txt_ids = batch["acc_ids"][:n], batch["txt_ids"][:n]
    rows = []
    for t in reveal_grid:
        m = evaluate_loader(pmodel, ptok, tmodel, ttok, filip_model, proteins, texts,
                            acc_ids, txt_ids, device, max_protein_tokens, max_text_tokens,
                            chunk=chunk, reveal=t, include_masked_in_score=include_masked_in_score)
        rows.append({"reveal_t": t, "p2t_R@1": m["p2t_R@1"], "p2t_mAP": m["p2t_mAP"],
                     "t2p_R@1": m["t2p_R@1"]})
    return rows
