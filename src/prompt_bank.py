"""Reference prompt bank for the softmax-over-prompts likelihood (Decision #1).

`p(prompt | sequence)` in softmax_bank mode is normalized over a fixed set of
reference captions: p(y|x) = softmax_j( scale * s(x, y_j) ) evaluated at the true
prompt. This module encodes that bank's per-token text embeddings ONCE (text is
never masked, so the bank is checkpoint-specific but sequence-independent) and
caches them, so the per-candidate partition Z(x) at guidance time is just a
matmul against cached vectors — the extra cost rides on the ESM-C forwards DEG
already does.

The bank is tied to a trained retrieval checkpoint (its text projection head), so
rebuild it whenever the checkpoint changes.
"""
from __future__ import annotations

from typing import List, Optional

import torch

from . import encoders


@torch.no_grad()
def build_prompt_bank(
    filip_model, tmodel, ttok, captions: List[str], device,
    max_text_tokens: int, *, batch_size: int = 64, dedup: bool = True,
) -> dict:
    """Encode reference captions to padded projected per-token embeddings.

    Returns {"z_t": [M,L,D], "mask_t": [M,L] bool, "captions": [M]}, padded to the
    bank-wide max token length. Kept on CPU for storage; move to device at use.
    """
    if dedup:
        seen, uniq = set(), []
        for c in captions:
            if c not in seen:
                seen.add(c); uniq.append(c)
        captions = uniq

    z_chunks: List[torch.Tensor] = []
    m_chunks: List[torch.Tensor] = []
    for s in range(0, len(captions), batch_size):
        chunk = captions[s:s + batch_size]
        h_t, mask_t = encoders.encode_text_batch(tmodel, ttok, chunk, device, max_text_tokens)
        z_t = filip_model.text_proj(h_t)                       # [b, L, D]
        z_chunks.append(z_t.cpu()); m_chunks.append(mask_t.cpu())

    L = max(z.size(1) for z in z_chunks)
    D = z_chunks[0].size(2)
    M = sum(z.size(0) for z in z_chunks)
    z_t = torch.zeros(M, L, D)
    mask_t = torch.zeros(M, L, dtype=torch.bool)
    r = 0
    for z, m in zip(z_chunks, m_chunks):
        b, li = z.size(0), z.size(1)
        z_t[r:r + b, :li] = z
        mask_t[r:r + b, :li] = m
        r += b
    return {"z_t": z_t, "mask_t": mask_t, "captions": captions}


def save_prompt_bank(bank: dict, path: str) -> None:
    torch.save(bank, path)


def load_prompt_bank(path: str, device: Optional[torch.device] = None) -> dict:
    bank = torch.load(path, map_location="cpu")
    if device is not None:
        bank = {**bank, "z_t": bank["z_t"].to(device), "mask_t": bank["mask_t"].to(device)}
    return bank
