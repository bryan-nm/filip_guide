"""Random protein masking over a [0,1) fraction of positions.

Ported from proteingen's `uniform_mask_noise` / `uniform_time`, adapted to the
ESM-C tokenizer and batched. The FILIP classifier must tolerate the partially
masked sequences the guided sampler presents, so during retrieval training we
mask each protein at a random fraction drawn to match the sampling schedule.

Convention (matches proteingen and the paper): per sequence draw t ~ Unif[t_min,
t_max]; mask each maskable residue independently with prob (1 - t). So t = 1
leaves the sequence fully revealed and t -> 0 fully masks it. Special tokens
(cls/eos/pad/unk/`|`/`<mask>`) are never masked. Masked positions are replaced by
`<mask>` (id 32) and, because `<mask>` is special, drop out of the FILIP valid
mask automatically (see encoders._protein_valid_mask; Decision #2).
"""
from __future__ import annotations

from typing import Optional

import torch


def _special_ids_tensor(tokenizer, device) -> torch.Tensor:
    ids = sorted(int(i) for i in getattr(tokenizer, "all_special_ids", []) if i is not None)
    return torch.tensor(ids, dtype=torch.long, device=device)


def maskable_positions(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                       tokenizer) -> torch.Tensor:
    """Bool [B, L]: True at real-residue positions eligible for masking."""
    specials = _special_ids_tensor(tokenizer, input_ids.device)
    is_special = torch.isin(input_ids, specials)
    return attention_mask.bool() & ~is_special


def sample_mask_fractions(batch_size: int, t_min: float, t_max: float,
                          device, generator: Optional[torch.Generator] = None
                          ) -> torch.Tensor:
    """Per-sequence reveal level t ~ Unif[t_min, t_max]; returns [B]."""
    u = torch.rand(batch_size, device=device, generator=generator)
    return t_min + (t_max - t_min) * u


def mask_protein_batch(
    input_ids: torch.Tensor,          # [B, L]
    attention_mask: torch.Tensor,     # [B, L]
    tokenizer,
    *,
    t_min: float = 0.0,
    t_max: float = 1.0,
    t: Optional[torch.Tensor] = None,   # [B] override reveal levels (else sampled)
    generator: Optional[torch.Generator] = None,
):
    """Mask a random fraction of each sequence's residues -> (masked_ids, applied, t).

    `applied` is the bool [B, L] of positions actually turned into `<mask>`; `t` is
    the per-sequence reveal level used (returned so eval can bucket by it).
    """
    device = input_ids.device
    B = input_ids.size(0)
    if t is None:
        t = sample_mask_fractions(B, t_min, t_max, device, generator)
    maskable = maskable_positions(input_ids, attention_mask, tokenizer)
    r = torch.rand(input_ids.shape, device=device, generator=generator)
    applied = maskable & (r > t[:, None])            # P(mask) = 1 - t
    masked_ids = input_ids.masked_fill(applied, int(tokenizer.mask_token_id))
    return masked_ids, applied, t
