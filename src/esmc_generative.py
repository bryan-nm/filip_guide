"""ESM-C as the unconditional generative prior p(x).

ESM-C is a masked LM, so generation is masked iterative unmasking (the paper's
MLM / AO-ARM case): start from an all-`<mask>` sequence and reveal one position at
a time, each token sampled from the ESM-C distribution at that position given the
current partially-masked sequence. `logits` returns that distribution; the guide
(guide.py) reweights it by the classifier likelihood.

The generator is deliberately thin — ESM-C is frozen and used as-is; nothing here
trains. The candidate token set is the 20 canonical amino acids (specials are
never proposed as residues).
"""
from __future__ import annotations

import torch

from . import encoders

CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"


class ESMCGenerator:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tok = tokenizer
        self.device = device
        self.mask_id = int(tokenizer.mask_token_id)
        self.cls_id = int(tokenizer.cls_token_id)
        self.eos_id = int(tokenizer.eos_token_id)
        self.pad_id = int(tokenizer.pad_token_id)
        # ESM-C token ids of the 20 residues we may propose at a decoded position.
        self.aa_ids = [int(tokenizer.convert_tokens_to_ids(a)) for a in CANONICAL_AA]

    @classmethod
    def load(cls, path: str, device) -> "ESMCGenerator":
        model, tok = encoders.load_protein_encoder(path, device, kind="esmc")
        return cls(model, tok, device)

    @torch.no_grad()
    def logits(self, ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        """ESM-C MLM logits [B, L, V] for the current (partially masked) sequence."""
        logits, _ = encoders.forward_esmc(self.model, ids, attn)
        return logits.float()

    @torch.no_grad()
    def log_probs(self, ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        """Per-position log p(token | x) [B, L, V]."""
        return torch.log_softmax(self.logits(ids, attn), dim=-1)

    def init_sequence(self, length: int, n: int = 1):
        """A fully-masked start: [cls] + length*[mask] + [eos], attn all ones.

        Content positions to decode are 1..length (0 is cls, length+1 is eos).
        """
        L = length + 2
        ids = torch.full((n, L), self.mask_id, dtype=torch.long, device=self.device)
        ids[:, 0] = self.cls_id
        ids[:, -1] = self.eos_id
        attn = torch.ones((n, L), dtype=torch.long, device=self.device)
        return ids, attn

    def content_positions(self, length: int):
        return list(range(1, length + 1))

    def decode(self, ids_row: torch.Tensor) -> str:
        """One id row -> amino-acid string (specials stripped)."""
        toks = self.tok.convert_ids_to_tokens(ids_row.tolist())
        return "".join(t for t in toks if len(t) == 1 and t.isalpha())
