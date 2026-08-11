"""Tokenizer bridge: map generator (ESM-C) token ids to classifier (AMPLIFY) ids.

Cross-model guidance runs the ESM-C generator in ESM-C's vocabulary but scores
candidates with the AMPLIFY-FILIP classifier, which has its own vocabulary. Both
are residue alphabets (one token per residue, cls/eos framing, a `<mask>` token),
so the mapping is an id->id lookup: residues map by letter, specials map by role
(cls->cls, eos->eos, pad->pad, mask->mask). A revealed ESM-C residue becomes the
same AMPLIFY residue; an ESM-C `<mask>` becomes the AMPLIFY `<mask>` (which the
classifier then excludes from the FILIP score, Decision #2). Positions and length
are preserved, so no re-alignment is needed.

Any ESM-C token with no AMPLIFY counterpart (rare non-standard residues) maps to
AMPLIFY `<unk>`; those positions are dropped from the score anyway if unk is a
special.
"""
from __future__ import annotations

import torch

# Standard + common non-standard residue letters to map by identity.
_RESIDUES = "ACDEFGHIKLMNPQRSTVWYBUXZOJ"
_ROLES = ("cls_token_id", "eos_token_id", "pad_token_id", "unk_token_id",
          "bos_token_id", "sep_token_id", "mask_token_id")


class TokenizerBridge:
    def __init__(self, src_tok, dst_tok):
        self.src_tok, self.dst_tok = src_tok, dst_tok
        src_vocab = int(max(src_tok.get_vocab().values())) + 1
        fallback = dst_tok.unk_token_id
        if fallback is None:
            fallback = dst_tok.mask_token_id or 0
        m = torch.full((src_vocab,), int(fallback), dtype=torch.long)

        def valid(i):
            return i is not None and int(i) >= 0

        # Residues by letter.
        for aa in _RESIDUES:
            s = src_tok.convert_tokens_to_ids(aa)
            d = dst_tok.convert_tokens_to_ids(aa)
            if valid(s) and valid(d) and d != dst_tok.unk_token_id and s < src_vocab:
                m[int(s)] = int(d)

        # Specials by role (overrides any letter collision).
        for role in _ROLES:
            s = getattr(src_tok, role, None)
            d = getattr(dst_tok, role, None)
            if valid(s) and valid(d) and int(s) < src_vocab:
                m[int(s)] = int(d)

        self.register = m

    def to(self, device) -> "TokenizerBridge":
        self.register = self.register.to(device)
        return self

    def convert(self, src_ids: torch.Tensor) -> torch.Tensor:
        """[*, L] ESM-C ids -> [*, L] AMPLIFY ids (same shape, positions preserved)."""
        return self.register.to(src_ids.device)[src_ids]
