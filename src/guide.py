"""Discrete Exact Guidance (DEG) for the MLM/AO-ARM case (ProteinGuide).

At the position being decoded, enumerate the candidate residues, and for each one
combine the generator's unconditional prior with the classifier likelihood via
Bayes in log-space:

    log p(x̃ | x, prompt)  ∝  log p(x̃ | x)  +  γ · log p(prompt | x̃)
         guided score            ESM-C prior       FILIP classifier

The prior for all candidates at a position comes from ONE ESM-C forward. Each
candidate x̃ (the sequence with that residue filled in, the rest still masked) is
bridged to the classifier's vocabulary and scored; the classifier excludes masked
positions, so it reads only the committed residues. γ=0 recovers unconditional
generation (the classifier is never called).

Exact DEG evaluates all C candidates (here the 20 residues) per position — the
paper's exact-guidance regime, which it used for ESM-C. Cost is one classifier
forward of batch C per decoded position (batched), on top of the prior forward.
"""
from __future__ import annotations

from typing import List, Optional

import torch

from .bridge import TokenizerBridge
from .esmc_generative import ESMCGenerator
from .filip_predictor import FilipPredictor


class DEGGuidance:
    def __init__(self, generator: ESMCGenerator, predictor: FilipPredictor,
                 bridge: TokenizerBridge, gamma: float,
                 candidate_ids: Optional[List[int]] = None):
        self.generator = generator
        self.predictor = predictor
        self.bridge = bridge
        self.gamma = float(gamma)
        # ESM-C ids of the residues we propose (default: the 20 canonical AAs).
        ids = candidate_ids if candidate_ids is not None else generator.aa_ids
        self.candidate_ids = torch.tensor(ids, dtype=torch.long, device=generator.device)

    @torch.no_grad()
    def guided_position_logprobs(self, x: torch.Tensor, attn: torch.Tensor,
                                 pos: int) -> torch.Tensor:
        """Unnormalized guided log-probs over the candidate residues at `pos`.

        x, attn: [B, L] current ESM-C ids + attention. Returns [B, C].
        """
        B, L = x.shape
        cand = self.candidate_ids
        C = cand.numel()

        # Unconditional prior over candidates at this position (one ESM-C forward).
        prior = self.generator.log_probs(x, attn)[:, pos, :][:, cand]   # [B, C]
        if self.gamma == 0.0:
            return prior

        # Build the C candidate sequences per row (fill `pos` with each residue).
        xC = x[:, None, :].expand(B, C, L).clone()
        xC[:, :, pos] = cand[None, :]
        xC = xC.reshape(B * C, L)
        attnC = attn[:, None, :].expand(B, C, L).reshape(B * C, L)

        # Bridge ESM-C ids -> classifier (AMPLIFY) ids and score log p(prompt | x̃).
        cls_ids = self.bridge.convert(xC)
        logp_y = self.predictor.log_prob(cls_ids, attnC).reshape(B, C)   # [B, C]

        return prior + self.gamma * logp_y
