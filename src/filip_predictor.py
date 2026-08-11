"""FILIP co-embedding as a guidance classifier: log p(prompt | sequence).

This is the ProteinGuide predictive model for prompt steering. It wraps a trained
retrieval model + the frozen ESM-C protein encoder and returns, for a batch of
(possibly masked) protein sequences, the scalar log-likelihood of a fixed target
prompt. The guided sampler (guide.py, DEG) consumes this: it reweights ESM-C's
prior by p(prompt | x̃)^gamma at each decoded position.

Two forms of p(y|x) (Decision #1, config-flagged):

  softmax_bank  p(y|x) = softmax_j( scale * s(x, y_j) )  over a reference prompt
                bank, evaluated at the target y. This is the "probability that y
                is the true prompt given x" and matches the InfoNCE training
                objective. `scale = exp(logit_scale)` is the trained temperature.

  sigmoid       p(y|x) = sigmoid(a * s(x, y) + b), independent per prompt. Cheaper
                (no bank), but uncalibrated until (a, b) are fit; defaults a=scale,
                b=0.

s(x, y) is the FILIP mean-of-max-sim, with masked protein positions excluded
(Decision #2), so the likelihood reflects only the committed residues.
"""
from __future__ import annotations

from typing import List, Optional, Union

import torch
import torch.nn.functional as F

from . import encoders
from .losses import filip_score_matrix
from .prompt_bank import load_prompt_bank


class FilipPredictor:
    def __init__(
        self, filip_model, pmodel, ptok, device, *,
        mode: str = "softmax_bank",
        include_masked_in_score: bool = False,
        sigmoid_a: Optional[float] = None,   # default: trained scale
        sigmoid_b: float = 0.0,
    ):
        assert mode in ("softmax_bank", "sigmoid"), mode
        self.filip = filip_model.eval()
        self.pmodel = pmodel
        self.ptok = ptok
        self.device = device
        self.mode = mode
        self.include_masked_in_score = include_masked_in_score
        self.scale = float(filip_model.logit_scale.exp().item())
        self.sigmoid_a = self.scale if sigmoid_a is None else float(sigmoid_a)
        self.sigmoid_b = float(sigmoid_b)

        # Target prompt (set via set_target) and reference bank (softmax mode).
        self._z_t_target: Optional[torch.Tensor] = None   # [1, Lt, D]
        self._mask_t_target: Optional[torch.Tensor] = None
        self._z_t_bank: Optional[torch.Tensor] = None     # [M, Lb, D]
        self._mask_t_bank: Optional[torch.Tensor] = None

    # --- configuration -----------------------------------------------------
    @torch.no_grad()
    def set_target(self, prompt: str, tmodel, ttok, max_text_tokens: int) -> None:
        """Encode the prompt we are conditioning generation on."""
        h_t, mask_t = encoders.encode_text_batch(tmodel, ttok, [prompt], self.device,
                                                 max_text_tokens)
        self._z_t_target = self.filip.text_proj(h_t)       # [1, Lt, D]
        self._mask_t_target = mask_t

    def set_bank(self, z_t_bank: torch.Tensor, mask_t_bank: torch.Tensor) -> None:
        self._z_t_bank = z_t_bank.to(self.device)
        self._mask_t_bank = mask_t_bank.to(self.device)

    def load_bank(self, path: str) -> None:
        bank = load_prompt_bank(path, self.device)
        self.set_bank(bank["z_t"], bank["mask_t"])

    def set_calibration(self, a: float, b: float) -> None:
        self.sigmoid_a, self.sigmoid_b = float(a), float(b)

    # --- scoring -----------------------------------------------------------
    @torch.no_grad()
    def similarity(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """FILIP similarity s(x, target) for a batch of sequences -> [B].

        Also returns (z_p, mask_p) so the bank partition can reuse the encoding.
        """
        assert self._z_t_target is not None, "call set_target(...) first"
        h_p, mask_p = encoders.encode_protein_tokens(
            self.pmodel, self.ptok, input_ids, attention_mask,
            include_masked=self.include_masked_in_score)
        z_p = self.filip.protein_proj(h_p)                 # [B, Lp, D]
        s = filip_score_matrix(z_p, self._z_t_target, mask_p, self._mask_t_target)  # [B,1]
        return s.squeeze(1), z_p, mask_p

    @torch.no_grad()
    def log_prob(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """log p(target prompt | sequence) for each row -> [B]."""
        s_target, z_p, mask_p = self.similarity(input_ids, attention_mask)

        if self.mode == "sigmoid":
            return F.logsigmoid(self.sigmoid_a * s_target + self.sigmoid_b)

        # softmax_bank: normalize scale*s over {target} ∪ bank.
        assert self._z_t_bank is not None, "softmax_bank mode needs a prompt bank"
        s_bank = filip_score_matrix(z_p, self._z_t_bank, mask_p, self._mask_t_bank)  # [B, M]
        logits = self.scale * torch.cat([s_target[:, None], s_bank], dim=1)          # [B, 1+M]
        return logits[:, 0] - torch.logsumexp(logits, dim=1)


def build_predictor_from_checkpoint(
    ckpt_path: str, cfg, device, *, mode: Optional[str] = None,
) -> "FilipPredictor":
    """Convenience: load a retrieval checkpoint + ESM-C and wrap as a predictor.

    (Bank/target are set separately by the caller.)
    """
    from .model import load_retrieval
    filip_model = load_retrieval(ckpt_path, device, cfg, freeze=True)
    pmodel, ptok = encoders.load_protein_encoder(
        cfg.model.protein_encoder_path, device, cfg.model.protein_encoder_type)
    return FilipPredictor(
        filip_model, pmodel, ptok, device,
        mode=mode or cfg.guidance.likelihood_mode,
        include_masked_in_score=cfg.mask.include_masked_in_score)
