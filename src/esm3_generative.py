"""ESM3-sm as the unconditional generative prior p(x).

ESM3 is an actual generative protein model (the ProteinGuide paper's unconditional
sequence generator), unlike ESM-C which is a representation model that folds poorly
when used to design sequences from scratch. It shares ESM-C's EsmSequenceTokenizer,
so the FILIP classifier and the tokenizer bridge are unchanged — only the prior
forward differs. We use ESM3's sequence track only (structure/function tracks left
masked), which is the sequence-only marginal p(seq).

Interface matches ESMCGenerator so DEGGuidance/sample.py are generator-agnostic.

Local weights: `ESM3.from_pretrained` resolves weights via `data_root("esm3")`,
which otherwise downloads the (gated) HF repo. We point `data_root` at the local
snapshot dir (layout: <path>/data/weights/esm3_sm_open_v1.pth + the structure/
function decoder weights), so nothing is downloaded. The checkpoint is bf16; we
cast to fp32 for a consistent dtype (ESM3's forward mixes fp32 plddt features).
"""
from __future__ import annotations

from pathlib import Path

import torch

from .encoders import _silence_stdout
from .esmc_generative import CANONICAL_AA, ESMCGenerator


class ESM3Generator:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tok = tokenizer
        self.device = device
        self.mask_id = int(tokenizer.mask_token_id)
        self.cls_id = int(tokenizer.cls_token_id)
        self.eos_id = int(tokenizer.eos_token_id)
        self.pad_id = int(tokenizer.pad_token_id)
        self.aa_ids = [int(tokenizer.convert_tokens_to_ids(a)) for a in CANONICAL_AA]

    @classmethod
    def load(cls, path: str, device) -> "ESM3Generator":
        # Point esm's weight resolver at the local snapshot (no HF download).
        import esm.pretrained as _P
        _P.data_root = lambda model=None: Path(path)
        try:
            import esm.utils.constants.esm3 as _C
            _C.data_root = lambda model=None: Path(path)
        except Exception:
            pass
        with _silence_stdout():
            from esm.models.esm3 import ESM3
            from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
            model = ESM3.from_pretrained("esm3_sm_open_v1", device=device).eval().to(torch.float32)
        for p in model.parameters():
            p.requires_grad_(False)
        return cls(model, EsmSequenceTokenizer(), device)

    @torch.no_grad()
    def logits(self, ids: torch.Tensor, attn: torch.Tensor = None) -> torch.Tensor:
        """ESM3 sequence-track logits [B, L, V] for the current (partial) sequence.

        `attn` is accepted for interface parity but unused: generation batches are
        uniform-length (no padding), and ESM3 takes `sequence_id` for padding, not
        an attention mask.
        """
        out = self.model.forward(sequence_tokens=ids)
        return out.sequence_logits.float()

    @torch.no_grad()
    def log_probs(self, ids: torch.Tensor, attn: torch.Tensor = None) -> torch.Tensor:
        return torch.log_softmax(self.logits(ids), dim=-1)

    def init_sequence(self, length: int, n: int = 1):
        L = length + 2
        ids = torch.full((n, L), self.mask_id, dtype=torch.long, device=self.device)
        ids[:, 0] = self.cls_id
        ids[:, -1] = self.eos_id
        attn = torch.ones((n, L), dtype=torch.long, device=self.device)
        return ids, attn

    def content_positions(self, length: int):
        return list(range(1, length + 1))

    def decode(self, ids_row: torch.Tensor) -> str:
        toks = self.tok.convert_ids_to_tokens(ids_row.tolist())
        return "".join(t for t in toks if len(t) == 1 and t.isalpha())


def build_generator(cfg, device):
    """Factory: return the generator selected by cfg.guidance.generator_type."""
    if cfg.guidance.generator_type == "esm3":
        return ESM3Generator.load(cfg.guidance.esm3_path, device)
    if cfg.guidance.generator_type == "esmc":
        return ESMCGenerator.load(cfg.guidance.generator_path, device)
    raise ValueError(f"unknown generator_type {cfg.guidance.generator_type!r} "
                     f"(expected 'esm3' or 'esmc')")
