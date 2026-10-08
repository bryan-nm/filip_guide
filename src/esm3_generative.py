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

from contextlib import nullcontext
from pathlib import Path

import torch

from .encoders import _silence_stdout
from .esmc_generative import CANONICAL_AA, ESMCGenerator


def _patch_esm3_xpu_autocast() -> None:
    """Teach esm's `fp32_autocast_context` about XPU.

    ESM3.forward runs the structure `affine3d` geometry even for sequence-only
    inputs, and that code calls `fp32_autocast_context(device.type)`, which only
    knows cpu/mps/cuda and RAISES `ValueError: Unsupported device type: xpu` on
    Aurora. Its sole purpose is to stop AMP from downcasting the geometry to
    bf16/fp16; our ESM3 model is entirely fp32 and we run no autocast, so a no-op
    (like the existing MPS branch) is correct. Idempotent; patches both the misc
    module and any name already imported into affine3d.
    """
    try:
        from esm.utils import misc as _misc
    except Exception:
        return
    if getattr(_misc, "_filipg_xpu_patched", False):
        return
    _orig = _misc.fp32_autocast_context

    def _patched(device_type):
        if device_type == "xpu":
            return nullcontext()
        return _orig(device_type)

    _misc.fp32_autocast_context = _patched
    _misc._filipg_xpu_patched = True
    try:  # affine3d may hold its own `from ... import fp32_autocast_context`
        from esm.utils.structure import affine3d as _aff
        if hasattr(_aff, "fp32_autocast_context"):
            _aff.fp32_autocast_context = _patched
    except Exception:
        pass


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
        _patch_esm3_xpu_autocast()   # ESM3's structure geometry raises on XPU otherwise
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
    """Factory: return the generator selected by cfg.guidance.generator_type.

    Note that "pld2" is not interchangeable with the other two at the SAMPLER
    level: ESM3/ESM-C are masked LMs decoded by src/sample.py's AO-ARM loop, while
    PLD2 is a parallel-decode diffusion model that runs its own sampler. Callers
    must dispatch on `generator_type` for generation, not just for loading — see
    src/pld2_generative.py.
    """
    if cfg.guidance.generator_type == "pld2":
        from .pld2_generative import build_pld2_generator
        return build_pld2_generator(cfg, device)
    if cfg.guidance.generator_type == "esm3":
        return ESM3Generator.load(cfg.guidance.esm3_path, device)
    if cfg.guidance.generator_type == "esmc":
        return ESMCGenerator.load(cfg.guidance.generator_path, device)
    raise ValueError(f"unknown generator_type {cfg.guidance.generator_type!r} "
                     f"(expected 'pld2', 'esm3' or 'esmc')")
