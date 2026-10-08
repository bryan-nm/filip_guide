"""PLD2 as the unconditional generative prior p(x).

PLD2 is a looped, uniform-width bidirectional transformer trained on a discrete
diffusion objective whose absorbing state is MASK. It differs from ESM3/ESM-C as a
prior in three ways that matter to guidance, and all three are why this module does
NOT plug into src/sample.py:

  1. It decodes in PARALLEL. Generation starts from an all-MASK canvas of fixed
     width (512) and each step scores EVERY position, committing the most confident
     few on a cosine schedule (MaskGIT-style), with substitution of an already-
     committed position a first-class move alongside unmasking. There is no
     one-position-at-a-time decode order, so there is no AO-ARM path to walk and no
     per-position `log p(x_pos | committed context)` to accumulate.
  2. EOS IS THE LENGTH. The canvas resolves to `[AA* EOS PAD*]`; PAD is a modelled,
     attended token, and the sampler commits EOS first so every later step knows
     the boundary. Length is therefore a property of the canvas, not of the tensor
     shape, which is the opposite of the ESM3 `[cls] + L*[mask] + [eos]` layout.
  3. The attached checkpoint has TWO TRACKS — amino acids and Foldseek 3Di
     structure states at the same L positions, separate embedding tables summed
     into one residual stream, one output head each. The canvas is (B, 2, L) and
     the logits are (B, 2, L, V). The 3Di track co-decodes with the sequence
     unguided: FILIP scores sequence only, and PLD2's hook deliberately exposes
     only the amino-acid track (a run of identical 3Di states is a helix, not a
     defect, so sequence-level guidance must not see it).

So filip_guide does not reimplement PLD2's reverse process. It drives PLD2's own
`generate()` through the `guidance_fn(canvas, logits) -> logits` hook that PLD2
exposes for exactly this purpose, and supplies the FILIP term through it (see
src/pld2_guidance.py). This module owns loading, the vocabulary bridge, and the
decode call; it implements the parts of the generator contract that are meaningful
for a parallel decoder and raises a pointed error on the parts that are not.

PLD2 is IMPORTED from its own checkout (config.PLD2_REPO), not vendored — see the
note in config.py.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch

from .esmc_generative import CANONICAL_AA

# PLD2's residue alphabet IS its id order: ids 0..19 index this string (PLD2
# src/blosum.py AA, and src/data.py DI for the 3Di track, which happens to use the
# same 20 letters for entirely different states). It coincides with filip_guide's
# CANONICAL_AA, which is what lets a [..., 20] guidance tensor line up with
# `logits[..., :20]` with no reindexing; assert it rather than trusting it.
PLD2_AA = "ACDEFGHIKLMNPQRSTVWY"
assert PLD2_AA == CANONICAL_AA, "PLD2 residue id order diverged from CANONICAL_AA"

# The checkpoint carries NO config — PLD2 saves {"model", "sched", "step"} and
# nothing else, so the architecture cannot be recovered from the file and a
# mismatch is a load_state_dict failure, not a graceful fallback. These are PLD2's
# `CFG.model_config()` values for the 1.35B run that produced the aligned
# checkpoints (d_model 1536, 4 + 36x1 + 4 = 44 distinct layers, 24 heads to keep
# head_dim at 64, two tracks). Verified against the attached ali_r10_step499.ckpt:
# embed.weight (23, 1536), mid_layers.0..35, struct_embed/struct_head present.
# If PLD2 retrains at another size, this is the one place that must follow.
PLD2_MODEL_CONFIG = dict(
    vocab_size=23, eos_token_id=20, pad_token_id=21, mask_token_id=22,
    d_model=1536, n_heads=24, d_ff=4608,
    n_upstream=4, n_middle=36, n_downstream=4, n_recurrence=1,
    checkpoint_chunk=6, n_tracks=2,
)

# PLD2's package is ALSO named `src`, so a plain `sys.path.insert` + `import src`
# would resolve to filip_guide's own package (already imported) and silently hand
# back the wrong modules. Load it under an alias instead. Its relative imports
# (`from .model import ...`) then resolve inside the alias, and because the subgraph
# we need imports nothing but stdlib and torch, PLD2's root config.py is never
# touched — which also means it cannot shadow filip_guide's config.py.
_PKG = "pld2_src"


def load_pld2_package(repo: str):
    """Import PLD2's `src` package as `pld2_src`. Idempotent."""
    if _PKG in sys.modules:
        return sys.modules[_PKG]
    pkg_dir = Path(repo).expanduser().resolve() / "src"
    init = pkg_dir / "__init__.py"
    if not init.is_file():
        raise FileNotFoundError(
            f"no PLD2 checkout at {repo!r} (expected {init}). Point "
            f"FILIPG_PLD2_REPO / cfg.guidance.pld2_repo at the PLD2 repo root.")
    spec = importlib.util.spec_from_file_location(
        _PKG, init, submodule_search_locations=[str(pkg_dir)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PKG] = module          # must be registered BEFORE exec for relative imports
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_PKG, None)
        raise
    return module


def _pld2(name: str):
    """`pld2_src.<name>`, assuming load_pld2_package has run."""
    return importlib.import_module(f"{_PKG}.{name}")


class PLD2Tokenizer:
    """An HF-shaped view of PLD2's 23-token vocabulary.

    PLD2 has no tokenizer object in its generation path — it is raw id arrays over
    `0..19 = residue, 20 = EOS, 21 = PAD, 22 = MASK`. filip_guide's TokenizerBridge
    maps generator ids to classifier ids by asking a tokenizer for residues by
    letter and specials by role, so exposing that handful of attributes lets the
    bridge be reused UNCHANGED rather than growing a PLD2 special case.

    There is deliberately no CLS/BOS: PLD2's canvas carries no framing token, and
    reporting one would make the bridge map a residue position onto the
    classifier's CLS. The framing the classifier wants is added by the guidance
    adapter, which is the only place that knows the alignment matters.
    """

    def __init__(self, mcfg):
        self.mask_token_id = int(mcfg.mask_token_id)
        self.eos_token_id = int(mcfg.eos_token_id)
        self.pad_token_id = int(mcfg.pad_token_id)
        self.cls_token_id = None
        self.bos_token_id = None
        self.sep_token_id = None
        self.unk_token_id = None
        self._vocab = {aa: i for i, aa in enumerate(PLD2_AA)}
        self._vocab["<eos>"] = self.eos_token_id
        self._vocab["<pad>"] = self.pad_token_id
        self._vocab["<mask>"] = self.mask_token_id
        self._inv = {i: t for t, i in self._vocab.items()}

    def get_vocab(self) -> dict:
        return dict(self._vocab)

    def convert_tokens_to_ids(self, token: str) -> int:
        """-1 for anything PLD2 has no state for (the bridge treats that as absent).

        PLD2's 20-residue alphabet has no non-standard letters, so B/U/X/Z/O/J land
        here. The training corpus folded them into standard residues, but a
        generator can never PROPOSE them, so the honest answer is "no such id".
        """
        return int(self._vocab.get(token, -1))

    def convert_ids_to_tokens(self, ids) -> List[str]:
        if isinstance(ids, int):
            return self._inv.get(int(ids), "<unk>")
        return [self._inv.get(int(i), "<unk>") for i in ids]

    @property
    def all_special_ids(self) -> List[int]:
        return [self.eos_token_id, self.pad_token_id, self.mask_token_id]


class PLD2Generator:
    """PLD2 wrapped for filip_guide: loading, vocabulary, and a guided decode.

    Implements the half of the generator contract that is meaningful for a parallel
    decoder — `tok`, `aa_ids`, `decode`, `init_sequence`, `log_probs` — and raises
    on `content_positions`, which only means something for a sequential decode
    order PLD2 does not have.
    """

    def __init__(self, model, mcfg, tokenizer, device, *, canvas: int = 512,
                 n_recurrence: Optional[int] = None):
        self.model = model
        self.mcfg = mcfg
        self.tok = tokenizer
        self.device = device
        self.canvas = int(canvas)
        self.n_recurrence = n_recurrence
        self.mask_id = int(mcfg.mask_token_id)
        self.eos_id = int(mcfg.eos_token_id)
        self.pad_id = int(mcfg.pad_token_id)
        self.cls_id = None                       # PLD2 has no framing token
        # PLD2 ids of the residues we may propose. Identity, because PLD2 puts the
        # 20 residues first by contract; kept for interface parity with the ESM
        # generators, whose residue ids are scattered through a larger vocabulary.
        self.aa_ids = list(range(len(PLD2_AA)))
        self.n_tracks = int(mcfg.n_tracks)

    # --- loading -----------------------------------------------------------
    @classmethod
    def load(cls, repo: str, ckpt: str, device, *, canvas: int = 512,
             n_recurrence: Optional[int] = None,
             model_config: Optional[dict] = None) -> "PLD2Generator":
        """Build the architecture from PLD2_MODEL_CONFIG and load `ckpt` into it."""
        load_pld2_package(repo)
        mod = _pld2("model")
        mcfg = mod.Config(**(model_config or PLD2_MODEL_CONFIG))
        model = mod.LoopedDiffusionLM(mcfg).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        path = Path(ckpt).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"no PLD2 checkpoint at {path}. Point FILIPG_PLD2_CKPT / "
                f"cfg.guidance.pld2_ckpt at the weights.")
        # mmap so a 5.4GB fp32 state dict is not held twice; weights_only because
        # the blob is plain tensors + an LR-scheduler state we never read.
        state = torch.load(str(path), map_location="cpu", mmap=True, weights_only=True)
        sd = state["model"] if "model" in state else state
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            # PLD2 embeds no config, so a size/track mismatch surfaces HERE or not
            # at all. Refuse rather than generate from a half-initialized prior.
            raise RuntimeError(
                f"PLD2 checkpoint does not match PLD2_MODEL_CONFIG: "
                f"{len(missing)} missing, {len(unexpected)} unexpected keys "
                f"(first missing {missing[:3]}, first unexpected {unexpected[:3]}). "
                f"The checkpoint carries no config, so update PLD2_MODEL_CONFIG in "
                f"src/pld2_generative.py to the run that produced it.")
        del state, sd
        return cls(model, mcfg, PLD2Tokenizer(mcfg), device,
                   canvas=canvas, n_recurrence=n_recurrence)

    # --- the prior ---------------------------------------------------------
    @torch.no_grad()
    def logits(self, canvas: torch.Tensor, attn: torch.Tensor = None) -> torch.Tensor:
        """PLD2 logits for a canvas.

        `canvas` is (B, K, L) over PLD2 ids, K = n_tracks. Returns (B, K, L, V).
        `attn` is accepted for interface parity and unused: PLD2 runs full
        bidirectional attention over a fixed canvas in which PAD is a MODELLED
        token, not attention filler, and every caller in PLD2 passes no mask.
        """
        canvas = self._as_canvas(canvas)
        kw = {} if self.n_recurrence in (None, 0) else {"n_recurrence": self.n_recurrence}
        if self.n_tracks == 2:
            out = self.model(canvas[:, 0], struct=canvas[:, 1], **kw)   # (B, 2, L, V)
        else:
            out = self.model(canvas[:, 0], **kw).unsqueeze(1)           # (B, 1, L, V)
        return out.float()

    @torch.no_grad()
    def log_probs(self, canvas: torch.Tensor, attn: torch.Tensor = None) -> torch.Tensor:
        """Per-position log p(token | canvas), (B, K, L, V)."""
        return torch.log_softmax(self.logits(canvas, attn), dim=-1)

    def _as_canvas(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return x
        if x.dim() == 2 and self.n_tracks == 1:
            return x[:, None, :]
        raise ValueError(
            f"expected a (B, {self.n_tracks}, L) PLD2 canvas, got {tuple(x.shape)}. "
            f"This checkpoint has {self.n_tracks} tracks, so the amino-acid ids "
            f"alone are not a complete input — the 3Di track is summed into the "
            f"same residual stream and the forward requires it.")

    # --- canvas helpers ----------------------------------------------------
    def init_sequence(self, length: Optional[int] = None, n: int = 1):
        """An all-MASK canvas, which is PLD2's stationary distribution.

        Returns (canvas (n, K, canvas_width), attn (n, canvas_width)). `length` is
        accepted for interface parity and intentionally NOT used to size the
        tensor: PLD2's width is fixed and the sequence length is decided by where
        the sampler commits EOS. Pass `length` to `generate(..., length=...)` to
        pin it instead.
        """
        c = torch.full((n, self.n_tracks, self.canvas), self.mask_id,
                       dtype=torch.long, device=self.device)
        attn = torch.ones((n, self.canvas), dtype=torch.long, device=self.device)
        return c, attn

    def content_positions(self, length: int):
        raise NotImplementedError(
            "PLD2 has no sequential decode order: it commits the most confident "
            "edits across the whole canvas each step, so there is no position "
            "sequence to walk and src/sample.py's AO-ARM loop does not apply. Use "
            "PLD2Generator.generate(), which drives PLD2's own sampler through its "
            "guidance_fn hook.")

    def decode(self, row: torch.Tensor) -> str:
        """One canvas row -> amino-acid string, truncated at EOS/PAD."""
        seqs = self.decode_batch(row[None] if row.dim() == 2 else row[None, None])
        return seqs[0] if seqs else ""

    def decode_batch(self, canvas: torch.Tensor) -> List[str]:
        """(B, K, L) canvas -> amino-acid strings via PLD2's own decoder.

        Delegated rather than reimplemented: PLD2 truncates at the first EOS/PAD of
        the AMINO-ACID track (the only track that carries the boundary) and drops
        surviving MASK tokens, and getting that wrong is silent.
        """
        sampler = _pld2("sampler")
        seqs, _ = sampler.decode_seqs(self._as_canvas(canvas), self.mcfg, min_len=0)
        return seqs

    def decode_struct_batch(self, canvas: torch.Tensor) -> List[str]:
        """The co-decoded 3Di structure track, index-aligned with decode_batch."""
        if self.n_tracks < 2:
            return []
        sampler = _pld2("sampler")
        dis, _ = sampler.decode_struct(self._as_canvas(canvas), self.mcfg, min_len=0)
        return dis

    # --- guided decode -----------------------------------------------------
    @torch.no_grad()
    def generate(
        self, n: int = 1, *, guidance_fn: Optional[Callable] = None,
        length: Optional[int] = None, steps: int = 512, temperature: float = 1.0,
        gumbel_temp: float = 0.1, subst_per_residue: float = 1.0,
        rep_penalty: float = 0.0, max_run: int = 5, min_len: int = 30,
        eos_first: bool = True, struct_first: float = 0.0, greedy: bool = False,
        return_canvas: bool = False,
    ):
        """Draw `n` sequences from PLD2 under `guidance_fn`.

        `guidance_fn(canvas_aa, logits_aa) -> logits_aa` is PLD2's steering hook;
        pass the FILIP adapter from src/pld2_guidance.py, or None for the
        unconditional baseline. PLD2 composes it AFTER its own repetition penalty,
        so the two stack rather than replacing one another.

        `length`, if given, pins the sequence length by pre-committing EOS at that
        position and PAD after it (the sampler's own `prompt`/`prompt_mask`
        inpainting path, which freezes those positions and disables its EOS-first
        draw). That is what the matched-length evaluation wants: every arm —
        guided, unconditional, and the true sequence — at the same length. Left
        None, PLD2 draws the length itself.

        Returns the decoded amino-acid strings, or `(seqs, structs, canvas)` with
        `return_canvas` so the co-decoded 3Di track can be kept for a
        self-consistency check.
        """
        sampler = _pld2("sampler")
        prompt = prompt_mask = None
        if length is not None:
            prompt, prompt_mask = self._pin_length(int(length), n)

        canvas, _lengths = sampler.generate(
            self.model, Lmax=self.canvas, batch_size=n, n_steps=steps,
            temperature=temperature, gumbel_temp=gumbel_temp, greedy=greedy,
            guidance_fn=guidance_fn, device=str(self.device),
            eos_first=(eos_first and length is None), min_len=min_len,
            rep_penalty=rep_penalty, max_run=max_run,
            subst_per_residue=subst_per_residue, struct_first=struct_first,
            prompt=prompt, prompt_mask=prompt_mask,
            **({} if self.n_recurrence in (None, 0) else {"n_recurrence": self.n_recurrence}),
        )
        seqs = self.decode_batch(canvas)
        if not return_canvas:
            return seqs
        return seqs, self.decode_struct_batch(canvas), canvas

    def _pin_length(self, length: int, n: int):
        """A prompt canvas that fixes the boundary: EOS at `length`, PAD after.

        PLD2's sampler requires a prompted row to carry EOS in the amino-acid
        track, and pads every track from the boundary INCLUSIVE with no EOS of its
        own on the 3Di side. Residue positions are left MASK and unfrozen, so
        guidance still decides every residue; only the length is given.
        """
        if not 0 < length < self.canvas:
            raise ValueError(f"length must be in 1..{self.canvas - 1}, got {length}")
        K, L = self.n_tracks, self.canvas
        prompt = torch.full((n, K, L), self.mask_id, dtype=torch.long, device=self.device)
        given = torch.zeros((n, K, L), dtype=torch.bool, device=self.device)
        prompt[:, 0, length] = self.eos_id
        prompt[:, 0, length + 1:] = self.pad_id
        given[:, 0, length:] = True
        if K == 2:
            prompt[:, 1, length:] = self.pad_id      # 3Di pads from the boundary inclusive
            given[:, 1, length:] = True
        return prompt, given


def build_pld2_generator(cfg, device) -> PLD2Generator:
    """Factory from a filip_guide Cfg."""
    g = cfg.guidance
    return PLD2Generator.load(
        g.pld2_repo, g.pld2_ckpt, device,
        canvas=g.pld2_canvas,
        n_recurrence=(g.pld2_n_recurrence or None),
    )
