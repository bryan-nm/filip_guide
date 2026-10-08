"""TAG guidance: tilt PLD2's logits by the FILIP classifier, in one backward.

ProteinGuide's Bayes rule is the same as in src/guide.py —

    log p(x̃ | x, prompt)  ∝  log p(x̃ | x)  +  γ · log p(prompt | x̃)
         guided score            PLD2 prior      FILIP classifier

— but the cost structure is completely different, and that is what dictates the
method.

WHY NOT EXACT DEG HERE. src/guide.py enumerates all 20 residues at ONE position
per step, which is 20 classifier forwards per decoded position: exact, and
affordable because a sequential decoder asks about one position at a time. PLD2
decodes in parallel and calls its guidance hook once per step with the WHOLE
canvas, so the same exactness would mean L·C classifier forwards per step —
~5M AMPLIFY forwards for a single 512-wide, 512-step sample. Not a tuning
problem; a different algorithm is needed.

TAG (first-order guidance). Run the classifier ONCE on the current canvas and take
the gradient of log p(prompt | x) with respect to the input embeddings. The score
for writing residue c at position i is then a directional derivative:

    δ[i, c]  ≈  ⟨ ∂ log p(prompt|x) / ∂ e_i ,  E[c] − e_i ⟩

The `⟨grad_i, e_i⟩` term depends only on i, so it is constant across candidates and
cancels under the per-position softmax — dropped. One forward and one backward per
step therefore score EVERY (position, residue) pair at once, at a cost independent
of sequence length, which is exactly the shape PLD2's hook wants.

It is an approximation: a first-order expansion around the current embedding,
whereas DEG evaluates the classifier at the candidate itself. The trade is
deliberate and is the same one PLD2 makes in production.

TWO THINGS THAT WOULD SILENTLY BREAK THIS, both guarded below:

  * Alignment. PLD2's canvas carries no framing token; the classifier wants
    `[cls] + body + [eos]`. So classifier position i+1 corresponds to canvas
    position i, and the gradient must be read back with that offset. Get it wrong
    and guidance is applied to neighbouring residues — plausible-looking and
    meaningless. The slice is `g[:, 1:L+1, :]`.
  * The embedding table. `model.get_input_embeddings()` is defined on the
    transformers base class, so `hasattr` always succeeds and AMPLIFY raises
    NotImplementedError from it rather than returning None. Falling back to "the
    biggest nn.Embedding" would pick a positional table, which TAG would then index
    with token ids — again confident and meaningless. `_input_embedding` identifies
    the token table by row count and REFUSES rather than guessing.

Masked positions are excluded from the FILIP score (Decision #2), so the score
reads only committed residues: guidance means "given what is placed so far, does
this look like the prompt". Before anything is committed the classifier has nothing
to read and the hook returns the prior untouched.
"""
from __future__ import annotations

from typing import Optional

import torch

from .bridge import TokenizerBridge
from .filip_predictor import FilipPredictor
from .pld2_generative import PLD2_AA, PLD2Tokenizer


def _input_embedding(model, tokenizer, verbose: bool = True) -> torch.nn.Embedding:
    """The module whose OUTPUT TAG differentiates with respect to.

    Identified by row count against the tokenizer's vocabulary, because the
    plausible fallbacks are all wrong in a way that does not announce itself (see
    the module docstring).
    """
    try:
        emb = model.get_input_embeddings()
        if isinstance(emb, torch.nn.Embedding):
            return emb
    except (NotImplementedError, AttributeError):
        pass

    named = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Embedding)]
    if not named:
        raise RuntimeError(
            "no nn.Embedding in the protein encoder, so TAG has nothing to "
            "differentiate with respect to.")
    try:
        vocab = len(tokenizer.get_vocab())
    except Exception:
        vocab = None
    exact = [(n, m) for n, m in named if vocab and m.num_embeddings == vocab]
    if not exact:
        if len(named) > 1:
            raise RuntimeError(
                f"cannot identify the token embedding: tokenizer vocabulary "
                f"{vocab!r} matches none of {[(n, m.num_embeddings) for n, m in named]}. "
                f"Guiding against a positional table would produce a "
                f"plausible-looking signal that means nothing, so this refuses "
                f"rather than guesses.")
        exact = named
    name, mod = exact[0]
    if verbose:
        print(f"[pld2-tag] differentiating at '{name}' "
              f"({mod.num_embeddings} x {mod.embedding_dim}; tokenizer vocab {vocab})",
              flush=True)
    return mod


class PLD2FilipGuidance:
    """A PLD2 `guidance_fn(canvas, logits) -> logits` backed by a FilipPredictor.

    PLD2 hands the hook only the AMINO-ACID track, by design: the 3Di structure
    track co-decodes unguided, because a run of identical structure states is a
    helix rather than a defect and sequence-level guidance must not penalize it.
    Only the 20 residue logits are tilted; EOS/PAD/MASK are left to PLD2, which
    owns the length.
    """

    def __init__(self, predictor: FilipPredictor, mcfg, device, *,
                 gamma: float = 10.0, tag_normalize: bool = True, chunk: int = 8,
                 verbose: bool = True):
        self.predictor = predictor
        self.mcfg = mcfg
        self.device = device
        self.gamma = float(gamma)
        self.tag_normalize = bool(tag_normalize)
        self.chunk = max(int(chunk), 1)
        self.calls = 0

        ptok = predictor.ptok
        # PLD2 ids -> classifier ids. The existing bridge, unchanged: it maps
        # residues by letter and specials by role, and PLD2Tokenizer presents
        # PLD2's vocabulary in the shape the bridge expects.
        self.bridge = TokenizerBridge(PLD2Tokenizer(mcfg), ptok).to(device)
        self.cls_id = _first_id(ptok, "cls_token_id", "bos_token_id")
        self.eos_id = _first_id(ptok, "eos_token_id", "sep_token_id")

        self.emb = _input_embedding(predictor.pmodel, ptok, verbose=verbose)
        # Classifier ids of the 20 residues IN PLD2's OWN ORDER, so the [.., 20]
        # result lines up with logits[..., :20] with no reindexing.
        aa_ids = torch.tensor(
            [int(ptok.convert_tokens_to_ids(a)) for a in PLD2_AA],
            dtype=torch.long, device=device)
        if int(aa_ids.max()) >= self.emb.num_embeddings:
            raise RuntimeError(
                f"the located embedding has {self.emb.num_embeddings} rows but the "
                f"tokenizer puts residues up to id {int(aa_ids.max())}. That is not "
                f"the token embedding.")
        self.aa_ids = aa_ids
        self.E_aa = self.emb.weight.detach()[aa_ids].float()        # [20, d]

    # --- canvas -> classifier ---------------------------------------------
    def to_classifier(self, canvas_aa: torch.Tensor):
        """(B, L) PLD2 ids -> (ids (B, L+2), attn (B, L+2), live (B, L)).

        Frames as `[cls] + body + [eos]`, so classifier position i+1 ↔ canvas
        position i. PAD is outside the molecule and gets attention 0; MASK stays a
        real attended position whose id the classifier then excludes from the score.
        """
        B, L = canvas_aa.shape
        body = self.bridge.convert(canvas_aa)
        ids = torch.empty(B, L + 2, dtype=torch.long, device=canvas_aa.device)
        ids[:, 0] = self.cls_id
        ids[:, 1:L + 1] = body
        ids[:, L + 1] = self.eos_id
        keep = canvas_aa != self.mcfg.pad_token_id
        attn = torch.ones(B, L + 2, dtype=torch.long, device=canvas_aa.device)
        attn[:, 1:L + 1] = keep.long()
        live = keep & (canvas_aa != self.mcfg.mask_token_id) \
                    & (canvas_aa != self.mcfg.eos_token_id)
        return ids, attn, live

    # --- the hook ----------------------------------------------------------
    def __call__(self, canvas_aa: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """canvas (B, L) long, logits (B, L, V) -> tilted logits (B, L, V)."""
        if self.gamma == 0.0:
            return logits
        self.calls += 1
        ids, attn, live = self.to_classifier(canvas_aa)
        if not bool(live.any()):
            return logits            # nothing committed: the classifier has nothing to read
        delta = self._tag(ids, attn, live)                           # [B, L, 20]
        out = logits.clone()
        out[..., :20] = out[..., :20] + self.gamma * delta.to(out.dtype)
        return out

    def _tag(self, ids, attn, live) -> torch.Tensor:
        """[B, L, 20] first-order guidance, chunked and length-trimmed.

        The BACKWARD is what costs: a transformer with grad enabled retains a
        [rows, heads, L, L] attention matrix per layer, so peak memory is linear in
        rows and quadratic in length. Chunking rows is EXACT, not an approximation —
        log p(prompt | x) sums over independent rows, so a row's gradient does not
        depend on which other rows shared its forward. Trimming to the batch's real
        extent compounds it: the canvas is 512 wide but a decoded protein is ~250,
        and the attention matrix goes as L², so the PAD tail can cost several times
        what the molecule does.
        """
        B, L = live.shape
        extent = int(attn.sum(1).max().item())
        keep = min(max(extent, 4), ids.shape[1])
        ids, attn = ids[:, :keep], attn[:, :keep]
        live_t = live[:, :max(keep - 2, 0)]

        parts = [self._tag_chunk(ids[s:s + self.chunk], attn[s:s + self.chunk],
                                 live_t[s:s + self.chunk])
                 for s in range(0, B, self.chunk)]
        g = torch.cat(parts, dim=0)
        # Scatter back onto the full canvas width; everything trimmed was PAD.
        full = torch.zeros(B, L, g.shape[-1], device=g.device, dtype=g.dtype)
        full[:, :g.shape[1]] = g
        return full

    def _tag_chunk(self, ids, attn, live) -> torch.Tensor:
        L = live.shape[1]
        captured = {}

        def hook(_m, _inp, out):
            out.requires_grad_(True)
            out.retain_grad()
            captured["e"] = out
            return out

        handle = self.emb.register_forward_hook(hook)
        try:
            # PLD2's sampler runs under no_grad, so the graph must be re-enabled
            # locally or the backward has nothing to walk.
            with torch.enable_grad():
                self.predictor.log_prob_grad(ids, attn).sum().backward()
        except RuntimeError as ex:
            if "out of memory" in str(ex).lower() or "OUT_OF_RESOURCES" in str(ex):
                raise RuntimeError(
                    f"the classifier ran out of device memory at chunk={self.chunk}, "
                    f"L={ids.shape[1]}. The backward retains an attention matrix per "
                    f"layer, so cost is linear in chunk and quadratic in length: "
                    f"lower cfg.guidance.pld2_tag_chunk.") from ex
            raise
        finally:
            handle.remove()

        g = captured.get("e", None)
        g = None if g is None else g.grad
        if g is None:
            raise RuntimeError(
                "no gradient reached the protein encoder's input embedding, so TAG "
                "cannot run. The likely cause is a frozen-graph path in the "
                "classifier (an inner no_grad) rather than a configuration error.")
        # ⟨grad_i, E[c]⟩, read back with the framing offset: classifier position
        # i+1 is canvas position i.
        delta = g[:, 1:L + 1, :].detach().float() @ self.E_aa.t()          # [b, L, 20]
        if self.tag_normalize:
            # The gradient's magnitude moves by orders of magnitude across a decode
            # (it depends on how many positions are committed and how peaked the
            # max-sim is), so an unnormalized gamma would mean something different
            # at every step. Standardizing across the 20 candidates puts gamma in
            # logit units; the per-position mean cancels under the softmax anyway,
            # so centring costs nothing.
            delta = (delta - delta.mean(-1, keepdim=True)) \
                / delta.std(-1, keepdim=True).clamp_min(1e-6)
        return delta


def _first_id(tok, *names) -> int:
    for n in names:
        v = getattr(tok, n, None)
        if v is not None and int(v) >= 0:
            return int(v)
    return 0


def build_pld2_guidance(predictor: FilipPredictor, generator, cfg, device, *,
                        gamma: Optional[float] = None,
                        verbose: bool = True) -> PLD2FilipGuidance:
    """Factory from a filip_guide Cfg + a loaded PLD2Generator."""
    g = cfg.guidance
    if g.pld2_guide_mode != "tag":
        raise ValueError(
            f"pld2_guide_mode={g.pld2_guide_mode!r} is not supported: PLD2's hook "
            f"is called once per step with the whole canvas, so exact DEG would be "
            f"~5M classifier forwards per sample. Only 'tag' is viable here.")
    return PLD2FilipGuidance(
        predictor, generator.mcfg, device,
        gamma=(g.gamma if gamma is None else gamma),
        tag_normalize=g.pld2_tag_normalize, chunk=g.pld2_tag_chunk, verbose=verbose)
