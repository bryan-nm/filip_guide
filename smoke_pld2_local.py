"""Structural smoke test for the PLD2 generator wiring, with no real weights.

The real prior is 1.35B fp32 parameters (5.4GB) and a decode is 512 forwards, so
the end-to-end run belongs on Aurora. What this checks is everything that can be
wrong WITHOUT the weights, which is most of what an integration gets wrong:

  * PLD2's `src` package loads under the `pld2_src` alias and does NOT shadow
    filip_guide's own `src` or `config` (both repos name them identically).
  * A tiny randomly-initialized PLD2 (2 middle layers, d_model 64, two tracks)
    builds, and a guided decode runs end to end and returns well-formed sequences.
  * The guidance hook is actually CALLED, receives the amino-acid track with the
    shapes it expects, and tilts only the 20 residue logits.
  * The tokenizer bridge maps PLD2 ids onto the classifier's vocabulary correctly:
    residues by letter, MASK->mask, PAD->pad, EOS->eos.
  * The canvas framing offset is right — classifier position i+1 is canvas
    position i — checked by placing a known residue and reading it back.
  * `--length` pins the length exactly, via the sampler's inpainting path.
  * TAG's gradient plumbing works against a stand-in encoder: a hook on the input
    embedding receives a gradient and the resulting delta has the right shape.

    FILIPG_PLD2_REPO=../PLD2 python smoke_pld2_local.py

Exits non-zero on the first failure.
"""
from __future__ import annotations

import os
import sys

import torch

from config import default_cfg
from src.bridge import TokenizerBridge
from src.pld2_generative import (PLD2_AA, PLD2Generator, PLD2Tokenizer,
                                 load_pld2_package)

TINY = dict(vocab_size=23, eos_token_id=20, pad_token_id=21, mask_token_id=22,
            d_model=64, n_heads=4, d_ff=192,
            n_upstream=1, n_middle=2, n_downstream=1, n_recurrence=1,
            checkpoint_chunk=2, n_tracks=2)
CANVAS = 48
FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""),
          flush=True)
    if not cond:
        FAILURES.append(name)


class StubTok:
    """A classifier tokenizer in AMPLIFY's shape: specials first, then residues."""
    def __init__(self):
        self.cls_token_id, self.eos_token_id, self.mask_token_id = 0, 1, 2
        self.pad_token_id, self.unk_token_id = 3, 4
        self.bos_token_id = self.sep_token_id = None
        self._v = {"<cls>": 0, "<eos>": 1, "<mask>": 2, "<pad>": 3, "<unk>": 4}
        for i, a in enumerate(PLD2_AA):
            self._v[a] = 5 + i

    def get_vocab(self):
        return dict(self._v)

    def convert_tokens_to_ids(self, t):
        return int(self._v.get(t, -1))

    @property
    def all_special_ids(self):
        return [0, 1, 2, 3, 4]


def main() -> None:
    cfg = default_cfg()
    repo = os.environ.get("FILIPG_PLD2_REPO", cfg.guidance.pld2_repo)
    device = torch.device("cpu")
    print(f"[smoke] PLD2 repo: {repo}", flush=True)

    # --- 1. aliased import does not shadow filip_guide -----------------------
    import config as filipg_config
    import src as filipg_src
    pkg = load_pld2_package(repo)
    check("pld2_src loaded under its alias", sys.modules.get("pld2_src") is pkg)
    check("filip_guide's own `src` is untouched", sys.modules["src"] is filipg_src)
    check("filip_guide's own `config` is untouched",
          sys.modules["config"] is filipg_config,
          f"generator_type={filipg_config.default_cfg().guidance.generator_type}")

    import importlib
    pmodel = importlib.import_module("pld2_src.model")
    check("PLD2 Config/LoopedDiffusionLM importable",
          hasattr(pmodel, "LoopedDiffusionLM") and hasattr(pmodel, "Config"))

    # --- 2. a tiny model builds and decodes ---------------------------------
    torch.manual_seed(0)
    mcfg = pmodel.Config(**TINY)
    net = pmodel.LoopedDiffusionLM(mcfg).to(device).eval()
    gen = PLD2Generator(net, mcfg, PLD2Tokenizer(mcfg), device, canvas=CANVAS)
    check("n_tracks reported", gen.n_tracks == 2)

    canvas, attn = gen.init_sequence(n=3)
    check("all-MASK canvas shape", tuple(canvas.shape) == (3, 2, CANVAS),
          str(tuple(canvas.shape)))
    check("canvas starts fully masked", bool((canvas == mcfg.mask_token_id).all()))
    lp = gen.log_probs(canvas)
    check("log_probs shape (B,K,L,V)", tuple(lp.shape) == (3, 2, CANVAS, 23),
          str(tuple(lp.shape)))

    # A 1-track tensor must be refused, not silently misread.
    try:
        gen.logits(canvas[:, 0])
        check("rejects an aa-only tensor at n_tracks=2", False)
    except ValueError:
        check("rejects an aa-only tensor at n_tracks=2", True)

    # content_positions must refuse rather than invent a decode order.
    try:
        gen.content_positions(10)
        check("content_positions refuses", False)
    except NotImplementedError:
        check("content_positions refuses", True)

    # --- 3. the guidance hook is called, with the right shapes --------------
    seen = {}

    def spy(canvas_aa, logits_aa):
        seen["canvas"] = tuple(canvas_aa.shape)
        seen["logits"] = tuple(logits_aa.shape)
        seen["n"] = seen.get("n", 0) + 1
        out = logits_aa.clone()
        out[..., :20] += 1.0          # a tilt that touches only residues
        return out

    seqs = gen.generate(n=2, guidance_fn=spy, steps=12, min_len=5)
    check("guidance hook was called", seen.get("n", 0) > 0, f"{seen.get('n')} calls")
    check("hook sees the aa track only", seen.get("canvas") == (2, CANVAS),
          str(seen.get("canvas")))
    check("hook sees (B,L,V) logits", seen.get("logits") == (2, CANVAS, 23),
          str(seen.get("logits")))
    check("decoded 2 sequences", len(seqs) == 2, f"lens={[len(s) for s in seqs]}")
    check("sequences are residues only",
          all(set(s) <= set(PLD2_AA) for s in seqs),
          "".join(sorted({c for s in seqs for c in s}))[:30])

    # --- 4. --length pins the length exactly --------------------------------
    pinned = gen.generate(n=2, guidance_fn=None, length=17, steps=12)
    check("length pinned exactly", all(len(s) == 17 for s in pinned),
          f"lens={[len(s) for s in pinned]}")

    # --- 5. the tokenizer bridge ------------------------------------------
    stok = StubTok()
    bridge = TokenizerBridge(PLD2Tokenizer(mcfg), stok).to(device)
    ids = torch.tensor([[0, 19, mcfg.mask_token_id, mcfg.eos_token_id,
                         mcfg.pad_token_id]], device=device)
    got = bridge.convert(ids)[0].tolist()
    want = [stok.convert_tokens_to_ids("A"), stok.convert_tokens_to_ids("Y"),
            stok.mask_token_id, stok.eos_token_id, stok.pad_token_id]
    check("bridge maps residues/specials", got == want, f"{got} vs {want}")

    # --- 6. framing offset + TAG gradient plumbing -------------------------
    from src.pld2_guidance import PLD2FilipGuidance

    d = 8
    emb = torch.nn.Embedding(len(stok.get_vocab()), d)

    class StubEncoder(torch.nn.Module):
        """Stands in for AMPLIFY: an embedding whose output we differentiate."""
        def __init__(self):
            super().__init__()
            self.e = emb

        def get_input_embeddings(self):
            return self.e

        def forward(self, input_ids=None, attention_mask=None, output_hidden_states=True):
            return type("O", (), {"hidden_states": [self.e(input_ids)]})()

    class StubPredictor:
        """Minimal FilipPredictor surface used by the TAG path."""
        def __init__(self):
            self.pmodel, self.ptok = StubEncoder(), stok

        def log_prob_grad(self, ids, attn):
            # A scalar that genuinely depends on the embedding, so a gradient flows.
            return self.pmodel(input_ids=ids).hidden_states[0].sum(dim=(1, 2))

    guide = PLD2FilipGuidance(StubPredictor(), mcfg, device, gamma=2.0,
                              tag_normalize=False, chunk=2, verbose=False)

    # Framing: put a known residue at canvas position 0 and MASK elsewhere; the
    # classifier row must read [cls, A, mask..., eos].
    cv = torch.full((1, CANVAS), mcfg.mask_token_id, dtype=torch.long, device=device)
    cv[0, 0] = 0                                        # residue 'A'
    cids, cattn, clive = guide.to_classifier(cv)
    check("framed width is L+2", tuple(cids.shape) == (1, CANVAS + 2), str(tuple(cids.shape)))
    check("position 0 is cls", int(cids[0, 0]) == stok.cls_token_id)
    check("canvas pos 0 lands at classifier pos 1",
          int(cids[0, 1]) == stok.convert_tokens_to_ids("A"))
    check("last position is eos", int(cids[0, -1]) == stok.eos_token_id)
    check("live marks only the committed residue", int(clive.sum()) == 1)

    logits0 = torch.zeros(1, CANVAS, 23)
    tilted = guide(cv, logits0)
    check("TAG produced a tilt", bool((tilted[..., :20] != 0).any()))
    check("TAG left EOS/PAD/MASK alone", bool((tilted[..., 20:] == 0).all()))
    check("TAG call counted", guide.calls == 1)

    # gamma=0 must be a true no-op: the classifier is never touched.
    z = PLD2FilipGuidance(StubPredictor(), mcfg, device, gamma=0.0, verbose=False)
    check("gamma=0 is a no-op", bool(torch.equal(z(cv, logits0), logits0)))

    # Nothing committed yet -> the prior passes through untouched.
    empty = torch.full((1, CANVAS), mcfg.mask_token_id, dtype=torch.long, device=device)
    check("all-masked canvas passes through",
          bool(torch.equal(guide(empty, logits0), logits0)))

    print()
    if FAILURES:
        print(f"[smoke] FAILED: {len(FAILURES)} check(s): {FAILURES}", flush=True)
        sys.exit(1)
    print("[smoke] all checks passed", flush=True)


if __name__ == "__main__":
    main()
