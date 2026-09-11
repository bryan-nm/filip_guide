"""Model-free smoke test for best-of-n sampling: needs torch only, no weights.

Exercises the REAL code paths — `sample.guided_generate(return_path_logp=True)`,
`guide.DEGGuidance.guided_position_logprobs(return_components=True)` and
`FilipPredictor.log_prob`'s row chunking — against stub generator/classifier
objects, and independently re-derives each candidate's path log-likelihood by
replaying the decode. Run this before spending Aurora hours:

    python smoke_best_of_n.py
"""
from __future__ import annotations

import torch

from src.filip_predictor import FilipPredictor
from src.guide import DEGGuidance
from src.sample import guided_generate

V, MASK, CLS, EOS = 32, 1, 0, 2
AA_IDS = list(range(4, 24))                      # 20 "residues"
LETTERS = "ACDEFGHIKLMNPQRSTVWY"


class FakeGen:
    """Deterministic context-dependent stand-in for ESM3Generator."""
    device = torch.device("cpu")
    aa_ids = AA_IDS

    def log_probs(self, ids, attn=None):
        B, L = ids.shape
        # A fixed pseudo-random function of the CURRENT token content, so the
        # "prior" genuinely depends on what has been committed so far.
        h = (ids.float() * 7.31 + torch.arange(L, dtype=torch.float)[None, :] * 1.17)
        logits = torch.sin(h[:, :, None] * torch.arange(1, V + 1, dtype=torch.float)) * 3.0
        return torch.log_softmax(logits, dim=-1)

    def init_sequence(self, length, n=1):
        ids = torch.full((n, length + 2), MASK, dtype=torch.long)
        ids[:, 0], ids[:, -1] = CLS, EOS
        return ids, torch.ones((n, length + 2), dtype=torch.long)

    def content_positions(self, length):
        return list(range(1, length + 1))

    def decode(self, row):
        return "".join(LETTERS[int(t) - 4] for t in row if 4 <= int(t) < 24)


class FakePredictor:
    """log p(prompt|x) that depends on the committed residues; counts its calls."""
    def __init__(self):
        self.calls, self.batch_sizes = 0, []

    def log_prob(self, ids, attn):
        self.calls += 1
        self.batch_sizes.append(ids.size(0))
        committed = (ids != MASK) & (ids >= 4)
        return -(ids.float() * committed).sum(1) / 100.0


class FakeBridge:
    def convert(self, x):
        return x


def replay_path_logp(gen, seqs, length):
    """Recompute sum_pos log p_gen(x_pos | committed context) from the final
    sequences alone, by walking the same ascending decode order."""
    n = len(seqs)
    x, attn = gen.init_sequence(length, n)
    tok = torch.tensor([[AA_IDS[LETTERS.index(c)] for c in s] for s in seqs])
    total = torch.zeros(n)
    for i, pos in enumerate(gen.content_positions(length)):
        lp = gen.log_probs(x, attn)[:, pos, :]
        total += lp.gather(1, tok[:, i:i + 1]).squeeze(1)
        x[:, pos] = tok[:, i]
    return total


def check_sampler(gamma, n=8, length=25):
    gen, pred = FakeGen(), FakePredictor()
    g = DEGGuidance(gen, pred, FakeBridge(), gamma)
    seqs, path = guided_generate(g, length, n=n, temperature=1.0,
                                 order="ascending", generator_seed=1234,
                                 return_path_logp=True)
    assert len(seqs) == n and len(path) == n, (len(seqs), len(path))
    assert all(len(s) == length for s in seqs), [len(s) for s in seqs]

    # The free accumulated path logp must equal an independent replay.
    ref = replay_path_logp(gen, seqs, length)
    err = max(abs(a - b) for a, b in zip(path, ref.tolist()))
    assert err < 1e-4, f"path_logp mismatch (max err {err})"

    # Same seed -> identical candidate set (guided/uncond arms stay paired).
    seqs2, path2 = guided_generate(DEGGuidance(gen, FakePredictor(), FakeBridge(), gamma),
                                   length, n=n, temperature=1.0, order="ascending",
                                   generator_seed=1234, return_path_logp=True)
    assert seqs2 == seqs, "same seed produced a different candidate set"
    assert max(abs(a - b) for a, b in zip(path, path2)) < 1e-6

    calls = pred.calls
    print(f"  gamma={gamma:<5g} n={n} distinct={len(set(seqs))}/{n} "
          f"path_logp [{min(path):+.2f}, {max(path):+.2f}] "
          f"replay_err={err:.2e} classifier_calls={calls}")
    return seqs, path


def check_row_chunking():
    """FilipPredictor.log_prob must split a big candidate batch into row blocks
    and reassemble it in order (this is what keeps n=10 off the OOM path)."""
    p = FilipPredictor.__new__(FilipPredictor)
    p.row_chunk = 32
    seen = []

    def block(ids, attn):
        seen.append(ids.size(0))
        return ids[:, 0].float()                 # order-revealing per-row value

    p._log_prob_block = block
    B = 200                                      # n_candidates=10 -> 10*20 rows
    ids = torch.arange(B)[:, None].repeat(1, 5)
    out = p.log_prob(ids, torch.ones_like(ids))
    assert seen == [32] * 6 + [8], seen
    assert torch.equal(out, torch.arange(B).float()), "row blocks reassembled out of order"
    print(f"  row_chunk=32 on B={B}: blocks {seen}, order preserved")


def main():
    print("== row chunking ==")
    check_row_chunking()

    print("== guided arm (gamma>0: prior + gamma*classifier) ==")
    check_sampler(5.0)
    print("== unconditional arm (gamma=0: classifier never called) ==")
    _, _ = check_sampler(0.0)
    gen, pred = FakeGen(), FakePredictor()
    guided_generate(DEGGuidance(gen, pred, FakeBridge(), 0.0), 25, n=8,
                    temperature=1.0, generator_seed=1234, return_path_logp=True)
    assert pred.calls == 0, f"gamma=0 called the classifier {pred.calls} times"
    print("  classifier calls at gamma=0: 0")

    print("\nBEST_OF_N_SMOKE_OK")


if __name__ == "__main__":
    main()
