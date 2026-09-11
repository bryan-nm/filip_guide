"""Masked iterative-unmasking sampler for guided ESM-C generation.

Starts from a fully-masked sequence and reveals one position per step, sampling
each residue from the guided distribution (guide.py). Decode order is either the
deterministic ascending sequence position (the paper's reproducible AO-ARM order)
or a random permutation. Sampling uses a temperature; `greedy` takes the argmax.

`n > 1` is a batched draw of n independent candidates (one generator forward of
batch n and one classifier forward of batch n*C per position), which is how
best-of-n sampling is served: `return_path_logp` additionally accumulates each
candidate's generator log-likelihood along the decode path, for free, as the
selection metric.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch

from .guide import DEGGuidance


@torch.no_grad()
def guided_generate(
    guidance: DEGGuidance, length: int, n: int = 1, *,
    temperature: float = 0.1, order: str = "ascending", greedy: bool = False,
    generator_seed: Optional[int] = None, return_path_logp: bool = False,
) -> Union[List[str], Tuple[List[str], List[float]]]:
    """Generate `n` sequences of `length` residues under prompt guidance.

    Returns the decoded amino-acid strings. `order`: "ascending" (left-to-right)
    or "random". Uses one generator forward + one classifier forward (batch =
    #candidates) per decoded position.

    With `return_path_logp`, also returns per-candidate sum_pos log p_gen(x_pos |
    committed context) — the generator's AO-ARM path log-likelihood under this
    decode order, accumulated from the prior term the guided score already
    computes (no extra forwards). Comparable across candidates of one entry (they
    share `length`); divide by `length` before comparing across entries.
    """
    gen = guidance.generator
    if generator_seed is not None:
        torch.manual_seed(generator_seed)               # seeds every device's default RNG

    x, attn = gen.init_sequence(length, n)
    positions = gen.content_positions(length)
    if order == "random":
        perm = torch.randperm(len(positions)).tolist()  # CPU op; global-seeded above
        positions = [positions[i] for i in perm]
    elif order != "ascending":
        raise ValueError(f"order must be 'ascending' or 'random' (got {order!r})")

    cand = guidance.candidate_ids                       # [C] ESM-C ids
    path_logp = torch.zeros(n, device=x.device) if return_path_logp else None
    for pos in positions:
        if return_path_logp:
            glp, prior, _ = guidance.guided_position_logprobs(x, attn, pos,
                                                              return_components=True)
        else:
            glp = guidance.guided_position_logprobs(x, attn, pos)  # [n, C]
        if greedy:
            idx = glp.argmax(dim=-1)                                # [n]
        else:
            probs = torch.softmax(glp / max(temperature, 1e-6), dim=-1)
            idx = torch.multinomial(probs, num_samples=1).squeeze(1)
        if return_path_logp:
            path_logp += prior.gather(1, idx[:, None]).squeeze(1)
        x[:, pos] = cand[idx]

    seqs = [gen.decode(x[i]) for i in range(n)]
    return (seqs, path_logp.tolist()) if return_path_logp else seqs
