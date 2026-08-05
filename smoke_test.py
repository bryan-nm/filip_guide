"""Milestone-1 smoke test: config, model heads, both encoders, one real ESM-C
forward, and the FILIP score plumbing between them.

    python smoke_test.py

The ESM-C forward requires the `esm` package (registers the `esmc` architecture
into transformers):

    pip install esm@git+https://github.com/Biohub/esm.git@main

Everything except the ESM-C forward runs without it.
"""
from __future__ import annotations

import torch

from config import default_cfg
from src import encoders
from src.model import build_retrieval


def main() -> None:
    cfg = default_cfg()
    dev = torch.device("cpu")

    print("== build retrieval heads ==")
    m = build_retrieval(cfg).eval()
    print("  trainable params:", sum(p.numel() for p in m.parameters()))

    print("== text encoder ==")
    tmodel, ttok = encoders.load_text_encoder(
        cfg.model.text_encoder_path, dev, cfg.data.caption_field_labels)
    h_t, valid_t = encoders.encode_text_batch(
        tmodel, ttok, ["FUNCTION: hydrolyzes ATP", "LINEAGE: Bacteria"],
        dev, cfg.data.max_text_tokens)
    print("  h_t", tuple(h_t.shape), "kept/row", valid_t.sum(1).tolist())

    print("== ESM-C protein encoder (needs `esm`) ==")
    pmodel, ptok = encoders.load_protein_encoder(cfg.model.protein_encoder_path, dev)
    seqs = ["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", "MSKGEELFTGVVPILVELDGDVNGHKF"]
    enc = ptok(seqs, padding=True, truncation=True,
               max_length=cfg.data.max_protein_tokens, return_tensors="pt")
    ids, attn = enc["input_ids"], enc["attention_mask"]

    # Mask ~40% of the second sequence's residues to exercise the masked path.
    masked = ids.clone()
    maskable = attn.bool() & ~torch.isin(ids, torch.tensor(sorted(ptok.all_special_ids)))
    pos = torch.where(maskable[1])[0]
    masked[1, pos[::2]] = ptok.mask_token_id  # every other residue -> <mask>

    logits, h_p = encoders.forward_esmc(pmodel, masked, attn)
    _, valid_p = encoders.encode_protein_tokens(pmodel, ptok, masked, attn)
    print("  logits", tuple(logits.shape), "h_p", tuple(h_p.shape))
    print("  revealed positions/row (masked excluded):", valid_p.sum(1).tolist())

    print("== end-to-end projection + a FILIP-style score ==")
    z_p, z_t = m.project(h_p, h_t)
    # crude symmetric mean-of-max over valid tokens, just to confirm shapes wire up
    sim = torch.einsum("bld,cmd->bclm", z_p, z_t)
    sim = sim.masked_fill(~valid_p[:, None, :, None], float("-inf"))
    sim = sim.masked_fill(~valid_t[None, :, None, :], float("-inf"))
    p2t = sim.max(dim=3).values.clamp_min(-1e4)
    score = (p2t * valid_p[:, None, :]).sum(-1) / valid_p.sum(1).clamp_min(1)[:, None]
    print("  score matrix [n_prot, n_text]:", tuple(score.shape))
    print("  scores:\n", score)
    print("\nSMOKE_OK")


if __name__ == "__main__":
    main()
