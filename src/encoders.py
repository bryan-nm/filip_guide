"""Load and run the two frozen encoders, returning per-token hidden states.

BioLinkBERT-base: standard HuggingFace BertModel; we return its full last hidden
state along with a "valid token" mask that excludes [PAD], [CLS], [SEP] and the
caption field-label special tokens.

ESM-C 300M: loaded through the transformers API (`AutoModelForMaskedLM`), which
is the officially documented path (biohub/ESMC-300M). The `esm` package
(`pip install esm@git+https://github.com/Biohub/esm.git@main`) registers the
`esmc` architecture into transformers; without it, `from_pretrained` raises
"does not recognize this architecture". We only use the transformers API — one
forward yields BOTH `.logits` [B,L,64] (the unconditional MLM prior p(x) used at
generation time) and `.hidden_states[-1]` [B,L,960] (the features the FILIP
protein head consumes). ESM-C is the generator AND the retrieval encoder.

Masking convention (Decision #2): masked (`<mask>`) positions are treated as
special and EXCLUDED from the returned valid mask, so they drop out of the FILIP
token-level max-sim. `include_masked=True` keeps them for the ablation.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn


def _quiet_transformers() -> None:
    """Raise transformers' log threshold to ERROR before loading the encoders.

    Suppresses the cosmetic spam that otherwise floods the job log: the
    "🚨 No checkpoint found for ESMCForTokenClassification.forward ... auto_docstring"
    advisories (emitted once per rank for ESM-C's unused classification heads —
    NOT a weight-loading or checkpoint problem) and the BioLinkBERT
    embedding-resize notice. Real errors are ERROR-level and still surface.
    """
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Text encoder: BioLinkBERT-base
# ---------------------------------------------------------------------------
def _register_caption_field_labels(model, tokenizer,
                                   field_labels: Sequence[str]) -> List[str]:
    """Register caption field labels (e.g. "FUNCTION:") as single special tokens.

    The captions are templated with UPPERCASE "LABEL:" prefixes carrying no
    biological/textual content. Making each a single `additional_special_token`
    (a) lets the tokenizer emit it as one id, and (b) lets cross-modal masking
    drop the whole label by id (see `_text_valid_mask`).

    The encoder is frozen, so the new embedding rows would never train; we
    mean-init each from the sub-word pieces the label used to tokenize into.
    Idempotent: labels already mapping to a single id are skipped. Returns the
    list of newly added labels; the full label->id map is stashed on the
    tokenizer as `_caption_field_label_ids`.
    """
    if not field_labels:
        tokenizer._caption_field_label_ids = []
        return []

    unk = tokenizer.unk_token_id
    fresh: List[str] = []
    pieces: dict = {}
    for lab in field_labels:
        cur = tokenizer.convert_tokens_to_ids(lab)
        if cur is not None and cur != unk:
            continue                                   # already a single token
        sub = tokenizer(lab, add_special_tokens=False)["input_ids"]
        if not sub:
            continue
        fresh.append(lab)
        pieces[lab] = sub                              # captured BEFORE we add it

    if fresh:
        tokenizer.add_special_tokens({"additional_special_tokens": fresh})
        model.resize_token_embeddings(len(tokenizer))
        emb = model.get_input_embeddings().weight.data
        for lab in fresh:
            new_id = tokenizer.convert_tokens_to_ids(lab)
            emb[new_id] = emb[pieces[lab]].mean(dim=0)

    tokenizer._caption_field_label_ids = [
        int(tokenizer.convert_tokens_to_ids(lab)) for lab in field_labels
    ]
    return fresh


def load_text_encoder(path: str, device: torch.device,
                      field_labels: Optional[Sequence[str]] = None):
    """Load the frozen text encoder + tokenizer.

    `field_labels` (caption-schema "LABEL:" prefixes) are registered as single
    special tokens if given. Masking of those positions is controlled separately
    at encode time (`encode_text_batch(mask_field_labels=...)`).
    """
    from transformers import AutoModel, AutoTokenizer

    _quiet_transformers()
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path).eval().to(device)
    _register_caption_field_labels(model, tok, field_labels or [])
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def _field_label_ids(tokenizer) -> set:
    ids = getattr(tokenizer, "_caption_field_label_ids", None)
    if ids:
        return {int(i) for i in ids if i is not None and int(i) >= 0}
    try:
        extra = tokenizer.additional_special_tokens_ids
    except Exception:
        extra = None
    return {int(i) for i in extra if i is not None and int(i) >= 0} if extra else set()


def _text_valid_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                     tokenizer, mask_specials: bool,
                     mask_field_labels: bool) -> torch.Tensor:
    """Bool mask: True at positions kept for FILIP / uniformity.

    `mask_specials` drops [CLS]/[SEP]/[PAD]; `mask_field_labels` independently
    drops the caption field-label special tokens.
    """
    valid = attention_mask.bool()
    drop_ids = set()
    if mask_specials:
        for tok_id in (tokenizer.cls_token_id, tokenizer.sep_token_id,
                       tokenizer.pad_token_id, tokenizer.bos_token_id,
                       tokenizer.eos_token_id):
            if tok_id is not None:
                drop_ids.add(int(tok_id))
    if mask_field_labels:
        drop_ids.update(_field_label_ids(tokenizer))
    if not drop_ids:
        return valid
    spec = torch.zeros_like(input_ids, dtype=torch.bool)
    for sid in drop_ids:
        spec = spec | (input_ids == sid)
    return valid & ~spec


def text_encoder_max_len(model) -> int:
    """Hard architectural input-length cap (BERT-family: max_position_embeddings)."""
    return int(getattr(model.config, "max_position_embeddings", 512))


@torch.no_grad()
def encode_text_batch(
    model, tokenizer, texts: List[str], device: torch.device,
    max_len: int, mask_specials: bool = True, mask_field_labels: bool = True,
):
    """Returns (h_t, valid_mask) where h_t is [B, L, 768] and mask is [B, L].

    `max_len` is silently capped at the model's max_position_embeddings.
    """
    effective_max = min(max_len, text_encoder_max_len(model))
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=effective_max,
        return_tensors="pt",
    ).to(device)
    out = model(**enc)
    h = out.last_hidden_state                            # [B, L, 768]
    valid = _text_valid_mask(enc["input_ids"], enc["attention_mask"],
                             tokenizer, mask_specials, mask_field_labels)
    return h, valid


# ---------------------------------------------------------------------------
# Protein encoder: ESM-C 300M
# ---------------------------------------------------------------------------
def _ensure_esm_registered() -> bool:
    """Import `esm` for its side effect of registering the `esmc` architecture.

    Loud on failure (unlike a silent pass): if `import esm` errors on a host, that
    is exactly why AutoModelForMaskedLM later can't find `esmc`, and we want it in
    the log rather than swallowed.
    """
    try:
        import esm  # noqa: F401
        return True
    except Exception as e:
        print(f"[encoders] WARNING: `import esm` failed ({type(e).__name__}: {e}); "
              f"the `esmc` architecture may be unregistered. If ESM-C fails to load, "
              f"install/repair esm: pip install 'esm @ git+https://github.com/Biohub/esm.git@main'",
              flush=True)
        return False


def _load_esmc_tokenizer(path: str):
    """Load the ESM-C tokenizer robustly across transformers versions.

    The checkpoint's tokenizer_config names `tokenizer_class: "ESMCTokenizer"`.
    Some transformers builds resolve that via a generic fast-tokenizer fallback;
    others (e.g. Aurora's `frameworks` build) raise "ESMCTokenizer does not exist".
    So we try AutoTokenizer, then fall back to building a plain fast tokenizer
    straight from tokenizer.json + special_tokens_map.json — which ignores the
    class name entirely and yields identical vocab/specials (cls 0, eos 2, pad 1,
    unk 3, `|` 31, mask 32; post-processor adds cls/eos).
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    try:
        return AutoTokenizer.from_pretrained(path)
    except Exception as e:
        print(f"[encoders] AutoTokenizer failed ({type(e).__name__}); falling back to "
              f"PreTrainedTokenizerFast from tokenizer.json", flush=True)
        return PreTrainedTokenizerFast.from_pretrained(path)


def load_protein_encoder(path: str, device: torch.device):
    """Load frozen ESM-C via transformers `AutoModelForMaskedLM` + its tokenizer.

    Importing `esm` registers the `esmc` architecture into transformers. The
    tokenizer is loaded version-robustly (see `_load_esmc_tokenizer`).
    """
    from transformers import AutoModelForMaskedLM

    _quiet_transformers()
    _ensure_esm_registered()
    tok = _load_esmc_tokenizer(path)
    try:
        model = AutoModelForMaskedLM.from_pretrained(path).eval().to(device)
    except (ValueError, KeyError) as e:
        raise RuntimeError(
            f"Could not load ESM-C from {path!r}: {e}\n"
            "The `esmc` architecture must be registered into transformers (via the "
            "`esm` package) AND transformers must be >=4.57. On Aurora the venv may "
            "be using the older `frameworks` transformers via --system-site-packages; "
            "force a venv-local upgrade:\n"
            "    pip install --upgrade --force-reinstall --no-deps 'transformers>=4.57'\n"
            "    pip install 'esm @ git+https://github.com/Biohub/esm.git@main'"
        ) from e

    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def _protein_valid_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                        tokenizer, mask_specials: bool,
                        include_masked: bool = False) -> torch.Tensor:
    """Bool mask: True at positions kept for the FILIP token-level max-sim.

    Special tokens (cls/eos/pad/unk/`|`/`<mask>`) are dropped. Because `<mask>` is
    a special token, masked (not-yet-decoded) positions are excluded by default —
    Decision #2. `include_masked=True` retains `<mask>` positions for the ablation
    that lets ESM-C's contextual belief at masks participate in the score.
    """
    valid = attention_mask.bool()
    if not mask_specials:
        return valid
    special_ids = {int(i) for i in getattr(tokenizer, "all_special_ids", []) if i is not None}
    if include_masked and tokenizer.mask_token_id is not None:
        special_ids.discard(int(tokenizer.mask_token_id))
    if not special_ids:
        return valid
    spec = torch.zeros_like(input_ids, dtype=torch.bool)
    for sid in special_ids:
        spec = spec | (input_ids == sid)
    return valid & ~spec


@torch.no_grad()
def forward_esmc(model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """One frozen ESM-C forward -> (logits [B,L,64], last_hidden [B,L,960]).

    `attention_mask` is the standard HF 1/0 padding mask. `.logits` is the
    unconditional MLM prior p(x) used by guided decoding; `.hidden_states[-1]` is
    the FILIP protein feature. Sharing this forward is the whole point of using
    ESM-C as both generator and encoder.
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True)
    return out.logits, out.hidden_states[-1]


@torch.no_grad()
def encode_protein_tokens(
    model, tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    mask_specials: bool = True, include_masked: bool = False,
):
    """Encode ALREADY-tokenized (and possibly masked) protein ids.

    This is the training/guidance entry point: the masking module tokenizes raw
    sequences, replaces a random fraction of residue ids with `<mask>`, then calls
    here. Returns (h_p [B,L,960], valid_mask [B,L]).
    """
    _, last = forward_esmc(model, input_ids, attention_mask)
    valid = _protein_valid_mask(input_ids, attention_mask, tokenizer,
                                mask_specials, include_masked)
    return last, valid


@torch.no_grad()
def encode_protein_batch(
    model, tokenizer, seqs: List[str], device: torch.device,
    max_len: int, mask_specials: bool = True,
):
    """Tokenize raw sequences (unmasked) and encode. Returns (h_p, valid_mask).

    Convenience for eval/inspection on clean sequences; training masks ids first
    and calls `encode_protein_tokens`.
    """
    enc = tokenizer(
        seqs, padding=True, truncation=True, max_length=max_len,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    return encode_protein_tokens(model, tokenizer, input_ids, attn, mask_specials)
