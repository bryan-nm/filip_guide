"""Central configuration for filip_guide.

Retrieval phase trains projection + expansion heads via FILIP-style late
interaction over ESM-C protein features and BioLinkBERT text features, with the
protein side masked at a random fraction of positions so the learned classifier
p(prompt | sequence) tolerates the partially-masked sequences seen during guided
generation. Generation (later milestone) decodes from frozen ESM-C by masked
iterative unmasking, reweighting each step by p(prompt | x)^gamma (ProteinGuide).

The protein encoder IS the generator: one frozen ESM-C serves both as the
unconditional prior p(x) and as the FILIP protein encoder. Both encoders stay
frozen; only the projection/expansion heads (and the logit temperature) train.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Model and dataset locations
# ---------------------------------------------------------------------------
# THIS FILE OWNS EVERY PATH (mirrors mini-embed-filip). Job scripts banner the
# output of `python config.py` rather than exporting paths on the command line,
# so the log always records which models and corpus a run actually read. The
# FILIP_* / FILIPG_* env vars are an escape hatch for hosts whose models live
# elsewhere; no job script sets them.
MODELS_DIR = os.environ.get("FILIPG_MODELS_DIR", "/flare/NLDesignProtein/bryan/FILIP-dev-space/models")
DATASETS_DIR = os.environ.get("FILIPG_DATASETS_DIR", "/flare/NLDesignProtein/bryan/FILIP-dev-space/datasets")

# One row per (protein, caption) pair. This build is filtered to 30..500 residues,
# which lets max_protein_tokens sit at 512 (+cls/eos) with nothing truncated.
DATA_CSV = os.environ.get(
    "FILIPG_DATA_CSV",
    f"{DATASETS_DIR}/SwissProt_full/fully_annotated_swiss_prot_080326.csv",
)

# ESM-C is loaded through the transformers API (AutoModelForMaskedLM); the `esm`
# package (pip install esm@git+https://github.com/Biohub/esm.git) must be present
# to register the `esmc` architecture. See src/encoders.py.
PROTEIN_ENCODER_PATH = os.environ.get("FILIPG_PROTEIN_ENCODER", f"{MODELS_DIR}/ESMC-300M")
TEXT_ENCODER_PATH = os.environ.get("FILIPG_TEXT_ENCODER", f"{MODELS_DIR}/BioLinkBERT-base")


@dataclass
class DataCfg:
    csv_path: str = DATA_CSV
    csv_id_col: str = "primary_Accession"
    csv_protein_col: str = "protein_sequence"
    csv_text_col: str = "[final]text_caption"
    csv_pfam_col: str = "pfam_label"             # optional; kept for future homology use
    subset_size: int = 0
    seed: int = 0
    splits: tuple = (0.90, 0.05, 0.05)
    max_text_tokens: int = 512
    # Corpus filtered to <=500 residues, so 512 (with cls/eos) truncates nothing.
    max_protein_tokens: int = 512

    # Caption-schema field labels: UPPERCASE "LABEL:" prefixes the captions are
    # templated with. The text tokenizer registers each as a single special token
    # (see encoders._register_caption_field_labels); cross-modal comparison masks
    # them out. Swap this list if the caption format changes.
    caption_field_labels: tuple = (
        "PROTEIN NAME:", "LINEAGE:", "GENE ONTOLOGY:", "SIMILARITY:", "FUNCTION:",
        "ENZYME CLASS:", "CATALYTIC ACTIVITY:", "PATHWAY:", "DOMAIN:", "PTM:",
        "SUBCELLULAR LOCATION:",
    )


@dataclass
class ModelCfg:
    text_encoder_path: str = TEXT_ENCODER_PATH
    protein_encoder_path: str = PROTEIN_ENCODER_PATH
    text_hidden: int = 768                    # BioLinkBERT-base
    protein_hidden: int = 960                 # ESM-C 300M (d_model)

    proj_d_hidden: int = 512
    proj_d_mid: int = 256
    # Width of the shared retrieval space, and the sweep variable. Nothing else in
    # the repo hard-codes a value — every head reads it from here. Bumped 16 -> 64
    # after the first full run collapsed (filip_pos ~0.97 with flat acc): 16-d is a
    # tight bottleneck for token-level FILIP over a 456k-protein corpus and makes
    # collapse easier. Try 128 if 64 still saturates.
    embed_dim: int = 64
    proj_dropout: float = 0.1

    expand_d_mid: int = 256
    expand_d_hidden: int = 512
    expand_dropout: float = 0.1


@dataclass
class MaskCfg:
    """Random protein masking applied during retrieval training (and matching the
    guided-decoding schedule). Per example we draw t ~ Unif[mask_t_min, mask_t_max]
    and mask each maskable residue independently with prob (1 - t), so the mask
    fraction sweeps [0,1): t=1 leaves the sequence fully revealed, t->0 fully masks
    it. Special tokens (cls/eos/pad) are never masked. This calibrates
    p(prompt | x_t) across the full range of masking the sampler will present.
    """
    enabled: bool = True
    mask_t_min: float = 0.0
    mask_t_max: float = 1.0
    # Decision #2 (resolved): masked positions are EXCLUDED from the FILIP
    # token-level max-sim. Flip to True only to run the include-masked ablation.
    include_masked_in_score: bool = False


@dataclass
class RetrievalCfg:
    """Retrieval (FILIP) training phase."""
    cache_dir: str = str(REPO_ROOT / "cache")
    ckpt_dir: str = str(REPO_ROOT / "checkpoints" / "retrieval")

    device: str = "auto"

    # Protein embeddings are mask-dependent, so ESM-C runs live each step (frozen,
    # no_grad); only text embeddings are cacheable. Batch is therefore modest.
    batch_size: int = 16
    num_workers: int = 0

    phase1_epochs: int = 1
    phase2_epochs: int = 3

    # Loss weights
    phase1_uniformity_weight: float = 0.1
    r2_uniformity_weight: float = 0.01
    align_aux_weight: float = 0.1
    recon_weight: float = 0.05
    uniformity_t: float = 2.0

    mask_text_special_tokens: bool = True
    mask_protein_special_tokens: bool = True
    mask_text_field_labels: bool = True

    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    grad_clip: float = 1.0

    init_temperature: float = 0.07
    max_temperature: float = 100.0

    # Distributed (Aurora): ranks per contrastive subgroup; global negatives per
    # anchor = group_size * batch_size. Chunk the FILIP score matrix to cap the
    # [chunk, group*bs, L_p, L_t] transient. All inert at world_size 1.
    group_size: int = 16
    filip_chunk_rows: int = 8
    val_subset: int = 1000          # per-epoch rank-0 eval on the first N val pairs
    means_sample: int = 4096        # sequences sampled to estimate centering means
    use_autocast: bool = True       # bf16 autocast on xpu/cuda (no-op on cpu/mps)

    log_every: int = 50
    eval_every_epoch: bool = True


@dataclass
class GuidanceCfg:
    """Guided generation (later milestone; scaffolded here for path ownership).

    ProteinGuide MLM/AO-ARM guidance: decode from ESM-C by masked iterative
    unmasking; at each decoded position reweight the prior by p(prompt | x)^gamma.
    """
    retrieval_ckpt: str = ""                  # trained retrieval checkpoint
    ckpt_dir: str = str(REPO_ROOT / "checkpoints" / "generation")
    device: str = "auto"

    # Decision #1 (resolved): both likelihood forms behind a flag.
    #   "softmax_bank" — p(y|x) normalized over a cached reference prompt bank.
    #   "sigmoid"      — p(y|x) = sigmoid(a*s + b), independent per prompt.
    likelihood_mode: str = "softmax_bank"
    prompt_bank_path: str = str(REPO_ROOT / "cache" / "prompt_bank.pt")

    gamma: float = 1.0                        # guidance strength exponent (paper: 1/10/100)
    sample_temperature: float = 0.1
    deterministic_order: bool = True          # ascending-position decode order


@dataclass
class Cfg:
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    mask: MaskCfg = field(default_factory=MaskCfg)
    retrieval: RetrievalCfg = field(default_factory=RetrievalCfg)
    guidance: GuidanceCfg = field(default_factory=GuidanceCfg)


def default_cfg() -> Cfg:
    return Cfg()


# ---------------------------------------------------------------------------
# Run record
# ---------------------------------------------------------------------------
def resolved_paths(cfg: Cfg | None = None) -> dict:
    cfg = cfg or default_cfg()
    # NOTE: there is no embedding cache / precompute step — ESM-C is encoded live
    # every step (masking makes it uncacheable), so cache_dir is intentionally not
    # listed here. Only inputs the run actually reads + the checkpoint sink.
    return {
        "models_dir": MODELS_DIR,
        "datasets_dir": DATASETS_DIR,
        "text_encoder": cfg.model.text_encoder_path,
        "protein_encoder": cfg.model.protein_encoder_path,
        "data_csv": cfg.data.csv_path,
        "ckpt_dir": cfg.retrieval.ckpt_dir,
    }


def describe_paths(cfg: Cfg | None = None) -> str:
    paths = resolved_paths(cfg)
    width = max(len(k) for k in paths)
    lines = []
    for k, v in paths.items():
        mark = "" if Path(v).exists() else "   [MISSING]"
        lines.append(f"{k:<{width}} = {v}{mark}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_paths())
