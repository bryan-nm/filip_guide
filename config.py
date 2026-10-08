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

# Protein encoder. Two kinds are supported (FILIPG_PROTEIN_ENCODER_TYPE):
#   "esmc"    - ESM-C 300M via transformers AutoModelForMaskedLM (needs the `esm`
#               package to register the `esmc` arch). This is ALSO the generator.
#   "amplify" - SaAMPLIFY-350M via trust_remote_code (mini-embed-filip's encoder).
#               Used as the guidance CLASSIFIER only (not generative); we can
#               initialize its FILIP heads from a mini-embed-filip checkpoint and
#               continue-train under the mask schedule. Both are 960-d hidden.
PROTEIN_ENCODER_PATH = os.environ.get("FILIPG_PROTEIN_ENCODER", f"{MODELS_DIR}/ESMC-300M")
PROTEIN_ENCODER_TYPE = os.environ.get("FILIPG_PROTEIN_ENCODER_TYPE", "esmc")
TEXT_ENCODER_PATH = os.environ.get("FILIPG_TEXT_ENCODER", f"{MODELS_DIR}/BioLinkBERT-base")

# The GENERATOR is the unconditional prior p(x) that guidance decodes from,
# separate from the FILIP CLASSIFIER's encoder. Three generators are supported:
#   "pld2" - PLD2, a looped bidirectional DISCRETE-DIFFUSION protein generator
#            (sibling repo). Unlike the two below it is not a masked LM decoded
#            one position at a time: it decodes a fixed 512 canvas in parallel,
#            committing the most confident edits per step (MaskGIT-style), and it
#            carries a second 3Di structure track that co-decodes with the
#            sequence. So it does NOT go through src/sample.py's AO-ARM loop —
#            filip_guide drives PLD2's own sampler through its `guidance_fn`
#            logit hook. See src/pld2_generative.py.
#   "esm3" - ESM3-sm (esm3-sm-open-v1) via the `esm` package. An actual generative
#            model (masked/any-order sequence design); the ProteinGuide paper's
#            unconditional sequence generator. Preferred — ESM-C folds poorly.
#   "esmc" - ESM-C 300M via transformers (representation model; kept for A/B).
# esm3/esmc share the EsmSequenceTokenizer, so the classifier + tokenizer bridge
# are identical for either; PLD2 has its own 23-token vocabulary, bridged by a
# tokenizer shim in src/pld2_generative.py (the bridge itself is unchanged).
GENERATOR_TYPE = os.environ.get("FILIPG_GENERATOR_TYPE", "esm3")
GENERATOR_ENCODER_PATH = os.environ.get("FILIPG_GENERATOR_ENCODER", f"{MODELS_DIR}/ESMC-300M")
ESM3_GENERATOR_PATH = os.environ.get("FILIPG_ESM3_PATH", f"{MODELS_DIR}/esm3-sm-open-v1")

# PLD2 lives in its own repo and is IMPORTED, not vendored — the same call PLD2
# itself makes for mini-embed-filip (its config.MINI_EMBED_REPO). PLD2 is actively
# trained, so a vendored copy of its trunk would rot; and the subgraph we need
# (src/model.py, src/sampler.py, src/corruption.py) imports nothing but stdlib and
# torch, so there is none of the env-conflict risk that made the `esm`-package
# dependency a problem. Loaded under the module alias `pld2_src` because PLD2's
# package is ALSO called `src` — see src/pld2_generative.load_pld2_package.
PLD2_REPO = os.environ.get("FILIPG_PLD2_REPO", str(REPO_ROOT.parent / "PLD2"))
PLD2_CKPT = os.environ.get("FILIPG_PLD2_CKPT", str(REPO_ROOT.parent / "PLD2" / "ali_r10_step499.ckpt"))

# Guidance-time memory bounds. Overridable from the PBS scripts because the safe
# values depend on n_candidates (the classifier batch is n_candidates * 20 rows)
# and on the host's per-tile HBM, not on anything in the model config.
BANK_CHUNK = int(os.environ.get("FILIPG_BANK_CHUNK", "32"))
CLASSIFIER_ROW_CHUNK = int(os.environ.get("FILIPG_CLASSIFIER_ROW_CHUNK", "32"))


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
    protein_encoder_type: str = PROTEIN_ENCODER_TYPE   # "esmc" | "amplify"
    text_hidden: int = 768                    # BioLinkBERT-base
    protein_hidden: int = 960                 # ESM-C 300M / SaAMPLIFY-350M (both 960-d)

    proj_d_hidden: int = 512
    proj_d_mid: int = 256
    # Width of the shared retrieval space, and the sweep variable. Nothing else in
    # the repo hard-codes a value — every head reads it from here.
    embed_dim: int = 16
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
    retrieval_ckpt: str = ""                  # trained FILIP classifier checkpoint
    generator_type: str = GENERATOR_TYPE      # "pld2" | "esm3" | "esmc"
    generator_path: str = GENERATOR_ENCODER_PATH   # ESM-C path (generator_type="esmc")
    esm3_path: str = ESM3_GENERATOR_PATH           # ESM3 path (generator_type="esm3")
    pld2_repo: str = PLD2_REPO                     # PLD2 checkout (generator_type="pld2")
    pld2_ckpt: str = PLD2_CKPT                     # PLD2 weights
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

    # Best-of-n sampling. n candidates are drawn as ONE batched decode and the
    # winner is picked by `select_by`:
    #   "path_logp"  — generator (ESM3) AO-ARM log-likelihood of the candidate.
    #                  Free (falls out of the guided score) and independent of both
    #                  the FILIP guidance objective and the ESMfold/TM-align eval,
    #                  so it does not make the conditioned-vs-unconditioned
    #                  comparison circular. The default.
    #   "filip_logp" — log p(prompt | seq) from the guidance classifier. Circular:
    #                  it is the steering objective, and applying it to the gamma=0
    #                  arm (required to keep the arms matched) smuggles conditioning
    #                  into the control. Diagnostic only.
    #   "joint_logp" — path_logp + gamma * filip_logp, i.e. the sampler's own
    #                  posterior objective; reduces to path_logp at gamma=0.
    #   "first"      — candidate 0, reproducing n=1 for an apples-to-apples delta.
    n_candidates: int = 1
    select_by: str = "path_logp"
    # Likelihood best-of-n rewards low-complexity sequences, so degenerate
    # candidates are filtered out BEFORE selection (not scored down).
    max_aa_frac: float = 0.25                 # reject if one residue exceeds this fraction
    max_homopolymer_run: int = 6              # reject if a single residue repeats > this

    # Memory bounds for the candidate batch (see FilipPredictor).
    bank_chunk: int = BANK_CHUNK              # prompts per bank-scoring block
    classifier_row_chunk: int = CLASSIFIER_ROW_CHUNK   # candidate rows per classifier block

    # --- PLD2 generator (generator_type="pld2") ---------------------------
    # PLD2 decodes a fixed canvas in PARALLEL, so its guidance hook is called once
    # per step with the whole canvas. Exact per-candidate DEG would then be
    # L * C classifier forwards per step (~5M per sample at 512x20), so guidance
    # here is TAG: ONE classifier forward + backward per step scores every
    # (position, residue) at once via the input-embedding gradient, at a cost
    # independent of length. This mirrors what PLD2 runs in production.
    pld2_guide_mode: str = "tag"              # "tag" (first-order) — DEG is not viable here
    # The gradient's SCALE moves by orders of magnitude across a decode (it depends
    # on how many positions are committed), so an unnormalized gamma would mean
    # something different at every step. Standardizing across the 20 candidates
    # puts gamma in logit units; the per-position mean cancels under the softmax.
    pld2_tag_normalize: bool = True
    pld2_tag_chunk: int = 8                   # candidate rows per classifier fwd+bwd

    # Sampler settings. Defaults are the ones the attached checkpoint's own run
    # used (PLD2 config.py sample_* / gen_* at the r10 alignment round), so a
    # filip_guide decode matches the regime the policy was tuned in.
    pld2_canvas: int = 512                    # fixed decode width (PLD2 DataCfg.canvas)
    pld2_steps: int = 512                     # decode steps (cosine commit schedule)
    pld2_temperature: float = 1.0
    pld2_gumbel_temp: float = 0.1             # stochastic edit ORDER (MaskGIT)
    pld2_subst_per_residue: float = 1.0       # substitution budget; 0 = absorbing-only
    pld2_rep_penalty: float = 0.0             # PLD2's own repetition guidance (composes before ours)
    pld2_max_run: int = 5                     # hard homopolymer run cap in the sampler
    pld2_min_len: int = 30                    # EOS floor
    pld2_eos_first: bool = True               # commit the length before any residue
    pld2_struct_first: float = 0.0            # fraction of steps biased toward the 3Di track
    pld2_n_recurrence: int = 0                # 0 = use the checkpoint's own (1 at 1.35B)


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
        "protein_encoder_type": cfg.model.protein_encoder_type,   # not a path
        # The generative prior is a separate model from the classifier's encoder,
        # and which one a run decoded from is the first thing a result needs.
        "generator_type": cfg.guidance.generator_type,            # not a path
        "generator": _generator_path(cfg),
        "data_csv": cfg.data.csv_path,
        "ckpt_dir": cfg.retrieval.ckpt_dir,
    }


def _generator_path(cfg: Cfg) -> str:
    """The weights the configured generator will actually load."""
    return {"pld2": cfg.guidance.pld2_ckpt,
            "esm3": cfg.guidance.esm3_path,
            "esmc": cfg.guidance.generator_path}.get(
                cfg.guidance.generator_type, cfg.guidance.generator_path)


def describe_paths(cfg: Cfg | None = None) -> str:
    paths = resolved_paths(cfg)
    width = max(len(k) for k in paths)
    lines = []
    for k, v in paths.items():
        # Only flag [MISSING] for actual filesystem paths, not scalar settings.
        is_path = "/" in str(v)
        mark = "" if (not is_path or Path(v).exists()) else "   [MISSING]"
        lines.append(f"{k:<{width}} = {v}{mark}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_paths())
