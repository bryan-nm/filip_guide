# filip_guide — Design Plan

Prompt-based steering of ESM-C generation via a FILIP-style text↔protein
co-embedding used as a Bayesian guidance classifier (ProteinGuide, Xiong et al.,
*Nat. Biotechnol.* 2026).

Modeled structurally on `mini-embed-filip` (retrieval half) and porting the
*ideas* — not the code — from `proteingen` (ProteinGuide guidance half).

---

## 1. One-paragraph statement

Train a FILIP late-interaction co-embedding of protein sequences and text
prompts in which the **protein encoder is ESM-C itself** (frozen), so the
classifier shares a representation space with the unconditional generator.
Express the co-embedding as a probability `p(prompt | sequence)`. At generation
time, decode from ESM-C by masked iterative unmasking; at each decoded position,
reweight ESM-C's unconditional token distribution `p(x̃|x)` by the classifier
likelihood `p(prompt | x̃)^γ` (Bayes rule, ProteinGuide Eq. 15–16). The one novel
retrieval-side requirement is that the co-embedding must be trained over
sequences masked at a fraction `~Unif[0,1)` of positions and must score FILIP
similarity sensibly when positions are masked.

Guidance object (MLM/AO-ARM case, ProteinGuide Eq. 16):

```
p(x̃ | x, prompt)  ∝  p(x̃ | x) · p(prompt | x̃)^γ
     └ ESM-C prior ┘        └ FILIP classifier ┘
log-space:  log p(x̃|x)  +  γ · log p(prompt | x̃)     (DEG: add per-candidate)
```

---

## 2. Resolved design decisions

| # | Decision | Choice |
|---|----------|--------|
| 1 | `p(prompt\|seq)` form | **Both behind a config flag.** Default = softmax over a fixed reference prompt bank; sigmoid `σ(a·s+b)` as an ablation/cheap fallback. |
| 2 | Masked positions in FILIP score | **Exclude.** Masked (not-yet-decoded) positions are dropped from the token-level max-sim, like pad. Guidance is meaningful only once ≥1 position is revealed. |
| 3 | Dependency / ESM-C loading | **Standalone.** Reimplement the DEG/sampling core native to this repo against HF-transformers `ESMCForMaskedLM`, loading the local `ESMC-300M` checkpoint directly. No `proteingen`/`esm`-package dependency. |
| 4 | Guidance algorithm | **DEG first** (exact enumeration; matches "propose → score → reweight"), TAG later behind the same predictor interface. |
| 5 | Protein→text (text generation) direction | **Dropped** for now. FILIP score stays *symmetric* for training quality; only the sequence-conditioned prompt score is used at guidance time. |
| 6 | Toolchain | Mirror `mini-embed-filip`: dataclass `config.py` owning all paths, flat `src/`, pure PyTorch, manual training loop, `--device cpu/cuda/xpu`. |

---

## 3. What carries over vs. what is new

**Reused ~verbatim from `mini-embed-filip`:**
- `model.py`: `ProjectionHead` (d_in→512→256→16, L2-normalized, mean-centered),
  `ExpansionHead`, `MiniEmbedFilip` container, `logit_scale`. Retarget `d_in` to
  ESM-C's 960.
- FILIP score core, symmetric InfoNCE, reconstruction + token-uniformity aux
  losses, mean-centering discipline.
- `config.py` path-ownership pattern; two-phase trainer skeleton (R1 warmup,
  R2 InfoNCE); `evaluate.py` recall@k / mAP.
- Frozen text encoder **BioLinkBERT-base** + text projection head + caption
  field-label special-token handling — unchanged.

**New or materially changed:**
- **Protein encoder swap to ESM-C** (HF `ESMCForMaskedLM`). Drop AMPLIFY's
  `trust_remote_code`, xformers stub, and SDPA monkeypatches — HF ESM-C is clean.
- **[0,1) protein masking in the data path** (ported from `proteingen/data.py`:
  `uniform_time`, `uniform_mask_noise`).
- **Mask-aware FILIP score** (Decision 2: exclude masked positions).
- Recompute the **centering buffer for ESM-C** features (over masked + unmasked
  positions).
- Entire **generation side** written native to HF ESM-C (ideas ported from
  `proteingen` `guide.py` / `sampling.py`).

**Dropped:**
- All of mini-embed's decoder/generation/RL: `decoder_adapters`,
  `train_generation`, `train_rl`, `generate*`, `best_of_n`, `roundtrip_eval`.
- All of proteingen's `finetune_esm3/` and LoRA plumbing.

---

## 4. Repository layout

```
filip_guide/
  PLAN.md                   # this file
  config.py                 # dataclass config; owns ESMC-300M + BioLinkBERT paths
  environment.yml           # conda + pip-only block (mirror mini-embed)
  src/
    encoders.py             # frozen ESM-C (HF) + BioLinkBERT loaders; valid/mask utils
    masking.py              # uniform_time(), uniform_mask_noise(); [0,1) fraction masking
    model.py                # ProjectionHead / ExpansionHead / MiniEmbedFilip (960-d in)
    losses.py               # filip_score_matrix (MASK-AWARE) + InfoNCE + recon + uniformity
    data.py                 # pairs; text precompute; protein live+masked; group-aware splits
    train_retrieval.py      # two-phase manual loop with masking
    evaluate.py             # recall@k / mAP; recall-vs-mask-fraction curves
    # ---- generation side (native HF ESM-C) ----
    esmc_generative.py      # ESM-C prior: get_log_probs, differentiable_embedding
    filip_predictor.py      # MiniEmbedFilip -> log p(prompt|seq); set_target(prompt); 2 modes
    prompt_bank.py          # build/cache reference-prompt embeddings for the softmax partition
    guide.py                # DEG (+ TAG stub): log p(x̃|x) + γ·log p(prompt|x̃)
    sample.py               # masked iterative-unmasking loop (deterministic order)
    generate.py             # CLI: prompt -> guided sequence(s)
  pbs_common.sh, *.pbs      # launch pattern from mini-embed, with --device abstraction
```

---

## 5. Retrieval side (training the classifier)

### 5.1 Encoders (`encoders.py`)
- Protein: `AutoModelForMaskedLM.from_pretrained(ESMC-300M)`, frozen, `eval()`.
  Expose (a) per-token hidden states `[B,L,960]` for the FILIP head, and (b) MLM
  logits `[B,L,64]` for the generative prior — **one forward yields both**.
  Valid mask drops pad/cls/eos/unk/`|` and (Decision 2) `<mask>` (id 32).
- Text: BioLinkBERT `AutoModel`, frozen; last hidden state `[B,L,768]`; caption
  field-label special tokens registered (mean-init), masked out of comparison.

### 5.2 Masking (`masking.py`)
- Per example draw `t ~ Unif[0,1]`; mask each maskable protein position with
  prob `(1−t)` → mask fraction sweeps `[0,1)`, calibrating `p(y|x_t)` across the
  full sampling schedule (paper: training mask dist must match sampling dist).
- Preserve special tokens (cls/eos). Text is **never** masked.

### 5.3 Mask-aware FILIP score (`losses.py`, Decision 2)
Extend `filip_score_matrix`: the protein valid-mask additionally excludes
`<mask>` positions before the `max`. Concretely:
```
sim = einsum("bld,cmd->bclm", z_p, z_t)
sim.masked_fill_(~valid_p_excluding_masked, -inf)   # <-- masked positions dropped
sim.masked_fill_(~valid_t, -inf)
score_p2t = (sim.max(dim=3).values).sum(2) / n_revealed_p     # avg over REVEALED positions
score_t2p = (sim.max(dim=2).values).sum(2) / n_valid_t
score     = 0.5 * (score_p2t + score_t2p)
```
Degenerate all-masked sequence → no revealed positions → score is neutral
(guidance no-op); handle explicitly.

### 5.4 Losses / trainer (`train_retrieval.py`)
- Phase R1 warmup: `1 − positive_FILIP + uniformity + recon` (keep).
- Phase R2 main: symmetric FILIP InfoNCE + small align aux + recon (+ optional
  uniformity), in-batch negatives; hard-negative mining as a later phase.
- **Compute consequence:** random masking defeats mini-embed's full
  embedding-precompute. Plan: **precompute text-side token embeddings once**
  (text unmasked); **run frozen ESM-C live under `no_grad` each step** on masked
  protein sequences, detach, train heads on its output. Optional cache of a few
  discrete mask levels {0, .25, .5, .75} for fast local iteration.
- AdamW, cosine-warmup, grad-clip 1.0; `--device` abstraction; keep
  print-based `[R2-NCE] ...` logging + JSON train-log checkpoints (no wandb).

### 5.5 Evaluation (`evaluate.py`)
- FILIP recall@{1,5,10} / mAP with multi-positive grouping (as mini-embed).
- **New:** recall-vs-mask-fraction curves — verify the classifier degrades
  gracefully as more positions are masked (its guidance-time operating regime).

---

## 6. Generation side (guided decoding)

### 6.1 ESM-C prior (`esmc_generative.py`)
Thin wrapper: `get_log_probs(seq) -> log p(·|seq)` `[B,L,64]` from MLM logits;
`differentiable_embedding` (one-hot → embed → transformer) for future TAG.

### 6.2 FILIP predictor (`filip_predictor.py`, Decision 1 — both modes)
`log p(prompt | seq)` as a per-sequence scalar; prompt set via `set_target`.
- **softmax-over-bank (default):**
  `log p(y|x) = s(x,y)/τ − logsumexp_{y'∈bank}(s(x,y')/τ)`; bank text-embeddings
  cached (`prompt_bank.py`). Per-candidate partition `Z(x̃)` recomputed cheaply
  on protein embeddings DEG already produces.
- **sigmoid (ablation):** `log σ(a·s(x,y)+b)`, independent per prompt; calibrate
  `(a,b)` post-hoc on matched/mismatched validation pairs (or reuse
  `logit_scale`).

### 6.3 Guidance (`guide.py`) — DEG
At the decoded position, enumerate real AA tokens; for each candidate `x̃`,
`log p(x̃|x) + γ·log p(prompt|x̃)`; renormalize; sample. Batch the `S≈20`
candidates into one ESM-C forward. **Encoder sharing:** that same forward
supplies both the prior logits and the FILIP hidden states. TAG stub behind the
same interface for later.

### 6.4 Sampler (`sample.py`)
Masked iterative unmasking from a fully-masked sequence; **deterministic decode
order** (ascending position, per paper's AO-ARM reproducibility note); low
sampling temperature (~0.1); `γ ∈ {1,10,100}` config (start 1). `generate.py`
is the prompt→sequence CLI.

---

## 7. Milestones

1. **Scaffold** — `config.py`, `encoders.py` (ESM-C + BioLinkBERT load + masks),
   `model.py` retargeted to 960-d, CPU smoke test on a few pairs. **[DONE]**
   Verified: one ESM-C forward yields both MLM logits and FILIP features; masked
   positions excluded from the valid mask (Decision #2); heads + FILIP score wire
   end-to-end. Env: dedicated `filip_guide` conda env (py3.12, esm 3.3.0 pulls
   transformers 4.57.6, torch 2.13). Run `python smoke_test.py`.
2. **Mask-aware retrieval** — `masking.py`, mask-aware `losses.py`, `data.py`
   (live masked protein; text live for now), `evaluate.py`, `train_retrieval.py`;
   overfit a tiny subset; recall-vs-mask curve sane. **[DONE]** Two-phase loop
   runs on MPS/CPU; tiny-subset overfit shows mAP rising and a graceful
   recall-vs-mask curve (mAP 0.25 at t=1.0 -> 0.16 at t=0, no collapse). The
   FILIP score itself is unchanged — mask-awareness lives entirely in the protein
   valid mask (excludes `<mask>`; Decision #2). `filip_score_matrix` is reused
   verbatim from mini-embed. Run: `python -m src.train_retrieval --subset-size 32
   --overfit --center`. NOTE: text-side embedding cache and full-corpus GPU
   training are deferred to Milestone 5.
3. **Predictor** — `filip_predictor.py` (both modes) + `prompt_bank.py`;
   sanity-check `p(prompt|seq)` ranks true prompt high across mask fractions.
   **[DONE, mechanics]** `FilipPredictor.log_prob` returns a per-sequence scalar
   in both softmax_bank (normalized over a cached reference bank) and sigmoid
   modes, and is well-defined on masked sequences (untrained model sits at the
   uniform log(1/M) baseline, as expected). Ranking quality awaits a trained
   checkpoint. `scale = exp(logit_scale)` is the trained temperature.
4. **Guided generation** — `esmc_generative.py`, `guide.py` (DEG), `sample.py`,
   `generate.py`; qualitative: does prompt steer sequence content?
5. **Eval + scale** — recall metrics, guidance ablations (γ, τ, softmax-vs-
   sigmoid, exclude-vs-include masked as a flag), PBS training run.
   **[distributed trainer + PBS DONE]** `src/train_retrieval.py` is now
   distributed-capable (Aurora Intel XPU / oneCCL, 16 nodes x 12 ranks = 192, in
   12 subgroups of 16), ported from mini-embed: manual data-parallel
   (`src/dist.py` broadcast + coalesced grad all-reduce, no DDP), grouped-negative
   InfoNCE (`phase_r2_loss_grouped`), sharded feature-means, bf16 autocast,
   `--resume auto`. Degrades to single-rank on MPS/CPU (verified). Launch:
   `pbs_common.sh` + `train_retrieval.pbs` (full 456k corpus, 20 epochs,
   B=8 x group16 = 128 negatives/anchor). ESM-C is encoded LIVE every step (no
   cache — masking). **The full training run is executed on Aurora by the user;
   nothing is trained locally.** Remaining M5: guidance ablations after M4.

---

## 8. Open questions / to revisit
- **Guidance-strength calibration:** interplay of `γ` (exponent) and `τ` (FILIP
  temperature); may need per-prompt normalization if `p(y|x)` has too little
  mass under the prior (paper's caution).
- **Prompt-bank composition & size** for the softmax partition (representative
  SwissProt captions; dedup; a few thousand).
- **TAG feasibility** through ESM-C (differentiability, cost) once DEG works.
- **Protein→text** direction — revisit later if needed.
- **Cost of DEG** at scale (`O(L·S)` ESM-C forwards/sequence); acceptable at
  research scale, TAG is the escape hatch.
