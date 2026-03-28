# Rebuttal Experiments

Experiments required to address reviewer concerns. Organized by priority (critical first), with cross-references to which reviewer(s) each experiment addresses.

---

## EXP-1: Inference Cost Analysis (UNDO Token Statistics)

**Addresses:** e2Aq-W5/Q5, DLhi-W4/Q4, cVJs-W2, fZNn-Q2

**Why:** All four reviewers request efficiency/cost analysis. This is the most universally raised concern.

**What to measure (per dataset, per model):**
- Average number of `<UNDO>` tokens generated per sequence
- Average total tokens generated (including `<UNDO>`) vs. standard AR baseline
- Distribution of `<UNDO>` counts (histogram)
- Portion of `<UNDO>` tokens that actually delete erroneous content vs. unnecessary backtracking
- Wall-clock latency per sample (N-MARS vs. standard SFT vs. ReVISE)
- Throughput (samples/sec)

**How:** Run inference on GSM8K/MATH/MBPP test sets with trained N-MARS models (all 3 backbones). Parse raw trajectories before stack post-processing to extract `<UNDO>` statistics. Time each sample.

**Estimated compute:** Minimal — just inference on existing models with logging.

---

## EXP-2: Compute-Matched Comparison (Test-Time Scaling Fairness)

**Addresses:** e2Aq-W4/Q4, DLhi-W4/Q4, fZNn-Q1

**Why:** Reviewers argue that N-MARS generates more tokens (due to `<UNDO>`), so improvements may come from extra compute, not the mechanism itself.

**What to run:**
- **Self-consistency / majority voting baseline** at matched inference FLOPs:
  - For each N-MARS test sample, compute total tokens generated (including `<UNDO>`)
  - Allow the standard SFT/ReVISE baseline to sample multiple independent completions using the same total token budget
  - Apply majority voting over the multiple samples
- Report: accuracy vs. total inference tokens (or FLOPs) curve for both N-MARS and self-consistency
- Use the `compute_budget.py` script to measure FLOPs

**Models:** Llama-3.2-1B (primary), optionally Qwen3-4B

**Estimated compute:** ~2-4h per model on a single A40 (multiple inference runs with varying sample counts).

---

## EXP-3: Self-Backtracking Baseline Comparison

**Addresses:** e2Aq-W2/Q2

**Why:** Reviewer specifically flags the missing comparison with Self-Backtracking (arXiv:2502.04404) as a critical weakness.

**What to run:**
- Train Self-Backtracking on GSM8K using our implementation (`src/n_mars/baselines/self_backtracking/`)
- Evaluate with both greedy decoding and backtracking decoding (b=1, n=32)
- Compare against N-MARS on GSM8K

**Status:** Implementation complete. CRC script ready (`scripts/crc/self_backtrack.sh`).

**Models:** Llama-3.2-1B

**Estimated compute:** ~10h on a single A40.

**Expected deliverable for rebuttal:**
- Table comparing N-MARS vs. Self-Backtracking (greedy + backtracking) on GSM8K
- Discussion of key differences: token-level vs. step-level, mSFT+GRPO vs. SFT+expert iteration, single-pass vs. multi-round search

---

## EXP-4: Gradient Cosine Similarity Analysis

**Addresses:** e2Aq-W3/Q3

**Why:** Reviewer questions the unverified gradient non-alignment assumption (B.2) in Proposition 2.1.

**What to measure:**
- During mSFT training, compute cosine similarity between:
  - Error-token gradients (`g_e`) and `<UNDO>` token gradients (`g_bk`)
  - Error-token gradients (`g_e`) and correction token gradients (`g_c`)
- Log these across training steps
- Report: mean cosine similarity, distribution, how it evolves during training
- If cosine similarities are consistently negative, the assumption holds

**How:** Add gradient logging hooks to the mSFT training loop. Compute per-batch gradient decomposition on augmented traces.

**Models:** Llama-3.2-1B (sufficient for theoretical validation)

**Estimated compute:** ~2h (one training run with gradient logging overhead).

---

## EXP-5: Natural Language Self-Correction Baseline (Data-Matched)

**Addresses:** DLhi-W1/Q1, DLhi-W3/Q3

**Why:** Reviewer asks whether the explicit `<UNDO>` token is necessary, or if natural-language revision markers (e.g., "Wait, that was incorrect, let me revise") would achieve the same effect.

**What to run:**
- Take the same augmented traces used for N-MARS training
- Replace `<UNDO>` tokens with natural language markers:
  - e.g., error tokens + "Wait, that is wrong. Let me reconsider." + correction tokens
- Train with standard SFT (no masking needed since error tokens are now "natural")
- Also train with mSFT (mask the error tokens, same as N-MARS)
- Evaluate on GSM8K

**Models:** Llama-3.2-1B

**Estimated compute:** ~2h per variant (2 variants = ~4h total).

---

## EXP-6: UNDO Behavioral Analysis (Qualitative + Quantitative)

**Addresses:** cVJs-W5 (minor), DLhi-W4/Q4, fZNn-Q2

**Why:** Reviewers want more insight into how the model actually uses `<UNDO>`.

**What to report:**
- Average correction depth (how many tokens are erased per backtrack event)
- What types of errors trigger `<UNDO>` (arithmetic errors, wrong variable, formatting, etc.)
- How early in the sequence does backtracking occur (early vs. late)
- Qualitative examples showing: (a) successful correction, (b) unnecessary backtracking, (c) missed errors
- Breakdown by $\kappa$ settings

**How:** Analyze raw trajectories from EXP-1 inference runs. Classify error types manually for a subset (~50-100 samples).

**Estimated compute:** Negligible (analysis of existing inference outputs).

---

## Summary Table

| Exp | Reviewers | Priority | Estimated Time | Dependencies |
|-----|-----------|----------|---------------|--------------|
| EXP-1: Inference cost analysis | All four | Critical | ~1h | Existing models |
| EXP-2: Compute-matched comparison | e2Aq, DLhi, fZNn | Critical | ~4h | EXP-1 results |
| EXP-3: Self-Backtracking baseline | e2Aq | Critical | ~10h | Implementation ready |
| EXP-4: Gradient cosine similarity | e2Aq | High | ~2h | mSFT training code |
| EXP-5: NL self-correction baseline | DLhi | High | ~4h | Augmented traces |
| EXP-6: UNDO behavioral analysis | cVJs, DLhi, fZNn | Medium | ~1h | EXP-1 outputs |

**Total estimated compute: ~22h on a single A40**

---

## Execution Order

Recommended order to maximize parallelism and dependency resolution:

1. **EXP-1** first (provides data for EXP-2 and EXP-6)
2. **EXP-3** in parallel with EXP-1 (independent, already implemented)
3. **EXP-4** in parallel with EXP-1 (independent)
4. **EXP-2** after EXP-1 completes (needs token budget data)
5. **EXP-5** in parallel with EXP-2 (independent)
6. **EXP-6** after EXP-1 completes (analysis of inference outputs)

With 2 GPUs running in parallel: **~12-14h total wall-clock time.**

---

## Notes

- All experiments use Llama-3.2-1B as the primary model unless otherwise noted. This is the most constrained model (smallest capacity), so improvements are most visible and results are fastest to obtain.
- EXP-1 and EXP-6 can largely be addressed with text/tables in the rebuttal (no new figures needed).
- EXP-2 would benefit from an accuracy-vs-compute plot (new figure for rebuttal).
- EXP-3 requires a new row in Table 2 for the Self-Backtracking baseline.
- EXP-4 can be presented as a small supplementary figure showing gradient cosine similarities over training.
- EXP-5 adds 1-2 rows to an ablation table.
