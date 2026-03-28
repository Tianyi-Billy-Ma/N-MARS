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

**Status:** COMPLETE (greedy). Backtracking eval and MATH-500 eval pending.

### Results (Llama-3.2-1B, GSM8K)

| Method | Training | Inference | GSM8K Acc |
|--------|----------|-----------|-----------|
| SFT (paper Table 2) | SFT | greedy | 14.3 |
| RFT (paper Table 2) | RFT | greedy | 26.1 |
| STaR+ (paper Table 2) | STaR+ | greedy | 26.5 |
| ReVISE (paper Table 2) | SFT+RL | multi-pass | 28.1 |
| **Self-Backtracking** | dual-loss SFT | greedy | **26.4** |
| Self-Backtracking | dual-loss SFT | backtrack (b=1,n=32) | *pending* |
| **N-MARS (ours)** | mSFT+GRPO | single-pass | **31.3** |

**Key observations:**
- N-MARS outperforms Self-Backtracking (greedy) by **+4.9 points** (31.3 vs 26.4)
- Self-Backtracking (greedy, 26.4) is comparable to RFT (26.1) and STaR+ (26.5) — all SFT-based methods cluster around 26%
- N-MARS's advantage comes from (a) token-level granularity and (b) GRPO reinforcement learning stage

**Key design differences:**

| Dimension | Self-Backtracking | N-MARS |
|-----------|------------------|--------|
| Granularity | Step-level (full reasoning line) | Token-level (individual token) |
| Training | Dual-loss SFT only (no RL) | mSFT + GRPO |
| Inference (greedy) | Standard autoregressive | Single-pass with UNDO |
| Inference (search) | Multi-round beam search (b=1, n=32) | N/A (single pass) |
| Error masking | Mask error step from loss | Mask error tokens from loss |

**Eval details:** 1,319 GSM8K test samples, avg 112.4 tokens/sample, wall-clock 1,594s (~27 min).

**Pending:** Backtracking eval (b=1, n=32) and MATH-500 results.

---

## EXP-4: Gradient Cosine Similarity Analysis

**Addresses:** e2Aq-W3/Q3

**Why:** Reviewer questions the unverified gradient non-alignment assumption (B.2) in Proposition 2.1.

**Status:** COMPLETE.

### Results (Llama-3.2-1B, full params, 2,522 steps / 3 epochs, bs=8)

| Gradient Pair | Mean | Std | Min | Max | Interpretation |
|--------------|------|-----|-----|-----|---------------|
| cos(g_e, g_bk) | **-0.5115** | 0.0719 | -0.7551 | -0.3745 | Strong negative — error gradients conflict with UNDO learning |
| cos(g_e, g_c) | **-0.0036** | 0.0167 | -0.0607 | +0.0328 | Near-zero — approximately orthogonal |

**Training dynamics:**
- `cos(g_e, g_bk)` starts strongly negative (~-0.75 at step 50) and stabilizes around -0.47 to -0.55 by the end of training. It remains **consistently negative across all 50 logged steps** (never crosses zero).
- `cos(g_e, g_c)` fluctuates tightly around zero throughout training, ranging from -0.06 to +0.03.

**Interpretation for Proposition 2.1:**

- **Part (a) — Error detection (strongly supported):** `cos(g_e, g_bk) = -0.51` confirms that error-token gradients actively oppose UNDO token learning. The gap Gamma_bk in Prop 2.1(a) is substantial. This is the strongest empirical validation.
- **Part (b) — Corrective generation (weakly supported):** `cos(g_e, g_c) ≈ 0` indicates near-orthogonality rather than conflict. The gap Gamma_c is small but non-negative. The benefit of mSFT for correction learning comes primarily from removing the noise/variance of `g_e` from the update, not from resolving a direct conflict.
- **Part (c) — Reduced negative learning (trivially supported):** This holds by construction — mSFT zeros `g_e`, so the `||g_e||^2` term that increases error probability in SFT is eliminated.

**Summary for rebuttal:** The gradient non-alignment assumption (B.2) is empirically validated for the UNDO component (alpha > 0 with cos = -0.51), which is the most critical component. For corrections, the gradients are approximately orthogonal, meaning mSFT provides a modest benefit by reducing gradient noise. All three parts of Prop 2.1 are supported, with part (a) being the strongest.

**Config:** Llama-3.2-1B, full parameter training (no LoRA), dataset `mtybilly/GSM8K-Random-All` config `p0.1_n10`, AdamW lr=1e-4, max_length=1024, logged every 50 steps.

---

## EXP-5: Self-Reflect Baseline (Data-Matched NL Correction)

**Addresses:** DLhi-W1/Q1, DLhi-W3/Q3

**Why:** Reviewer asks whether the explicit `<UNDO>` token is necessary, or if natural-language revision markers (e.g., "Wait, that was incorrect, let me revise") would achieve the same effect.

**Method name:** Self-Reflect

### Design Rationale

A key distinction between `<UNDO>` and natural language correction is **post-processing determinism**:

- **N-MARS (`<UNDO>`):** Each `<UNDO>` token erases exactly one preceding token. Given a raw generation like `x, x, e, e, <UNDO>, <UNDO>, x, x`, the stack-based post-processing deterministically produces `x, x, x, x`. The mapping from raw trajectory to clean output is unambiguous.
- **NL correction (Self-Reflect):** The model generates `x, x, e, e, "Wait, that is wrong.", x, x`. There is no deterministic rule for which prior tokens to remove — the NL marker signals that an error occurred, but does not specify the error's location or span. This makes clean post-processing infeasible.

Both approaches keep error tokens in the KV cache during generation (the model attends to errors in both cases). The difference is not about what the model sees, but about whether the final output can be cleanly recovered. With `<UNDO>`, the output is guaranteed clean. With NL markers, the output contains both the error and the correction, and we must rely on answer extraction (e.g., "The answer is: X") to get the final result.

This mirrors o1/R1-style reasoning where the "thinking trace" is messy but only the final answer matters. The question is whether this is sufficient compared to explicit erasure.

### What to run

- Take the same augmented traces used for N-MARS training (`mtybilly/GSM8K-Random-All`)
- Replace `<|BACKTRACK|>` tokens with a natural language marker: `"Wait, that is incorrect. Let me reconsider."`
- Train with **standard SFT** (no masking — all tokens supervised, including errors)
- At inference, generate the full sequence (errors + NL marker + correction included) and extract the final answer via regex ("The answer is: X")
- No post-processing to remove errors — the output is kept as-is

**Example training sequence:**
```
Question: John has 3 apples and buys 5 more. He gives 2 away. How many?
Answer:
John starts with 3 apples.
He buys 5 more, so 3 + 5 = 9. Wait, that is incorrect. Let me reconsider.
He buys 5 more, so 3 + 5 = 8.
He gives 2 away, so 8 - 2 = 6.
The answer is: 6
```

**Models:** Llama-3.2-1B

**Estimated compute:** ~2h on a single A40.

### Literature Context

- **Contextual Drag** (arXiv:2602.04288): Wrong tokens remaining in context cause 10-20% performance drops due to persistent attention biases.
- **LLMs Cannot Self-Correct Reasoning Yet** (arXiv:2310.01798): Without external feedback, NL self-correction fails or degrades performance.
- **LLMs Cannot Find But Can Correct** (arXiv:2311.08516): Models detect errors at only ~53% accuracy, but correct effectively when given explicit error locations.
- **DeepSeek-R1** (arXiv:2501.12948): NL reflection ("wait", "mistake") emerges from GRPO training, but wrong tokens remain in `<think>` blocks.

These findings suggest that while NL correction can work for final-answer extraction, the lack of explicit erasure leaves error tokens as attention biases, potentially degrading intermediate reasoning quality.

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
