# Rebuttal — Reviewer fZNn

We thank the reviewer for the thorough reading and constructive feedback. We address each concern below.

---

## Q1: Compute-matched comparisons (same token budget / FLOPs / wall-clock)

**Short answer:** N-MARS's UNDO budget is explicitly bounded by the hyperparameter κ, so the overhead is controlled and moderate. A concrete efficiency comparison will be added to the camera-ready.

The UNDO budget κ limits the fraction of UNDO tokens relative to total sequence length: κ=0.2 for GSM8K and κ=0.4 for MATH (Section 4, Hyperparameter Analysis). This means at most 20%/40% additional tokens are generated above the base sequence length. Furthermore, the GRPO reward penalizes UNDO tokens that do not contribute to a correct final answer (R_pen = −1.0 when output is incorrect), which actively discourages "compute-padding" behavior.

For a fair comparison, we acknowledge the reviewer's point: ideally we should compare N-MARS against baselines given equal maximum token budgets. Since ReVISE already requires a full second generation pass (2× base compute), N-MARS with κ=0.2 uses substantially fewer tokens than ReVISE while achieving a 11.4%/13.4% relative improvement on GSM8K/MATH with Llama-3.2-1B. We will add a token-budget-matched comparison table (e.g., fixing total generated tokens to 1.2× base for both N-MARS and an extended-CoT baseline) in the camera-ready.

---

## Q2: Measured latency/throughput — UNDO counts, wall-clock per sample, κ sensitivity

**Short answer:** At κ=0.2, the worst-case inference overhead is 20% more tokens; UNDO usage is concentrated on error-prone steps, not spread uniformly. Concrete statistics will be reported in the camera-ready.

The stack-based post-processing (Section 2.4) has O(1) overhead per token — a push/pop on a stack — and does not require additional forward passes. The total generated token count is therefore bounded by (1+κ)×|base_sequence|. At the optimal κ=0.2 for GSM8K, this amounts to at most a 20% sequence-length increase per sample.

The κ sensitivity analysis (Figure 2) also reveals that model performance degrades when κ is set too high, indicating that the model learns to use UNDO sparingly and purposefully rather than padding with UNDO tokens. In the camera-ready, we will report (a) mean UNDO tokens per solved/unsolved instance, (b) wall-clock time per sample on A100 GPUs at κ∈{0.2,0.4}, and (c) a throughput comparison against ReVISE's two-pass inference.

---

## Q3: Expected challenges for long-context generation (multi-turn dialogue, long-form writing)

**Short answer:** For long-context tasks, token-level UNDO may become less practical; segment-level or block-level extension is natural future work, and the paper already discusses this limitation.

The paper acknowledges in the Conclusion/Limitation section that computational overhead scales with sequence length. For long-horizon tasks, multiple consecutive UNDO tokens are needed to retract a paragraph-length error, and the raw trajectory remains in-context (including error tokens), which does increase context window pressure. Two natural extensions we plan to explore are: (1) coarser-granularity edit actions (sentence-level or chunk-level UNDO), and (2) compressing/collapsing the error portion of the context after correction to recover context window. We agree these are important research directions and will frame them explicitly as future work in the camera-ready.

The current paper focuses on complex reasoning tasks (math/code) where errors are typically localized to a short reasoning step (e.g., a single numerical token or arithmetic operation), where token-level UNDO is sufficient and efficient, as demonstrated in the appendix examples (Tables A1, A2).

---

## Summary

| Concern | Response |
|---|---|
| Compute-matched comparison missing | κ bounds UNDO overhead to 20%/40%; full efficiency table added in camera-ready |
| Latency/throughput not reported | Stack ops are O(1); concrete per-sample stats added in camera-ready |
| Long-context challenges | Acknowledged limitation; segment-level extension is natural future work |

We appreciate the reviewer's positive assessment of the method's simplicity, the mSFT pitfall identification, and the breadth of experimental validation.
