# Rebuttal — Reviewer e2Aq

We thank the reviewer for the detailed and constructive review. We address each question directly.

---

## Q1 / W1: Benchmark evaluation details — MATH subset, MBPP split, Qwen3-4B baseline re-implementation

**Short answer:** We use the standard HuggingFace test splits via lm-evaluation-harness. We will clarify all split details and re-implementation procedures in the camera-ready.

- **MATH:** We use the MATH test split via lm-evaluation-harness with the HuggingFace dataset (hendrycks/competition_math), which corresponds to the full MATH test set. We will explicitly state this in the camera-ready to avoid ambiguity with the MATH-500 subset used in some prior works.

- **MBPP:** We use the standard MBPP test split (500 problems) from HuggingFace, following the same split as ReVISE [3]. We will specify this explicitly in the camera-ready.

- **Qwen3-4B baselines:** Qwen3-4B postdates the original ReVISE paper, so all baseline methods (SFT, RFT, STaR+, ReVISE) on Qwen3-4B were re-implemented by the authors using the original papers' methodologies and hyperparameters. For ReVISE specifically, we followed the two-stage training procedure described in [3] (SFT on correct reasoning traces followed by RL-based revision training). We will add a disclosure statement in the camera-ready clarifying which baselines were re-run and the implementation details used.

---

## Q2 / W2: Missing comparison with Self-Backtracking [Yang et al., 2025, arXiv:2502.04404]

**Short answer:** We missed this concurrent work and thank the reviewer for pointing it out. We will add a comparison and discussion in the camera-ready. The key technical differences are substantial.

We thank the reviewer for identifying this important related work. Self-Backtracking [Yang et al., arXiv:2502.04404] shares the high-level motivation of enabling LLMs to backtrack during reasoning. However, there are several key technical distinctions:

1. **Granularity:** Self-Backtracking operates at the **step level** — the ⟨backtrack⟩ token signals restarting from a previous reasoning step. N-MARS operates at the **token level** — each ⟨UNDO⟩ erases exactly one preceding token, allowing fine-grained correction of individual tokens within a step without discarding entire steps.

2. **Training methodology:** Self-Backtracking uses SFT with expert iteration. N-MARS uses mSFT (which provably avoids negative learning on error tokens; Proposition 2.1) followed by GRPO with a multi-objective reward that regulates UNDO frequency. The mSFT component is absent from Self-Backtracking and is a key technical contribution.

3. **Evaluation scope:** Self-Backtracking evaluates on the Countdown game. N-MARS evaluates on GSM8K, MATH, and MBPP across three backbone models, providing broader coverage.

4. **Reward design:** N-MARS's reward explicitly penalizes unsuccessful UNDO usage (R_pen = −1.0 when output is incorrect despite UNDO), which teaches the model *when* to backtrack, not just *how*.

We will add a dedicated paragraph comparing N-MARS to Self-Backtracking in the related work section, and cite [Yang et al., 2025] appropriately.

---

## Q3 / W3: Proposition 2.1 — gradient non-alignment assumption (B.2) unverified empirically

**Short answer:** The assumption is theoretically motivated and structurally similar to gradient surgery settings, but we agree empirical validation would strengthen the paper. We will add gradient cosine similarity measurements.

The reviewer correctly identifies that Assumption B.2 (gradient non-alignment) is borrowed from PCGrad/gradient surgery [Yu et al., NeurIPS 2020], which analyzes gradient conflict across *distinct task objectives* in multi-task learning. We apply an analogous assumption to *token-position gradients within a single language modeling objective*, where the structural motivation is: error tokens push the parameter update in a direction that increases the likelihood of generating errors (negative learning), which conflicts with the gradient direction that increases the likelihood of ⟨UNDO⟩ and correction tokens.

The reviewer's observation that α=0 (gradient orthogonality) would make the guarantee fail under Descent Lemma alone is correct — in that degenerate case, the mSFT advantage vanishes to zero (Γ_⟨UNDO⟩ → 0). The proposition statement in the paper says "for sufficiently small η," which requires α>0. We will add an explicit clarification that the guarantee holds strictly only when α>0 (gradient conflict) and state the α=0 boundary condition explicitly in the camera-ready.

We will also add empirical validation: cosine similarities between g_error and g_{⟨UNDO⟩}/g_correction during mSFT training, measured on held-out batches at regular training checkpoints. Based on the empirical evidence that mSFT significantly outperforms SFT (Table 1b: 19.3 vs 7.8 on GSM8K-1B), we expect to confirm that these gradients are indeed non-positively aligned in practice.

We also agree with the reviewer that Part (c) (reduced negative learning) is essentially trivial by construction: mSFT zeroes the error gradient, so the proof is direct. We will reframe Part (c) more concisely in the camera-ready to acknowledge this.

---

## Q4 / W4: Missing self-consistency / majority voting comparison for test-time scaling (Fig. 3)

**Short answer:** Figure 3 already includes majority voting (maj@k) for all methods including N-MARS, but does not control for equal FLOPs. We will add a compute-matched self-consistency comparison in the camera-ready.

Figure 3 plots maj@k for all methods as k increases from 2 to 64, so self-consistency (majority voting) is already applied to all methods. The figure shows N-MARS dominates all baselines across all values of k. However, the reviewer's point is that each N-MARS sample is longer than a baseline sample (due to UNDO tokens), so comparing at equal k is not compute-matched.

We will add an analysis where we equalize total generated tokens: e.g., if N-MARS with κ=0.2 uses 1.2× tokens per sample, we compare N-MARS maj@k with baseline maj@⌈1.2k⌉ to match FLOPs. Based on Figure 3 (N-MARS starts at 31.5% at maj@2 vs ReVISE's 28.3%), we expect N-MARS to maintain advantages even under matched-compute comparisons, because the intra-trajectory exploration diversity is qualitatively different from sampling multiple independent trajectories.

---

## Q5 / W5: Average ⟨UNDO⟩ tokens per sequence, wall-clock latency impact

**Short answer:** The UNDO budget κ formally bounds the maximum overhead. Concrete statistics will be added in the camera-ready.

At κ=0.2 (GSM8K), at most 20% of generated tokens are ⟨UNDO⟩; at κ=0.4 (MATH), at most 40%. Since the GRPO reward penalizes unsuccessful UNDO usage, the *actual* fraction is expected to be well below this bound for correctly-solved instances. The stack-based post-processing has O(1) cost per token (push/pop).

In the camera-ready, we will report: (a) mean ⟨UNDO⟩ tokens per sequence on each benchmark, (b) fraction of UNDO tokens that successfully correct errors (i.e., the post-processed output is correct), (c) wall-clock time per sample at κ∈{0.2,0.4} vs. baselines on A100 GPUs, and (d) total tokens generated per solved instance for N-MARS vs. ReVISE.

---

## Summary

| Concern | Response |
|---|---|
| MATH/MBPP split ambiguity, Qwen3 re-impl | Full MATH test; MBPP standard 500; Qwen baselines re-implemented by authors — will clarify in camera-ready |
| Missing Self-Backtracking comparison | Will add; key differences: token-level vs step-level, mSFT+GRPO vs SFT+EI, different benchmarks |
| Proposition 2.1 assumption unverified | Empirical grad cosine similarity measurements added in camera-ready; α=0 boundary condition clarified |
| Missing matched-compute self-consistency | Fig 3 already shows maj@k; compute-matched analysis added in camera-ready |
| ⟨UNDO⟩ count / latency absent | κ bounds overhead; full efficiency stats in camera-ready |

We appreciate the reviewer's recognition of our novel mechanism, comprehensive training pipeline, thorough ablation study, and broad experimental coverage.
