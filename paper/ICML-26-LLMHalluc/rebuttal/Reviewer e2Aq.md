# Summary 

This paper introduces the Non-Monotonic Autoregressive Sequence Model (N-MARS), which augments the vocabulary of autoregressive LLMs with a learnable ⟨UNDO⟩ token, enabling models to erase erroneous tokens and re-generate corrections within a single left-to-right forward pass. The method consists of a three-stage pipeline: (1) sequence augmentation via Myers diff alignment to construct error→⟨UNDO⟩→correction training data, (2) masked supervised fine-tuning (mSFT) that excludes error tokens from the loss to prevent negative learning, and (3) Group Relative Policy Optimization (GRPO) with a multi-objective reward combining correctness and UNDO-usage regulation. A theoretical analysis (Proposition 2.1) provides one-step guarantees that mSFT improves over standard SFT in error detection, corrective generation, and reduced negative learning under a gradient non-alignment assumption. Experiments on GSM8K, MATH, and MBPP with Llama-3.2-1B, Llama-3.1-8B, and Qwen3-4B backbones show consistent improvements over SFT, RFT, STaR+, and ReVISE baselines. Ablations validate each component of the pipeline.

# Strengths and Weaknesses 

Strengths
S1. Novel and intuitive core mechanism. The ⟨UNDO⟩ token is an elegant and conceptually simple idea that breaks the monotonic constraint of autoregressive generation. Unlike multi-turn self-correction methods (e.g., Self-Refine) or external verifier approaches, N-MARS enables intra-sequence error correction within the standard left-to-right decoding loop, requiring no architectural changes beyond a single vocabulary addition. The analogy to backspace editing in human writing is natural and well-motivated.

S2. Comprehensive and well-structured training pipeline. The three-stage design (augmentation → mSFT → GRPO) is systematic, with each stage addressing a specific challenge: data construction, avoiding negative learning, and reinforcement-based optimization. The masked SFT idea — masking error tokens to prevent the model from learning to reproduce mistakes — is a clean and well-justified design choice, supported both by theory (Proposition 2.1) and compelling empirical evidence (Table 1b showing SFT degrades performance from 8.8→7.8 while mSFT improves to 19.3 on GSM8K-1B).

S3. Thorough ablation study. Table 1 systematically validates all design dimensions: token initialization strategy (semantic > random > zero), mSFT vs SFT, reward components (R_inc and R_pen), GRPO contribution, augmentation methods (hard sample > stochastic), and positional encoding (hurts performance). The ablations convincingly demonstrate that each component contributes and the choices are non-trivial.

S4. Broad experimental coverage. The paper evaluates across three benchmarks (GSM8K, MATH, MBPP), three backbone models of varying sizes (1B, 4B, 8B), and includes domain transfer experiments (Table 3), instruction-tuned model results (Table 4), hyperparameter analysis (Fig. 2), and test-time scaling behavior (Fig. 3). This breadth significantly strengthens the empirical claims.

S5. Theoretical motivation for mSFT. Proposition 2.1 provides a formal argument for why masking error tokens is beneficial, decomposing the advantage into three interpretable components. While the theory has limitations (see W3), it adds analytical depth to what would otherwise be a purely empirical contribution.

Weaknesses
W1. Benchmark evaluation details underspecified. The paper does not clarify: (a) whether MATH refers to full MATH (12,500 problems) or the commonly used MATH-500 subset — these yield non-comparable numbers; (b) which MBPP split is used (standard 500, sanitized 427, or MBPP+ 399). Additionally, the Qwen3-4B baselines (SFT, RFT, STaR+, ReVISE) appear to have been re-run by the authors since Qwen3-4B postdates the original ReVISE paper [3] — the re-implementation details should be disclosed to ensure fair comparison.

W2. Critical missing comparison: Self-Backtracking. A concurrent work [1] introduces a ⟨backtrack⟩ special token with a nearly identical mechanism: training LLMs to recognize suboptimal reasoning paths and autonomously backtrack during both training and inference. This paper is not cited or discussed. Given the close similarity (special vocabulary token for intra-sequence backtracking in reasoning tasks, masking error tokens during training, test-time scaling via backtracking), its omission weakens the novelty claim and will likely be flagged by any reviewer familiar with the 2025 backtracking literature. The related work section should explicitly compare N-MARS with Self-Backtracking, discussing differences in training methodology (mSFT+GRPO vs SFT+expert iteration), evaluation scope (GSM8K/MATH/MBPP vs Countdown), and the backtracking granularity (token-level vs step-level).

W3. Proposition 2.1's assumptions are unverified and the result is weaker than presented. The central Assumption B.2 (gradient non-alignment) requires strict negative correlation between error-token gradients and ⟨UNDO⟩/correction gradients (α > 0). This is borrowed from PCGrad / gradient surgery for multi-task learning [2], but that work studies gradient conflict across distinct task objectives, whereas N-MARS applies the same assumption to token-position gradients within a single language modeling objective — the analogy is plausible but not formally justified. Key concerns: (a) No empirical validation of gradient cosine similarities is provided to confirm the assumption holds in this setting. (b) The proposition statement says gaps are non-negative "for sufficiently small η," but this is only true when α > 0; when α = 0 (orthogonality), the Descent Lemma remainder dominates and the guarantee fails — this critical condition is not stated clearly. (c) Part (c) (reduced negative learning) is essentially trivial by construction since mSFT zeroes the error gradient. (d) The result is a one-step local guarantee, not a convergence analysis.

W4. Missing self-consistency baseline undermines test-time scaling claims. The paper claims test-time scaling benefits from ⟨UNDO⟩ tokens (Fig. 3), but does not compare against self-consistency / majority voting [4] at matched inference compute budgets. Since ⟨UNDO⟩ tokens increase sequence length (and thus inference cost), the performance gains may be partially or fully explained by additional compute. A fair comparison requires equating inference FLOPs or wall-clock time between N-MARS (with UNDO) and self-consistency (multiple independent samples + voting).

W5. Inference cost analysis absent. ⟨UNDO⟩ tokens inherently increase sequence length during generation, adding compute overhead. The paper provides no analysis of latency, throughput, or the average number of UNDO tokens generated. Understanding the cost-accuracy tradeoff is important for practical deployment.

References

[1] Yang, X.-W., Zhu, X.-Y., Wei, W.-D., Zhang, D.-C., Shao, J.-J., Zhou, Z., Guo, L.-Z., and Li, Y.-F. "Step Back to Leap Forward: Self-Backtracking for Boosting Reasoning of Language Models." arXiv preprint arXiv:2502.04404, 2025.

[2] Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., and Finn, C. "Gradient Surgery for Multi-Task Learning." In NeurIPS, 2020.

[3] Lee, H., Oh, S., Kim, J., Shin, J., and Tack, J. "ReVISE: Learning to Refine at Test-Time via Intrinsic Self-Verification." In ICML, 2025.

[4] Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., and Zhou, D. "Self-Consistency Improves Chain of Thought Reasoning in Language Models." In ICLR, 2023.


# Questions 

Q1. Could you specify whether MATH refers to the full MATH dataset or the MATH-500 subset, and which MBPP split (standard 500, sanitized 427, or MBPP+ 399) is used? For the Qwen3-4B baselines, were all methods (SFT, RFT, STaR+, ReVISE) re-implemented by the authors?

Q2. How does N-MARS compare to Self-Backtracking [1], which introduces a ⟨backtrack⟩ token for LLM reasoning in a very similar manner? What are the key technical differences?

Q3. Have you measured gradient cosine similarities between error-token gradients and ⟨UNDO⟩/correction gradients during training? This would provide empirical support for the gradient non-alignment assumption (B.2) underlying Proposition 2.1, which is borrowed from [2] but applied in a different setting.

Q4. Could you provide a comparison with self-consistency [4] (majority voting) at matched inference compute? Given that ⟨UNDO⟩ tokens increase sequence length, it is important to disentangle the effect of the correction mechanism from the effect of additional compute.

Q5. What is the average number of ⟨UNDO⟩ tokens generated per sequence at inference time? How does this affect wall-clock latency relative to standard autoregressive generation?

References

[1] Yang, X.-W., Zhu, X.-Y., Wei, W.-D., Zhang, D.-C., Shao, J.-J., Zhou, Z., Guo, L.-Z., and Li, Y.-F. "Step Back to Leap Forward: Self-Backtracking for Boosting Reasoning of Language Models." arXiv preprint arXiv:2502.04404, 2025.

[2] Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., and Finn, C. "Gradient Surgery for Multi-Task Learning." In NeurIPS, 2020.

[3] Lee, H., Oh, S., Kim, J., Shin, J., and Tack, J. "ReVISE: Learning to Refine at Test-Time via Intrinsic Self-Verification." In ICML, 2025.

[4] Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., and Zhou, D. "Self-Consistency Improves Chain of Thought Reasoning in Language Models." In ICLR, 2023.
