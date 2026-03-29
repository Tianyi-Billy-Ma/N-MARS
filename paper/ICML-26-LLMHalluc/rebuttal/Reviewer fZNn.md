# Summary 

This paper proposes N-MARS, a non-monotonic autoregressive sequence modeling framework that enables a model to revise earlier tokens during a single decoding pass via a learned erase action token ⟨UNDO⟩. The training pipeline consists of (i) sequence alignment augmentation that converts model deviations into error-correction traces by inserting error spans followed by the same number of ⟨UNDO⟩ tokens and then the ground-truth continuation, (ii) masked supervised fine-tuning (mSFT) that masks loss on the injected error tokens to avoid “negative learning” while still conditioning on them as context, and (iii) GRPO-based reinforcement learning with a reward that combines final correctness and a penalty/bonus term that regulates ⟨UNDO⟩ usage. Experiments on GSM8K, MATH, and MBPP across several backbones (e.g., Llama-3.x and Qwen) show consistent gains over SFT/RFT/STaR+ and a multi-pass revision baseline (ReVISE), with ablations analyzing token initialization, augmentation strategies, reward components, and κ sensitivity.


# Strengths And Weaknesses:
Pros:

Implementing backtracking via an explicit ⟨UNDO⟩ token and stack-style post-processing is simple, model-agnostic, and easy to integrate into standard decoding pipelines.
The paper clearly identifies the “train on error tokens → imitate errors” pitfall and proposes mSFT to address it, supported by a theoretical argument and empirical evidence that SFT on augmented traces can degrade performance while mSFT improves it.
Results span multiple backbones and multiple reasoning domains (math + code), with thoughtful design-space exploration (token initialization, reward variants, augmentation variants, κ sweeps, test-time scaling).
Cons:

Because N-MARS can generate extra tokens (error spans + ⟨UNDO⟩ + regenerated content), improvements may partially reflect more test-time compute, but the paper does not present compute-matched comparisons (e.g., fixed total token budget across methods) or wall-clock/throughput reporting.
The approach operates at token-level revision and the raw trajectory remains in-context, which may exacerbate cost and context-window pressure for long-horizon tasks (dialogue, summarization, tool traces). This is acknowledged only qualitatively, without dedicated experiments.
The RL reward uses a correctness signal (rule-based verifier in the current setup), which is natural for GSM8K/MATH/MBPP but may not transfer cleanly to open-ended generation without strong verifiers or preference models; the paper could clarify how robust the method is under weaker feedback.

# Questions 

Did you compare N-MARS to standard SFT (or ReVISE-style baselines) under the same inference compute budget (e.g., same maximum generated tokens / same FLOPs / same wall-clock), including allowing the baseline to emit longer reasoning traces or “pause/deliberation” tokens? If not, can you add a controlled study to rule out “extra compute” as the main driver of gains?
What is the measured latency/throughput impact of ⟨UNDO⟩ at inference (e.g., tokens generated per solved instance, wall-clock per sample) across datasets and κ settings? How often does the model backtrack, and what is the distribution of ⟨UNDO⟩ counts on test data?
Since the raw trajectory (including error tokens and ⟨UNDO⟩) remains part of the context, what are the expected challenges when applying N-MARS to long-context generation (multi-turn dialogue, long-form writing)? Do you anticipate needing higher-level edit actions (sentence/segment-level undo) or memory/position handling changes to avoid context bloat?
