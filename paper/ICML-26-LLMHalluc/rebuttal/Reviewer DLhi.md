# Summary 

This paper proposes N-MARS, which introduces an <UNDO> token so that a language model can retract previously generated tokens
within a single decoding trajectory. The method combines alignment-based augmentation, mSFT, and GRPO-based RL, and report improvements over several baselines on GSM8K, MATH, and MBPP.

The paper is well motivated and clearly written, and mSFT is a useful component. However, I have concerns about the novelty of
the explicit <UNDO> mechanism, the fact that the underlying prediction process remains standard left-to-right autoregression, and the fairness of the main comparisons given that N-MARS is trained on richer augmented error-correction traces than simpler baselines such as standard SFT.


# Strengths and Weaknesses

## Strengths
1. Important problem.
The paper trys to address error propagation in autoregressive language models, which is a meaningful limitation for reasoning tasks.
2. Clean training recipe.
The overall pipeline is coherent, and mSFT is a useful component for avoiding negative learning on augmented traces.
3. Generally consistent gains.
The method improves over the reported baselines across multiple models and tasks.
## Weaknesses
1. The paper does not establish that explicit <UNDO> tokens are preferable to natural-language self-correction.
A major missing baseline is a data-matched verbal-reflection variant, where the model is trained on the same augmented traces but uses natural-language revision patterns (e.g., “that was incorrect, let me revise”) instead of a newly introduced control token. Without such a comparison, it is hard to know whether the explicit <UNDO> mechanism is truly necessary or simply one possible implementation.
2. The claimed non-monotonicity is limited to the visible output sequence, not the underlying prediction process.
The model still performs standard left-to-right next-token prediction over an edit-action trajectory. In that sense, the method is still autoregressive at the level of actual computation; only the reduced final sequence appears non-monotonic after applying the stack-like post-processing rule. This makes the conceptual claim somewhat weaker than it initially appears.
3. The main comparisons may not be fully fair.
N-MARS is trained on augmented error-correction traces, which provide richer supervision than the training data used by standard baselines such as SFT, and arguably a different kind of synthetic signal than RFT/STaR+-style methods. As a result, Table 2 does not cleanly isolate the benefit of the <UNDO> mechanism itself; it instead compares full recipes with different supervision richness.
4. Efficiency claims are under-supported.
One motivation of the paper is that integrating correction into a single trajectory could be more efficient than post-hoc verification or multi-pass refinement. However, the paper does not provide direct evidence such as latency, total generated tokens, <UNDO> frequency, or matched-compute comparisons against methods like ReVISE.


# Questions 

1. Why is an explicit [object Object] token preferable to natural-language self-correction? Please compare against a strong baseline that uses the same augmented training traces but replaces [object Object] with natural-language revision markers.
2. In what precise sense is the method non-monotonic? Since the model still performs left-to-right autoregressive prediction over the edit trajectory, would it be more accurate to describe the method as autoregressive generation over editable trajectories rather than truly non-autoregressive or fundamentally non-monotonic generation?
3. How fair are the comparisons in Table 2? Since N-MARS is trained on richer augmented traces, can the authors provide data-matched baselines where SFT/ReVISE-style methods are trained with the same amount and type of synthetic supervision?
4. What is the actual efficiency cost of the method? Please report average generated tokens, average number of [object Object] tokens, average correction derepth, and wall-clock latency, ideally compared with multi-pass baselines under matched compute budgets.
5. Does the method generalize beyond tasks with strong verification signals? The current experiments focus mainly on math and code benchmarks where correctness can be measured relatively cleanly. How would the method extend to more open-ended generation tasks?


# Limitations 

The paper has several limitations that currently restrict its impact.

First, the core mechanism depends on a newly introduced control token whose semantics are not part of standard pretraining. This
raises the question of whether the method is leveraging the model’s natural language ability in the most effective way, or instead
imposing an artificial editing protocol at post-training time.

Second, the method is only “non-monotonic” at the level of the post-processed visible output. The underlying prediction process
remains fully autoregressive and linear. As a result, the approach may not address the deeper limitation of autoregressive reasoning as strongly as the paper suggests.

Third, the current empirical evidence does not cleanly disentangle method improvements from data/supervision improvements, since the model is trained on richer augmented traces than the simpler baselines. This makes it difficult to determine how much of the gain comes from the [object Object] mechanism itself.

Fourth, the method is evaluated mostly on verifiable reasoning tasks such as GSM8K, MATH, and MBPP. It remains unclear whether the same approach would be effective in settings where correctness is ambiguous and reward design is less straightforward.
