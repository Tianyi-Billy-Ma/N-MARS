# Rebuttal — Reviewer cVJs

We thank the reviewer for the positive assessment and the thoughtful questions. We address each concern below.

---

## W1 (Major): ⟨UNDO⟩ erases only 1 token — seems inefficient for long errors

**Short answer:** Multi-token erasure is naturally achieved by issuing multiple consecutive ⟨UNDO⟩ tokens. The reward structure and κ budget make this efficient rather than degenerate.

Each ⟨UNDO⟩ pops exactly one token from the stack. To erase a k-token error span, the model emits k consecutive ⟨UNDO⟩ tokens. This is precisely the mechanism shown in the training construction (Equation 2 in the paper): after an error span of length k, the augmented sequence contains k ⟨UNDO⟩ tokens followed by the correction. The appendix example (Table A2) illustrates this concretely: "He spends 4 hours driving at 30mph [⟨UNDO⟩⟨UNDO⟩⟨UNDO⟩], but 2 hours are in standstill traffic" — three consecutive ⟨UNDO⟩ tokens retract a three-token error.

Regarding the concern that it's "very hard to train models to do that": the sequence augmentation (Section 2.1) specifically constructs training examples where the model must learn to emit exactly k ⟨UNDO⟩ tokens after an error span of length k. The mSFT objective then trains the model on these traces while masking the error tokens (so the model is not incentivized to generate errors). The ablation in Table 1 validates this — mSFT on augmented traces successfully teaches multi-token erasure, as evidenced by empirical gains on GSM8K (8.8→19.3 for 1B model) and MATH (3.2→15.2 after full N-MARS pipeline).

The κ budget (R_pen = +1.0 for successful UNDO within κ, +0.5 for excessive but correct, −1.0 for unsuccessful) also encourages the model to detect and correct errors as early as possible, since catching a k-token error at an earlier position is more computationally favorable than erasing a larger trailing segment.

---

## W2 (Major): No efficiency analysis — computational overhead not quantified

**Short answer:** The UNDO budget κ formally bounds overhead to 20%/40% extra tokens for GSM8K/MATH. A concrete efficiency comparison will be added to the camera-ready.

The paper's Conclusion/Limitation section notes: "computational overhead: external verification methods > non-monotonic AR > basic AR." N-MARS's overhead is controlled by κ: at κ=0.2 (GSM8K), at most 20% more tokens are generated; at κ=0.4 (MATH), at most 40%. ReVISE, by comparison, requires a second full generation pass (≥100% overhead for each sample). The stack-based post-processing has O(1) cost per token (one push or pop per token), adding negligible latency.

In the camera-ready, we will add a dedicated efficiency analysis reporting: (a) mean ⟨UNDO⟩ tokens per sequence, (b) wall-clock time per sample vs. baselines, (c) total tokens generated per solved problem for N-MARS vs. ReVISE. This will make the claimed efficiency advantage concrete and verifiable.

---

## W3 (Major): Error identification via unmatched tokens may not correspond to actual errors

**Short answer:** The unmatched tokens are positions where the model diverged from the reference; while not always semantically "wrong", they are precisely the positions where the model's reasoning differed from ground truth. Empirical ablation confirms this criterion works effectively.

The reviewer is correct that, strictly speaking, a model-generated token that differs from the reference may not always be "wrong" (e.g., valid paraphrases, equivalent calculation paths). The sequence alignment augmentation identifies *divergence points* from the reference rather than asserting semantic incorrectness at each position.

In practice, for mathematical reasoning tasks, the reference provides a canonical step-by-step solution, and divergences at arithmetic operations (e.g., generating "5" instead of "4" in an intermediate calculation) are indeed errors. The Myers diff alignment preserves matching blocks — so structurally equivalent prefix and suffix segments are not marked as errors.

Critically, the empirical results confirm that this criterion is effective: (1) mSFT trained on these augmented traces improves over few-shot baselines, while SFT without masking *degrades* performance (Table 1b), suggesting the augmented error positions do correspond to correction opportunities; (2) ablation over augmentation strategies (Table 1d) shows that "real error" augmentation (from model deviations) outperforms both random token injection and hard-sample augmentation, suggesting the alignment-based criterion captures meaningful error patterns.

For domains where multiple valid solution paths exist (e.g., open-ended writing), the error criterion could be relaxed using similarity-based matching instead of exact token matching. We will discuss this extension in the camera-ready.

---

## W4 (Major): Hard to design errors for different tasks/domains

**Short answer:** The augmentation is domain-agnostic: it requires only a model-generated completion and a reference completion. Error identification is via sequence alignment, which applies to any token sequence.

The sequence alignment augmentation (Section 2.1) operates on raw token sequences and requires only (prompt, reference) pairs from the training dataset. It does not rely on task-specific error templates or domain knowledge. For images (autoregressive visual generation), the token sequence would be visual tokens; for code, the token sequence would be program tokens. The same Myers diff alignment would identify divergence points.

We note that our experiments already span two distinct domains: mathematical reasoning (GSM8K, MATH) and code generation (MBPP). The consistent improvements across both domains (e.g., MBPP: 33.3→34.1 on Llama-3.2-1B, 60.3→61.1 on Llama-3.1-8B) demonstrate that the method transfers across reasoning types. Extension to image generation would be an interesting future direction, though the definition of a "reference" image token sequence raises its own challenges that we acknowledge as future work.

---

## W5 (Minor): More examples and observations on ⟨UNDO⟩ behavior

**Short answer:** The appendix (Tables A1, A2) already provides two generation examples. We will expand this analysis with quantitative statistics in the camera-ready.

Appendix Tables A1 and A2 show (1) a case where N-MARS solves correctly without any UNDO, and (2) a case where the model emits ⟨UNDO⟩ tokens to correct mid-generation arithmetic errors. In the camera-ready, we will add:
- Distribution of ⟨UNDO⟩ counts per sequence (how many UNDO tokens are used on average)
- Error type analysis: what fraction of UNDO tokens correct arithmetic errors, intermediate calculation errors, or structural errors
- Accuracy conditional on UNDO count: do sequences with more UNDO tokens have higher or lower final accuracy?
- Position analysis: are UNDO tokens more frequent early or late in the reasoning chain?

These analyses will provide the richer behavioral understanding the reviewer requests.

---

## Summary

| Concern | Response |
|---|---|
| UNDO erases only 1 token | Multiple consecutive ⟨UNDO⟩ tokens erase multi-token spans; shown in Appendix Table A2; mSFT trains this explicitly |
| No efficiency analysis | κ bounds overhead to 20%/40%; full efficiency table in camera-ready |
| Unmatched tokens ≠ wrong | Divergence points in reasoning are correlated with errors; ablation (Table 1d) confirms alignment-based augmentation outperforms alternatives |
| Hard to design errors cross-domain | Augmentation is alignment-based, domain-agnostic; GSM8K+MATH+MBPP already span two domains |
| More UNDO examples | Appendix has 2 examples; quantitative UNDO statistics added in camera-ready |

We appreciate the reviewer's recognition of the novelty of the ⟨UNDO⟩ mechanism and the clear motivation of the approach.
