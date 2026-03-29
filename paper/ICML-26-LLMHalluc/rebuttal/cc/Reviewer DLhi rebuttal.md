# Rebuttal — Reviewer DLhi

We thank the reviewer for the careful and detailed reading. We address each concern directly.

---

## Q1 / W1: Why is ⟨UNDO⟩ preferable to natural-language self-correction? Missing verbal-reflection baseline.

**Short answer:** The ⟨UNDO⟩ token is more token-efficient and precise than natural-language reflection, and the existing augmentation ablation partially speaks to this. We will add a direct verbal-reflection baseline in the camera-ready.

The core advantage of ⟨UNDO⟩ over a verbal-reflection pattern such as "that was incorrect, let me revise" is **compactness and precision**: each ⟨UNDO⟩ deterministically deletes exactly one preceding token via a stack-pop, producing a clean corrected output without additional tokens for the revision preamble. A natural-language reflection variant would (a) expand sequence length with prose, (b) require the model to learn *where* the error is from the prose rather than from the token's position, and (c) conflate the correction signal with fluency concerns that are irrelevant for reasoning.

That said, we acknowledge this is an important empirical question, and the reviewer is right that a data-matched verbal-reflection baseline — same augmented traces but with "let me redo" markers instead of ⟨UNDO⟩ — is currently missing. We commit to adding this comparison in the camera-ready. Our prediction, grounded in the mSFT theoretical analysis, is that the verbal variant will show weaker separation between the error-token gradient and the correction gradient (since the "redo" text and error context are both natural language), making mSFT masking less precise.

---

## Q2 / W2: In what precise sense is N-MARS "non-monotonic"?

**Short answer:** N-MARS is non-monotonic at the level of the *visible output sequence* (the semantic content the model produces), not at the level of the forward computation. This is the claim the paper makes, and we agree the framing could be sharpened.

The paper explicitly states (Section 2.4): "We view the inference process as non-monotonic generation, where the semantic content of the sequence is not strictly append-only." The generation trajectory τ is still a left-to-right sequence of tokens, but the *visible output* y — obtained by the deterministic stack-based operator — is non-monotonically constructed. The trajectory `(The, answer, is, 5, ⟨UNDO⟩, 4)` produces output `(The, answer, is, 4)`, where token `5` never appears in the final sequence despite being part of the computation.

We agree with the reviewer that describing this as "autoregressive generation over editable trajectories" is a more precise framing at the computation level, and we will revise the abstract and introduction in the camera-ready to clarify: N-MARS is non-monotonic in that the *output sequence* is not a strict prefix-extension of prior output, even though the *trajectory* is computed left-to-right. The conceptual contribution is that the model can revise its committed output, not that it uses non-autoregressive attention.

---

## Q3 / W3: Are the comparisons in Table 2 fair? N-MARS trained on richer augmented traces.

**Short answer:** All baselines have access to the same underlying problem set; the augmentation is the **method**, not an unfair data advantage. ReVISE also uses augmented training. Ablations show the ⟨UNDO⟩ mechanism — not just the data augmentation — drives gains.

All methods (SFT, RFT, STaR+, ReVISE, N-MARS) are trained on the same problem dataset. The augmented traces in N-MARS are derived from the model's own generations (Section 2.1, Sequence Alignment Augmentation) — this is the training recipe, not an external data advantage. This is analogous to how RFT uses self-generated filtered solutions, and STaR+ iterates over self-generated correct solutions. ReVISE similarly employs an augmented training regime: it first trains on correct reasoning traces (SFT stage) then applies RL for revision behavior.

To isolate the ⟨UNDO⟩ mechanism's contribution, we point to **Table 1b (SFT ablation)**: when N-MARS's augmented data is trained with *standard SFT* (without mSFT), performance *degrades* below the few-shot baseline (7.8 vs 8.8 on GSM8K-1B), while the mSFT variant improves to 19.3. This shows the augmented data alone is not beneficial — the ⟨UNDO⟩ mechanism (enabled by mSFT and GRPO together) is what drives improvement.

We will add a data-matched baseline in the camera-ready: training SFT and RFT on our augmented dataset (without ⟨UNDO⟩ tokens) to further isolate the mechanism.

---

## Q4 / W4: Efficiency claims under-supported — no latency, tokens, ⟨UNDO⟩ frequency, or matched-compute comparisons.

**Short answer:** We agree, and will add a concrete efficiency table. The UNDO budget κ provides a formal upper bound on overhead.

At κ=0.2 (GSM8K), at most 20% more tokens are generated than the base sequence; at κ=0.4 (MATH), at most 40% more. ReVISE, by contrast, requires a full second generation pass (≥100% overhead). The GRPO reward (R_pen = −1.0 for UNDO on incorrect output) actively penalizes wasteful UNDO usage. A full efficiency analysis — mean UNDO tokens per instance, fraction of UNDO that correct errors vs. false-positives, wall-clock latency — will be included in the camera-ready.

---

## Q5: Does the method generalize beyond verifiable reasoning tasks?

**Short answer:** The current method requires a rule-based correctness signal (verifier). Extension to open-ended generation is an important direction that would require a learned reward model or preference signal; we will frame this explicitly as future work.

The RL stage uses R_inc ∈ {−1,+1} from a rule-based verifier, which is natural for GSM8K/MATH/MBPP. For open-ended generation (e.g., summarization, dialogue), one would need a preference model or LLM-as-judge reward. The mSFT stage itself does not require a verifier — it only requires aligned reference sequences — so the first two training stages generalize to any sequence-to-sequence task where reference completions are available. The RL stage's transferability is the open research question, and we will discuss it explicitly.

---

## Summary

| Concern | Response |
|---|---|
| Missing verbal-reflection baseline | Will add in camera-ready; ⟨UNDO⟩ is more compact/precise by construction |
| "Non-monotonic" claim scope | Non-monotonic at output level, not forward-computation level; will sharpen framing |
| Fairness of comparisons | Augmentation is the method; SFT-on-augmented-data *hurts* (Table 1b), isolating mechanism |
| Efficiency under-supported | κ bounds overhead formally; full efficiency table in camera-ready |
| Open-ended generation | mSFT stage generalizes; RL stage requires verifier — flagged as future work |
