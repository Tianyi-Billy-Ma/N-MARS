

We thank the reviewer for the detailed and constructive review. We address each question directly. 

**Q1 / W1: Benchmark evaluation details**

**Response to Q1 / W1:**
We leverage lm_eval for evaluation across all methods and backbones to ensures a fair comparison.
Moreover, in the paper, the dataset MATH refers to the MATH-500 subset, and the MBPP split used is the standard 500 for evaluation. 
Since Qwen3-4B postdates the original Revise paper, we reimplement the baselines (SFT, RFT, STaR+ and Revise) for Qwen3-4B following the training recipe described in Revise. 
We have briefly described the training details in the paper (Appendix A.4 and A.5).

**SFT**: We explore the learning rate in {1e-4, 1e-5} with a 0.1 warmup ratio, 0.1 weight decay, and the training batch size is set to 16.
**RFT**: Following Revise, we sampled 10 completions for GSM8K, 4 for MetaMath (used for evaluation for MATH-500), and 20 for MBPP. 
**STaR+**: We draw the same number of samples as in RFT. The out loop is fixed to three with one epoch per outer loop. Rationalization was performed with a hint, where the answer was provided except for the rationale, which served as the hint.
**Revise**: For the data sampling phase in Revise, we sample 10 for GSM8K, 4 for MATH-500, and 20 for MBPP.

We appreciate the reviewer to point out this question. 
We will clarify these implementation details in our next revision, and also release our source code to ensure reproducibility. 


**Q2 / W2:** Missing comparison with Self-Backtracking [1] 
**Response to Q2 / W2:**
We appreciate the reviewer for pointing out this concurrent work. 
We did aware of this work, while it handles the backtracking mechanism at the thinking step level, which is not directly comparable to our token-level backtracking mechanism.
Hence, we did not include this work in our original manuscript.










[1] Yang, X.-W., Zhu, X.-Y., Wei, W.-D., Zhang, D.-C., Shao, J.-J., Zhou, Z., Guo, L.-Z., and Li, Y.-F. "Step Back to Leap Forward: Self-Backtracking for Boosting Reasoning of Language Models." arXiv preprint arXiv:2502.04404, 2025.
