We thank the reviewer for the careful and detailed reading. We address each concern directly. 




**Q2 / W2:** In what precise sense is N-MARS "non-monotonic"?

**Response to Q2 / W2:**

**Q3 / W3:** Are the comparisons in Table 2 fair? N-MARS trained on richer augmented traces.

**Response to Q3 / W3:**
In our experiment (reported in Table 2), all baseline methods and N-MARS are fine-tuned on the exactly the same **training dataset to ensure a fair comparison**. 
The purpose of the Table 2 in our paper is aim to compare the overall performance of the full framework with baseline methods.

The augmented traces used for N-MARS are generated from the model's own generations (Section 2.1).
We want to clarify that **this is the training recipe, not an external data advantage**, which is analogous to RFT, STaR+, and ReVISE, i.e., these baseline methods also leverage self-generated augmented data for training. 
Moreover, we already conduct experiments in a data-matched manner and report the results in Tables 1b (SFT) and 1d (GRPO).
Specifically, we 
For your convinces, we have list the performance below:
Table: Performance comparison over GSM8K for difference training methods.
| **Method** | **Accuracy** |
| :--- | :--- |
| Base | 8.8 |
| Base + GRPO | 25.0 |
| SFT | 7.8 |
| SFT + GRPO | 23.9 |
| mSFT | 19.3 |
| mSFT + GRPO (N-MARS) | 31.1 |



**Q5:** Does the method generalize beyond verifiable reasoning tasks?