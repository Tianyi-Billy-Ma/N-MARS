# Summary 
AR models generate sequences in a monotonic manner. That means, once a token is sampled, even if it is incorrect or suboptimal, the model cannot revisit or revise it.

To address this limitation, the authors propose N-MARS, a non-monotonic framework that allows models to generate, evaluate, and revise tokens within a single forward pass, thereby enabling exploration before commitment. The authors insert a learnable < UNDO > token into the vocabulary that retracts the previously generated token, making on-the-fly revision possible within the standard enabling on-the-fly revision within standard autoregressive decoding.

For training, the authors introduce a sequence augmentation strategy, mSFT, and GRPO-based method. Experiments show the benefits of N-MARS with different LLMs on different benchmarks.

# Strengths and Weaknesses

Strengths
N-MARS enables AR models to revisit and refine the tokens previously generated, which is very interesting.
The proposed method and training strategies are well-motivated and easy to understand.
Experiments show significant advantages of N-MARs and provide more insights.
Weaknesses
Major

The < UNDO > token erases only 1 token each time. This might not be efficient in some cases. For example, if the model finds some error in the last paragraph in its response, it needs to generate a number of < UNDO > tokens. Even though it is feasible in computation, it seems very hard to train models to do that.
Lack of efficiency analysis. Most of experiments focus on accuracy on benchmarks, and I didn’t find efficiency reports in the paper. I notice that the authors mentioned in the limitation section “the computational overhead: external verification methods > non-monotonic AR > basic AR”. But I think it would be better to include an efficiency comparison.
I’m not sure if it is easy to construct “errors” for models to refine. In the paper, the authors propose a way, “identify error indices as positions in \hat{y} not covered by any matching block”. This doesn’t sound very convincing to me, because tokens that are not matched may not be wrong.
Considering that AR has been used in various domains, such as code and images, I feel it’s probably hard to design “errors” in different tasks.

Minor

It would be better if the authors could provide more examples or observations on LLM behaviors with N-MARS. I think the < UNDO > token is very interesting, so I’m wondering how LLMs will utilize this token, like how many tokens are used in average, how much portions of < UNDO > really delete errors, and what kind of errors are erased.


