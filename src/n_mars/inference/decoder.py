"""Non-monotonic autoregressive decoder with stack-based <UNDO> post-processing.

The model generates a raw trajectory that may contain <UNDO> tokens.
A stack-based post-processor removes undone tokens to produce the final
semantic sequence.

Example: [The, answer, is, 5, <UNDO>, 4] -> [The, answer, is, 4]
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Optional

import torch


def stack_postprocess(tokens: list[int], undo_token_id: int) -> list[int]:
    """Stack-based post-processing: push normal tokens, pop on <UNDO>.

    Args:
        tokens: Raw token ID sequence from the model.
        undo_token_id: Token ID of the <UNDO> special token.

    Returns:
        Final semantic token sequence with undone tokens removed.
    """
    stack: list[int] = []
    for tok in tokens:
        if tok == undo_token_id:
            if stack:
                stack.pop()
        else:
            stack.append(tok)
    return stack


class NMARSDecoder:
    """Standard N-MARS decoder with stack-based <UNDO> post-processing.

    The model generates normally via autoregressive decoding. The raw output
    may contain <UNDO> tokens. After generation, stack_postprocess is applied
    to recover the final semantic sequence.
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Resolve <UNDO> token ID from tokenizer
        undo_token = "<UNDO>"
        if undo_token in tokenizer.get_vocab():
            self.undo_token_id: Optional[int] = tokenizer.convert_tokens_to_ids(undo_token)
        else:
            ids = tokenizer.encode(undo_token, add_special_tokens=False)
            self.undo_token_id = ids[0] if len(ids) == 1 else None

        self.eos_token_id: int = tokenizer.eos_token_id

    def generate(
        self,
        prompt: str,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> dict:
        """Generate and post-process a single response.

        Args:
            prompt: Input prompt string.
            do_sample: Whether to use sampling (True) or greedy (False).
            num_return_sequences: Number of sequences to generate; if > 1,
                returns the first post-processed result.
            **kwargs: Additional kwargs forwarded to model.generate().

        Returns:
            Dict with keys:
              - text: Decoded text after stack post-processing.
              - raw_text: Decoded text before post-processing.
              - num_undos: Number of <UNDO> tokens in the raw sequence.
              - tokens_generated: Length of post-processed generated sequence.
              - raw_tokens_generated: Length of raw generated sequence.
        """
        device = next(self.model.parameters()).device
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                pad_token_id=self.tokenizer.pad_token_id or self.eos_token_id,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else 1.0,
                num_return_sequences=num_return_sequences,
                **kwargs,
            )

        # Take the first sequence
        raw_ids = outputs[0][prompt_len:].tolist()

        raw_text = self.tokenizer.decode(raw_ids, skip_special_tokens=False)

        # Count <UNDO> tokens before post-processing
        num_undos = raw_ids.count(self.undo_token_id) if self.undo_token_id is not None else 0

        # Stack post-processing
        if self.undo_token_id is not None:
            processed_ids = stack_postprocess(raw_ids, self.undo_token_id)
        else:
            processed_ids = raw_ids

        text = self.tokenizer.decode(processed_ids, skip_special_tokens=True)

        return {
            "text": text,
            "raw_text": raw_text,
            "num_undos": num_undos,
            "tokens_generated": len(processed_ids),
            "raw_tokens_generated": len(raw_ids),
        }


class NMARSMajorityVotingDecoder:
    """N-MARS decoder with majority voting for test-time scaling (maj@k).

    Generates k samples, post-processes each with stack_postprocess,
    extracts answers, and returns the majority vote answer.
    """

    def __init__(
        self,
        model,
        tokenizer,
        k: int = 64,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> None:
        self.k = k
        self._decoder = NMARSDecoder(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    def generate(
        self,
        prompt: str,
        answer_extractor: Optional[Callable[[str], Optional[str]]] = None,
    ) -> dict:
        """Generate k samples and return the majority vote answer.

        Args:
            prompt: Input prompt string.
            answer_extractor: Optional callable that extracts a canonical
                answer string from decoded text. If None, full post-processed
                text is used as the "answer" for voting.

        Returns:
            Dict with keys:
              - answer: Majority vote answer (None if no answers extracted).
              - vote_count: Number of votes for the winning answer.
              - total_samples: k (number of samples generated).
              - samples: List of per-sample dicts from NMARSDecoder.generate().
              - answer_counts: Counter mapping answer -> vote count.
        """
        samples = []
        answers = []

        for _ in range(self.k):
            result = self._decoder.generate(prompt, do_sample=True)
            samples.append(result)

            if answer_extractor is not None:
                ans = answer_extractor(result["text"])
            else:
                ans = result["text"].strip()

            answers.append(ans)

        # Filter None answers for voting
        valid_answers = [a for a in answers if a is not None]
        answer_counts: Counter = Counter(valid_answers)

        if answer_counts:
            best_answer, best_count = answer_counts.most_common(1)[0]
        else:
            best_answer, best_count = None, 0

        return {
            "answer": best_answer,
            "vote_count": best_count,
            "total_samples": self.k,
            "samples": samples,
            "answer_counts": dict(answer_counts),
        }
