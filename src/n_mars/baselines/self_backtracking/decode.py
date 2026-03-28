"""Self-Backtracking decoder implementation.

Faithfully reproduces the decoder from:
  arXiv:2502.04404 — Self-Backtracking Language Models
  GitHub: LAMDASZ-ML/Self-Backtracking (src/decoder/decoders.py)

Key design choices matched to their codebase:
  - Beam search (num_beams=n) + sampling for candidate generation
  - Teacher-forcing re-scoring for accurate log-prob / seq_len scores
  - Backtrack queue limited to sqrt(b) unique states (not sqrt(n))
  - Deduplication + score aggregation of completed candidates
  - Guard against rolling back past the initial prompt
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F


class SelfBackTrackingDecoder:
    """Decoder that uses backtracking search over reasoning steps.

    The decoder runs in rounds. Each round:
    1. Generates n candidates via beam search + sampling from each queued state.
    2. Splits candidates into completed (EOS, no backtrack) and backtracked.
    3. Rolls backtracked states back one reasoning step (to second-to-last \\n).
    4. Deduplicates rolled-back states, enqueues up to sqrt(b) for next round.
    5. After b+1 rounds (or queue exhausted), aggregates completed candidates
       by output text, selects the best by summed log_prob / seq_len.
    """

    def __init__(
        self,
        model,
        tokenizer,
        b: int = 1,
        n: int = 32,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.b = b
        self.n = n
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Resolve backtrack token id — must already be in the tokenizer
        backtrack_token = "<backtrack>"
        if backtrack_token in tokenizer.get_vocab():
            self.backtrack_token_id: Optional[int] = tokenizer.convert_tokens_to_ids(
                backtrack_token
            )
        else:
            ids = tokenizer.encode(backtrack_token, add_special_tokens=False)
            self.backtrack_token_id = ids[0] if len(ids) == 1 else None

        self.eos_token_id: int = tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def backtrack(
        self, cur_input_ids: torch.Tensor, init_input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Roll back to before the last reasoning step.

        Matches their implementation: finds second-to-last newline, truncates
        there, re-encodes. If the result is shorter than or equal to the
        initial prompt, returns the initial prompt (guard against over-rollback).
        """
        ids_1d = cur_input_ids.view(-1)
        cur_text = self.tokenizer.decode(ids_1d.tolist(), skip_special_tokens=True)

        last_index = cur_text.rfind("\n")
        if last_index == -1:
            return init_input_ids.view(-1)

        second_last_index = cur_text.rfind("\n", 0, last_index)
        if second_last_index == -1:
            return init_input_ids.view(-1)

        new_text = cur_text[: second_last_index + 1]
        new_input_ids = self.tokenizer(
            new_text, return_tensors="pt"
        ).input_ids.squeeze(0)

        # Guard: don't roll back past the initial prompt
        if new_input_ids.shape[0] <= init_input_ids.view(-1).shape[0]:
            return init_input_ids.view(-1)

        return new_input_ids

    def _score_sequences(
        self, sequences: torch.Tensor, prompt_len: int
    ) -> torch.Tensor:
        """Re-score sequences via teacher-forcing forward pass.

        This matches their approach: run a full forward pass on the completed
        sequences, gather log-probs of actual tokens, compute
        sum(log_probs) / generated_length for each sequence.

        Returns a 1-D tensor of scores (one per sequence).
        """
        device = next(self.model.parameters()).device
        sequences = sequences.to(device)

        with torch.no_grad():
            logits = self.model(sequences).logits
            log_probs = F.log_softmax(logits, dim=-1)

            # Shift for next-token prediction
            shift_logits = log_probs[:, :-1, :].contiguous()
            shift_labels = sequences[:, 1:].contiguous()

            # Mask: only score generated tokens (exclude prompt)
            mask = torch.ones_like(shift_labels, dtype=torch.bool)
            mask[:, : prompt_len - 1] = False  # -1 due to shift

            # Gather log-probs of actual next tokens
            gathered = torch.gather(
                shift_logits, dim=2, index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            gathered = gathered * mask

            seq_lengths = mask.sum(dim=1).clamp(min=1)
            scores = gathered.sum(dim=1) / seq_lengths

        return scores.cpu()

    def _generate_candidates(
        self, input_ids: torch.Tensor, prompt_len: int
    ) -> tuple[list[dict], list[dict]]:
        """Generate n candidates via beam search + sampling.

        Returns (completed, backtracked) lists of candidate dicts.
        Each dict has keys: input_ids (2-D, 1×seq), text, score.
        """
        device = next(self.model.parameters()).device
        # input_ids: 1-D or 2-D
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device)

        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    pad_token_id=self.tokenizer.pad_token_id or self.eos_token_id,
                    max_new_tokens=self.max_new_tokens,
                    num_return_sequences=self.n,
                    do_sample=True,
                    temperature=self.temperature,
                    num_beams=self.n,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            except Exception:
                return [], []

        sequences = outputs.sequences  # (n, seq_len) on device

        # Re-score via teacher forcing
        scores = self._score_sequences(sequences, prompt_len)

        # Decode and classify
        texts = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)

        completed = []
        backtracked = []
        for i in range(sequences.shape[0]):
            txt = texts[i]
            seq = sequences[i].unsqueeze(0).cpu()
            entry = {"input_ids": seq, "text": txt, "score": scores[i].item()}

            if "<backtrack>" in txt:
                backtracked.append(entry)
            else:
                completed.append(entry)

        return completed, backtracked

    def _aggregate_completed(self, candidates: list[dict]) -> list[dict]:
        """Deduplicate completed candidates by output text, summing scores.

        Matches their agg() method.
        """
        if not candidates:
            return []

        output_dict: dict[str, list] = {}
        for cand in candidates:
            # Extract only the response portion for dedup key
            txt = cand["text"]
            response_marker = "###Response:\n"
            idx = txt.find(response_marker)
            if idx != -1:
                key = txt[idx + len(response_marker) :]
            else:
                key = txt

            if key not in output_dict:
                output_dict[key] = [1, cand["score"], cand]
            else:
                output_dict[key][0] += 1
                output_dict[key][1] += cand["score"]

        aggregated = []
        for _key, (count, sum_score, representative) in output_dict.items():
            aggregated.append(
                {
                    "input_ids": representative["input_ids"],
                    "text": representative["text"],
                    "score": sum_score,
                }
            )
        return aggregated

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> dict:
        """Run full backtracking search for the given prompt string.

        Returns:
            A dict with keys:
              - ``text``: decoded output text (generated portion only)
              - ``log_prob``: score of chosen path
              - ``num_backtracks``: total backtrack events observed
              - ``tokens_generated``: number of tokens in the chosen output
        """
        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt"
        ).input_ids.squeeze(0)
        prompt_len = prompt_ids.shape[0]

        # Queue of states to explore (1-D CPU tensors)
        current_queue: list[torch.Tensor] = [prompt_ids]
        visited_states: set[str] = {
            self.tokenizer.decode(prompt_ids.tolist(), skip_special_tokens=True)
        }

        all_completed: list[dict] = []
        total_backtracks = 0

        # Their code: sqrt(b), not sqrt(n)
        max_backtrack_states = max(1, int(math.isqrt(self.b)))

        for round_idx in range(self.b + 1):
            if not current_queue:
                break

            next_queue: list[torch.Tensor] = []

            for state_ids in current_queue:
                completed, backtracked = self._generate_candidates(
                    state_ids, prompt_len
                )
                all_completed.extend(completed)

                for cand in backtracked:
                    if len(next_queue) >= max_backtrack_states:
                        break
                    total_backtracks += 1
                    rolled_back = self.backtrack(
                        cand["input_ids"].squeeze(0), prompt_ids
                    )
                    rb_text = self.tokenizer.decode(
                        rolled_back.tolist(), skip_special_tokens=True
                    )
                    if rb_text not in visited_states:
                        visited_states.add(rb_text)
                        next_queue.append(rolled_back)

            current_queue = next_queue

        # Aggregate and select best
        all_completed = self._aggregate_completed(all_completed)

        if not all_completed:
            # Fallback: greedy from original prompt
            device = next(self.model.parameters()).device
            with torch.no_grad():
                fallback_out = self.model.generate(
                    prompt_ids.unsqueeze(0).to(device),
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.eos_token_id,
                )
            seq = fallback_out[0].cpu()
            text = self.tokenizer.decode(seq.tolist(), skip_special_tokens=True)
            all_completed = [
                {"input_ids": seq.unsqueeze(0), "text": text, "score": 0.0}
            ]

        best = max(all_completed, key=lambda c: c["score"])
        best_ids = best["input_ids"].squeeze(0)
        generated_ids = best_ids[prompt_len:]

        return {
            "text": self.tokenizer.decode(
                generated_ids.tolist(), skip_special_tokens=True
            ),
            "log_prob": best["score"],
            "num_backtracks": total_backtracks,
            "tokens_generated": generated_ids.shape[0],
        }
