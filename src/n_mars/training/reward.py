"""Reward functions for N-MARS GRPO training.

Implements Section 3.3 multi-objective reward:
  R(tau) = R_inc + R_pen

Also includes:
- Alternative reward variants for ablation (SCoRe, MGRPO)

Shared utilities (stack_postprocess, answer extraction) are imported from
their canonical locations to avoid divergence.
"""

from __future__ import annotations

from n_mars.inference.answer_extraction import answers_match, extract_gsm8k_answer
from n_mars.inference.decoder import stack_postprocess

# ---------------------------------------------------------------------------
# Core reward components
# ---------------------------------------------------------------------------

def compute_reward_inc(predicted_answer: str | None, gold_answer: str | None) -> float:
    """R_inc: +1 if final answer is correct, -1 otherwise."""
    return 1.0 if answers_match(predicted_answer, gold_answer) else -1.0


def compute_reward_pen(
    tau: list[int],
    undo_token_id: int,
    kappa: float,
    r_inc: float,
) -> float:
    """R_pen: backtrack penalty/reward based on <UNDO> budget.

    Args:
        tau: Token ids of the raw trajectory (before stack post-processing).
        undo_token_id: Token id of the <UNDO> special token.
        kappa: Budget hyperparameter (fraction of |tau| allowed as <UNDO> tokens).
        r_inc: Value of R_inc for this trajectory (+1 or -1).

    Returns:
        +1.0 if correct AND N_undo <= kappa * |tau|
        +0.5 if correct AND N_undo > kappa * |tau|
        -1.0 otherwise
    """
    n_undo = tau.count(undo_token_id)
    budget = kappa * len(tau)

    if r_inc > 0:
        return 1.0 if n_undo <= budget else 0.5
    return -1.0


def compute_total_reward(
    tau: list[int],
    gold_answer: str | None,
    undo_token_id: int,
    kappa: float,
    tokenizer,
) -> float:
    """Compute R(tau) = R_inc + R_pen.

    Args:
        tau: Raw token ids of the trajectory (response only, not prompt).
        gold_answer: Ground-truth answer string.
        undo_token_id: Token id of <UNDO>.
        kappa: <UNDO> budget fraction.
        tokenizer: Used to decode tau* for answer extraction.

    Returns:
        Total scalar reward.
    """
    # Post-process to get the clean sequence tau*
    cleaned = stack_postprocess(tau, undo_token_id)
    decoded = tokenizer.decode(cleaned, skip_special_tokens=True)
    predicted = extract_gsm8k_answer(decoded)

    r_inc = compute_reward_inc(predicted, gold_answer)
    r_pen = compute_reward_pen(tau, undo_token_id, kappa, r_inc)
    return r_inc + r_pen


# ---------------------------------------------------------------------------
# Alternative reward variants (ablation)
# ---------------------------------------------------------------------------

def compute_reward_score(
    tau: list[int],
    gold_answer: str | None,
    undo_token_id: int,
    tokenizer,
) -> float:
    """SCoRe-style reward: +1 for correct, 0 otherwise (no penalty term)."""
    cleaned = stack_postprocess(tau, undo_token_id)
    decoded = tokenizer.decode(cleaned, skip_special_tokens=True)
    predicted = extract_gsm8k_answer(decoded)
    return 1.0 if answers_match(predicted, gold_answer) else 0.0


def compute_reward_mgrpo(
    tau: list[int],
    gold_answer: str | None,
    undo_token_id: int,
    kappa: float,
    tokenizer,
) -> float:
    """MGRPO-style reward: binary correctness + soft length penalty.

    Simplified variant that uses a length-ratio penalty instead of the
    two-threshold budget from N-MARS.
    """
    cleaned = stack_postprocess(tau, undo_token_id)
    decoded = tokenizer.decode(cleaned, skip_special_tokens=True)
    predicted = extract_gsm8k_answer(decoded)

    if answers_match(predicted, gold_answer):
        n_undo = tau.count(undo_token_id)
        length_penalty = max(0.0, 1.0 - (n_undo / max(1, kappa * len(tau))))
        return 1.0 + length_penalty
    return -1.0
