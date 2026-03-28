"""Build D_op and D_back datasets for the Self-Backtracking baseline (arXiv:2502.04404).

D_op: optimal paths — gold chain-of-thought from GSM8K.
D_back: backtrack traces — perturbed arithmetic errors followed by <backtrack> and the
        correct continuation.

Usage:
    python -m baselines.self_backtracking.build_data \
        --output_dir data/self_backtracking/gsm8k \
        --error_rate 0.5 \
        --seed 42
"""

import argparse
import json
import random
import re
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

# ---------------------------------------------------------------------------
# GSM8K parsing
# ---------------------------------------------------------------------------


def parse_gsm8k_answer(answer_field: str) -> tuple[list[str], str]:
    """Split a GSM8K answer field into reasoning steps and final answer.

    The answer field looks like:
        Step one text\\nStep two text\\n#### 42

    Returns:
        steps: list of non-empty lines before the #### delimiter
        final_answer: the numeric string after ####
    """
    parts = answer_field.split("####")
    cot_text = parts[0].rstrip()
    final_answer = parts[1].strip() if len(parts) > 1 else ""

    steps = [line for line in cot_text.splitlines() if line.strip()]
    return steps, final_answer


# ---------------------------------------------------------------------------
# Arithmetic perturbation
# ---------------------------------------------------------------------------

# Matches patterns like "3 + 5 = 8" or "= $12.00" or "= 42"
_RESULT_RE = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*([+\-*/])\s*(\d[\d,]*(?:\.\d+)?)\s*=\s*(\$?)([\d,]+(?:\.\d+)?)"
    r"|=\s*(\$?)([\d,]+(?:\.\d+)?)"
)


def _parse_number(s: str) -> float | None:
    """Parse a number string that may contain commas."""
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def _format_number(original: str, value: float) -> str:
    """Re-format a perturbed number to match the style of the original."""
    has_comma = "," in original
    has_decimal = "." in original.replace(",", "")

    if has_decimal:
        # Preserve same number of decimal places as original
        decimal_part = original.replace(",", "").split(".")[-1]
        decimal_places = max(len(decimal_part), 1)
        formatted = f"{value:.{decimal_places}f}"
    else:
        # Original was an integer — keep result as integer
        formatted = str(int(round(value)))

    # Restore comma thousands separators if original had them
    if has_comma and not has_decimal:
        try:
            formatted = f"{int(round(value)):,}"
        except (ValueError, OverflowError):
            pass

    return formatted


def perturb_step(step: str, rng: random.Random) -> str | None:
    """Attempt to perturb the arithmetic result in a reasoning step.

    Returns the perturbed step string, or None if no arithmetic was found.
    The result is changed by a random integer in [1, 5] (direction chosen randomly).
    """
    match = _RESULT_RE.search(step)
    if match is None:
        return None

    # Determine which group holds the result number
    if match.group(5) is not None:
        # Full equation match: groups 1-5
        result_str = match.group(5)
        dollar_prefix = match.group(4)
    elif match.group(7) is not None:
        # Bare "= N" match: groups 6-7
        result_str = match.group(7)
        dollar_prefix = match.group(6)
    else:
        return None

    original_value = _parse_number(result_str)
    if original_value is None:
        return None

    delta = rng.randint(1, 5)
    if rng.random() < 0.5:
        delta = -delta
    perturbed_value = original_value + delta

    # Avoid negative results for dollar amounts or counts
    if perturbed_value < 0:
        perturbed_value = original_value + abs(delta)

    new_result = _format_number(result_str, perturbed_value)
    new_result_with_dollar = (dollar_prefix or "") + new_result

    # Replace only the matched result portion
    old_result_with_dollar = (dollar_prefix or "") + result_str

    # Build replacement: swap old result for new result at the match location
    start, end = match.span()
    old_substring = step[start:end]
    new_substring = old_substring.replace(old_result_with_dollar, new_result_with_dollar, 1)
    return step[:start] + new_substring + step[end:]


def has_arithmetic(step: str) -> bool:
    """Return True if the step contains an arithmetic expression."""
    return bool(_RESULT_RE.search(step))


# ---------------------------------------------------------------------------
# D_op construction
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = "###Question: {question}\n###Response:\n"


def build_op_text(question: str, steps: list[str], final_answer: str) -> str:
    """Build a D_op (optimal path) training example."""
    body = "\n".join(steps)
    return f"{PROMPT_TEMPLATE.format(question=question)}{body}\nThe answer is {final_answer}."


# ---------------------------------------------------------------------------
# D_back construction
# ---------------------------------------------------------------------------


def build_back_text(
    question: str,
    steps: list[str],
    final_answer: str,
    rng: random.Random,
) -> str | None:
    """Build a D_back (backtrack trace) training example.

    Randomly selects one step containing arithmetic, perturbs it, inserts
    <backtrack>, then continues with the correct step and remainder.

    Returns None if no arithmetic step can be perturbed.
    """
    arithmetic_indices = [i for i, s in enumerate(steps) if has_arithmetic(s)]
    if not arithmetic_indices:
        return None

    err_idx = rng.choice(arithmetic_indices)
    perturbed = perturb_step(steps[err_idx], rng)
    if perturbed is None:
        return None

    prefix_steps = steps[:err_idx]
    correct_steps = steps[err_idx:]  # includes the step that was perturbed (correct version)

    parts: list[str] = []
    if prefix_steps:
        parts.append("\n".join(prefix_steps))
    parts.append(perturbed)
    parts.append("<backtrack>")
    parts.append("\n".join(correct_steps))
    parts.append(f"The answer is {final_answer}.")

    body = "\n".join(parts)
    return f"{PROMPT_TEMPLATE.format(question=question)}{body}"


# ---------------------------------------------------------------------------
# Main dataset construction
# ---------------------------------------------------------------------------


def build_datasets(
    error_rate: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Load GSM8K and construct D_op and D_back samples.

    Returns:
        op_samples: list of dicts with keys text, has_backtrack, split
        back_samples: list of dicts with keys text, has_backtrack, split
    """
    rng = random.Random(seed)

    gsm8k = load_dataset("gsm8k", "main")

    op_samples: list[dict] = []
    back_samples: list[dict] = []

    split_data = gsm8k["train"]
    for row in split_data:
        question = row["question"]
        steps, final_answer = parse_gsm8k_answer(row["answer"])

        # Always add an optimal path sample
        op_text = build_op_text(question, steps, final_answer)
        op_samples.append({"text": op_text, "has_backtrack": False, "split": "op"})

        # Conditionally add a backtrack sample
        if rng.random() < error_rate:
            back_text = build_back_text(question, steps, final_answer, rng)
            if back_text is not None:
                back_samples.append({"text": back_text, "has_backtrack": True, "split": "back"})

    return op_samples, back_samples


def save_datasets(
    op_samples: list[dict],
    back_samples: list[dict],
    output_dir: Path,
) -> None:
    """Save D_op and D_back as HuggingFace datasets (arrow) and JSONL."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = op_samples + back_samples

    # HuggingFace Dataset (arrow)
    hf_dataset = Dataset.from_list(all_samples)
    hf_dataset.save_to_disk(str(output_dir / "hf_dataset"))

    # Also save D_op and D_back as separate splits for convenience
    op_dataset = Dataset.from_list(op_samples)
    back_dataset = Dataset.from_list(back_samples)
    dataset_dict = DatasetDict({"op": op_dataset, "back": back_dataset})
    dataset_dict.save_to_disk(str(output_dir / "hf_dataset_split"))

    # JSONL for human inspection
    jsonl_path = output_dir / "data.jsonl"
    with open(jsonl_path, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")

    op_jsonl_path = output_dir / "op.jsonl"
    with open(op_jsonl_path, "w") as f:
        for sample in op_samples:
            f.write(json.dumps(sample) + "\n")

    back_jsonl_path = output_dir / "back.jsonl"
    with open(back_jsonl_path, "w") as f:
        for sample in back_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Saved {len(op_samples)} D_op samples and {len(back_samples)} D_back samples")
    print(f"  HuggingFace dataset: {output_dir / 'hf_dataset'}")
    print(f"  HuggingFace split dataset: {output_dir / 'hf_dataset_split'}")
    print(f"  JSONL (combined): {jsonl_path}")
    print(f"  JSONL (D_op): {op_jsonl_path}")
    print(f"  JSONL (D_back): {back_jsonl_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Self-Backtracking D_op and D_back datasets from GSM8K."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/self_backtracking/gsm8k"),
        help="Directory to write outputs (default: data/self_backtracking/gsm8k)",
    )
    parser.add_argument(
        "--error_rate",
        type=float,
        default=0.5,
        help="Fraction of samples to perturb into D_back examples (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Building Self-Backtracking datasets from GSM8K")
    print(f"  error_rate={args.error_rate}, seed={args.seed}")
    print(f"  output_dir={args.output_dir}")

    op_samples, back_samples = build_datasets(
        error_rate=args.error_rate,
        seed=args.seed,
    )

    save_datasets(op_samples, back_samples, args.output_dir)


if __name__ == "__main__":
    main()
