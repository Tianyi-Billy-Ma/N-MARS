"""Build D_op and D_back datasets for Self-Backtracking on MATH.

Uses mtybilly/MetaMathQA (MATH-50K config) — a curated 50K MATH-only split
with columns: id, query, response, type, original_question.

D_op: optimal paths — gold CoT from the curated MATH-50K split.
D_back: backtrack traces — perturbed arithmetic/numeric errors followed by
        <backtrack> and the correct continuation.

Usage:
    python -m baselines.self_backtracking.build_data_math \
        --output_dir data/self_backtracking/math \
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
# MetaMath parsing
# ---------------------------------------------------------------------------

# Final answer pattern: "The answer is: <answer>"
_FINAL_ANSWER_RE = re.compile(r"The answer is:\s*(.+?)\.?\s*$")


def parse_metamath_response(response: str) -> tuple[list[str], str]:
    """Split a MetaMath response into reasoning steps and final answer.

    Returns:
        steps: list of non-empty lines (reasoning chain)
        final_answer: the answer string after "The answer is:"
    """
    final_answer = ""
    m = _FINAL_ANSWER_RE.search(response)
    if m:
        final_answer = m.group(1).strip()

    steps = [line for line in response.splitlines() if line.strip()]
    return steps, final_answer


# ---------------------------------------------------------------------------
# Numeric perturbation (handles LaTeX math and plain numbers)
# ---------------------------------------------------------------------------

# Match patterns like:
#   "3 + 5 = 8", "= 42", "= \frac{3}{4}", "= $12"
#   Also integers/floats at end of equations: "= 91", "= -3.5"
_NUM_RESULT_RE = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*([+\-*/])\s*(\d[\d,]*(?:\.\d+)?)"
    r"\s*=\s*(\$?)([\d,]+(?:\.\d+)?)"
    r"|=\s*(\$?)([\d,]+(?:\.\d+)?)"
)


def _parse_number(s: str) -> float | None:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def _format_number(original: str, value: float) -> str:
    has_comma = "," in original
    has_decimal = "." in original.replace(",", "")

    if has_decimal:
        decimal_part = original.replace(",", "").split(".")[-1]
        decimal_places = max(len(decimal_part), 1)
        formatted = f"{value:.{decimal_places}f}"
    else:
        formatted = str(int(round(value)))

    if has_comma and not has_decimal:
        try:
            formatted = f"{int(round(value)):,}"
        except (ValueError, OverflowError):
            pass

    return formatted


def perturb_step(step: str, rng: random.Random) -> str | None:
    """Perturb a numeric result in a reasoning step.

    Returns the perturbed step string, or None if no numeric result found.
    """
    match = _NUM_RESULT_RE.search(step)
    if match is None:
        return None

    if match.group(5) is not None:
        result_str = match.group(5)
        dollar_prefix = match.group(4)
    elif match.group(7) is not None:
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

    if perturbed_value < 0 and dollar_prefix:
        perturbed_value = original_value + abs(delta)

    new_result = _format_number(result_str, perturbed_value)
    new_result_with_dollar = (dollar_prefix or "") + new_result
    old_result_with_dollar = (dollar_prefix or "") + result_str

    start, end = match.span()
    old_substring = step[start:end]
    new_substring = old_substring.replace(old_result_with_dollar, new_result_with_dollar, 1)
    return step[:start] + new_substring + step[end:]


def has_numeric_result(step: str) -> bool:
    return bool(_NUM_RESULT_RE.search(step))


# ---------------------------------------------------------------------------
# D_op / D_back construction
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = "###Question: {question}\n###Response:\n"


def build_op_text(question: str, response: str) -> str:
    """Build a D_op (optimal path) training example."""
    return f"{PROMPT_TEMPLATE.format(question=question)}{response}"


def build_back_text(
    question: str,
    steps: list[str],
    full_response: str,
    rng: random.Random,
) -> str | None:
    """Build a D_back (backtrack trace) training example.

    Randomly selects one step with a numeric result, perturbs it, inserts
    <backtrack>, then continues with the correct step and remainder.
    """
    numeric_indices = [i for i, s in enumerate(steps) if has_numeric_result(s)]
    if not numeric_indices:
        return None

    err_idx = rng.choice(numeric_indices)
    perturbed = perturb_step(steps[err_idx], rng)
    if perturbed is None:
        return None

    prefix_steps = steps[:err_idx]
    correct_steps = steps[err_idx:]

    parts: list[str] = []
    if prefix_steps:
        parts.append("\n".join(prefix_steps))
    parts.append(perturbed)
    parts.append("<backtrack>")
    parts.append("\n".join(correct_steps))

    body = "\n".join(parts)
    return f"{PROMPT_TEMPLATE.format(question=question)}{body}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_datasets(
    error_rate: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Load the curated MATH-50K dataset and construct D_op and D_back samples."""
    rng = random.Random(seed)

    ds = load_dataset("mtybilly/MetaMathQA", "MATH-50K", split="train")
    print(f"Loaded {len(ds)} MATH-50K samples")

    op_samples: list[dict] = []
    back_samples: list[dict] = []

    for row in ds:
        question = row["query"]
        response = row["response"]
        steps, final_answer = parse_metamath_response(response)

        op_text = build_op_text(question, response)
        op_samples.append(
            {
                "text": op_text,
                "has_backtrack": False,
                "split": "op",
            }
        )

        if rng.random() < error_rate:
            back_text = build_back_text(question, steps, response, rng)
            if back_text is not None:
                back_samples.append(
                    {
                        "text": back_text,
                        "has_backtrack": True,
                        "split": "back",
                    }
                )

    return op_samples, back_samples


def save_datasets(
    op_samples: list[dict],
    back_samples: list[dict],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = op_samples + back_samples

    hf_dataset = Dataset.from_list(all_samples)
    hf_dataset.save_to_disk(str(output_dir / "hf_dataset"))

    op_dataset = Dataset.from_list(op_samples)
    back_dataset = Dataset.from_list(back_samples)
    dataset_dict = DatasetDict({"op": op_dataset, "back": back_dataset})
    dataset_dict.save_to_disk(str(output_dir / "hf_dataset_split"))

    for name, samples in [("data", all_samples), ("op", op_samples), ("back", back_samples)]:
        path = output_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

    print(f"Saved {len(op_samples)} D_op + {len(back_samples)} D_back samples")
    print(f"  HuggingFace dataset: {output_dir / 'hf_dataset'}")
    print(f"  JSONL: {output_dir / 'data.jsonl'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Self-Backtracking datasets from MetaMathQA")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/self_backtracking/math"),
    )
    parser.add_argument(
        "--error_rate",
        type=float,
        default=0.5,
        help="Fraction of samples to perturb into D_back (default: 0.5)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Building Self-Backtracking datasets from mtybilly/MetaMathQA MATH-50K")
    print(f"  error_rate={args.error_rate}, seed={args.seed}")
    print(f"  output_dir={args.output_dir}")

    op_samples, back_samples = build_datasets(
        error_rate=args.error_rate,
        seed=args.seed,
    )

    save_datasets(op_samples, back_samples, args.output_dir)


if __name__ == "__main__":
    main()
