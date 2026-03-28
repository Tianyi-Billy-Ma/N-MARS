"""Compute-matched comparison: SFT baseline with self-consistency vs N-MARS.

Uses EXP-1 (inference_cost.py) results to determine the total token budget that
N-MARS spends, then runs the SFT baseline with majority-voting (self-consistency)
at the same token budget for increasing values of K independent samples.

Usage
-----
python -m n_mars.scripts.compute_matched \\
    --model_path outputs/sft-llama3.1-8b-gsm8k \\
    --nmars_results outputs/nmars-inference-cost.json \\
    --output_path outputs/compute-matched-results.json \\
    --k_values 1,2,4,8 \\
    --task gsm8k --max_new_tokens 512 --seed 42

# Custom temperature and k values
python -m n_mars.scripts.compute_matched \\
    --model_path outputs/sft-llama3.1-8b-gsm8k \\
    --nmars_results outputs/nmars-inference-cost.json \\
    --output_path outputs/compute-matched-results.json \\
    --k_values 1,2,4,8,16 --temperature 0.7 \\
    --task gsm8k --max_new_tokens 512 --seed 42
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Answer extraction (mirrored from baselines.self_backtracking.evaluate)
# ---------------------------------------------------------------------------

_GSM8K_ANSWER_PATTERNS = [
    re.compile(r"[Tt]he answer is\s*(\-?[\d,]+)"),
    re.compile(r"####\s*(\-?[\d,]+)"),
]


def extract_answer_gsm8k(text: str) -> str | None:
    """Return the first numeric answer found in text, or None."""
    for pat in _GSM8K_ANSWER_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).replace(",", "").strip()
    return None


def normalize_answer_gsm8k(answer: str) -> str:
    """Strip commas and leading zeros for comparison."""
    return answer.replace(",", "").strip().lstrip("0") or "0"


def answers_match_gsm8k(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return float(normalize_answer_gsm8k(pred)) == float(normalize_answer_gsm8k(gold))
    except ValueError:
        return normalize_answer_gsm8k(pred) == normalize_answer_gsm8k(gold)


def extract_boxed(text: str) -> str | None:
    """Extract the innermost \\boxed{...} content, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[idx + len("\\boxed{") : i - 1]


_MATH_THE_ANSWER_PATTERN = re.compile(r"[Tt]he answer is[:\s]+(.+?)(?:\.|$)", re.DOTALL)


def extract_answer_math(text: str) -> str | None:
    """Extract answer from MATH-500 model output (\\boxed{} or fallback)."""
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed.strip()
    m = _MATH_THE_ANSWER_PATTERN.search(text)
    if m:
        return m.group(1).strip()
    return None


def normalize_math_answer(answer: str) -> str:
    """Normalize a MATH answer string for string comparison."""
    s = answer.strip()
    s = s.replace("\\$", "").replace("$", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    return s.strip().rstrip(".")


def answers_match_math(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    return normalize_math_answer(pred) == normalize_math_answer(gold)


def extract_answer(text: str, task: str) -> str | None:
    """Dispatch answer extraction by task."""
    if task == "math500":
        return extract_answer_math(text)
    return extract_answer_gsm8k(text)


def check_correct(pred_text: str, gold: str, task: str) -> bool:
    """Check whether the predicted text matches the gold answer."""
    pred = extract_answer(pred_text, task)
    if task == "math500":
        return answers_match_math(pred, gold)
    return answers_match_gsm8k(pred, gold)


def get_gold_answer(sample: dict, task: str) -> str:
    """Extract ground-truth answer from a dataset sample."""
    if task == "math500":
        return normalize_math_answer(sample["answer"])
    raw = sample["answer"]
    return extract_answer_gsm8k(raw) or raw.strip()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(model_path: str):
    """Load a HuggingFace model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def format_prompt(question: str) -> str:
    return f"###Question: {question}\n###Response:\n"


def sample_k_completions(
    model,
    tokenizer,
    prompt: str,
    k: int,
    max_new_tokens: int,
    temperature: float,
) -> list[dict]:
    """Generate K independent completions via sampling.

    Returns a list of dicts with keys ``text`` and ``num_tokens``.
    K=1 with temperature=0 is equivalent to greedy decoding.
    """
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    # Expand prompt K times for batched generation
    expanded = input_ids.expand(k, -1)

    do_sample = temperature > 0.0
    with torch.no_grad():
        out = model.generate(
            expanded,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    results = []
    for i in range(k):
        gen_ids = out[i][prompt_len:].tolist()
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        results.append({"text": text, "num_tokens": len(gen_ids)})
    return results


def majority_vote(answers: list[str | None]) -> str | None:
    """Return the most common non-None answer, or None if all are None."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def load_nmars_token_budget(nmars_results_path: str) -> float:
    """Read avg_total_tokens from EXP-1 output JSON."""
    with open(nmars_results_path) as fh:
        data = json.load(fh)
    return data["nmars"]["avg_total_tokens"]


def load_task_dataset(task: str):
    """Load GSM8K or MATH-500 test set."""
    if task == "math500":
        print("Loading MATH-500 test set...")
        return load_dataset("HuggingFaceH4/MATH-500", split="test")
    print("Loading GSM8K test set...")
    return load_dataset("gsm8k", "main", split="test")


def evaluate_at_k(
    model,
    tokenizer,
    dataset,
    task: str,
    k: int,
    max_new_tokens: int,
    temperature: float,
) -> dict:
    """Evaluate majority-voting accuracy at K completions per sample.

    Returns a dict with ``accuracy``, ``total_tokens``, and ``avg_tokens_per_sample``.
    """
    correct = 0
    total_tokens = 0

    for sample in tqdm(dataset, desc=f"K={k}", leave=False):
        question = sample["problem"] if task == "math500" else sample["question"]
        gold = get_gold_answer(sample, task)
        prompt = format_prompt(question)

        completions = sample_k_completions(model, tokenizer, prompt, k, max_new_tokens, temperature)
        total_tokens += sum(c["num_tokens"] for c in completions)

        answers = [extract_answer(c["text"], task) for c in completions]
        voted = majority_vote(answers)

        if task == "math500":
            if answers_match_math(voted, gold):
                correct += 1
        else:
            if answers_match_gsm8k(voted, gold):
                correct += 1

    n = len(dataset)
    return {
        "accuracy": correct / n if n > 0 else 0.0,
        "total_tokens": total_tokens,
        "avg_tokens_per_sample": total_tokens / n if n > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute-matched comparison: SFT self-consistency vs N-MARS token budget"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to SFT baseline model",
    )
    parser.add_argument(
        "--nmars_results",
        type=str,
        required=True,
        help="Path to EXP-1 JSON output (from inference_cost.py)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to write JSON results",
    )
    parser.add_argument(
        "--k_values",
        type=str,
        default="1,2,4,8",
        help="Comma-separated K values for self-consistency (default: 1,2,4,8)",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["gsm8k", "math500"],
        default="gsm8k",
        help="Evaluation benchmark (default: gsm8k)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max new tokens per completion (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for K>1 completions (default: 0.7)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    nmars_avg_tokens = load_nmars_token_budget(args.nmars_results)
    print(f"N-MARS avg total tokens (from EXP-1): {nmars_avg_tokens:.1f}")
    print(f"Evaluating K values: {k_values}")

    dataset = load_task_dataset(args.task)
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    results_by_k: dict[str, dict] = {}

    for k in k_values:
        print(f"\n=== K={k} ===")
        metrics = evaluate_at_k(
            model, tokenizer, dataset, args.task, k, args.max_new_tokens, args.temperature
        )
        results_by_k[str(k)] = metrics
        print(f"  Accuracy:            {metrics['accuracy']:.4f}")
        print(
            f"  Avg tokens/sample:   {metrics['avg_tokens_per_sample']:.1f}"
            f"  (N-MARS budget: {nmars_avg_tokens:.1f})"
        )

    # Summarise accuracy vs token budget
    token_budgets = {k: v["avg_tokens_per_sample"] for k, v in results_by_k.items()}
    accuracies = {k: v["accuracy"] for k, v in results_by_k.items()}

    output = {
        "config": {
            "model_path": args.model_path,
            "nmars_results": args.nmars_results,
            "nmars_avg_total_tokens": nmars_avg_tokens,
            "task": args.task,
            "k_values": k_values,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
        },
        "results_by_k": results_by_k,
        "summary": {
            "token_budgets": token_budgets,
            "accuracies": accuracies,
        },
    }

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)

    print("\n=== Compute-Matched Comparison Summary ===")
    print(f"{'K':>4}  {'Avg tokens/sample':>20}  {'Accuracy':>10}")
    print("-" * 40)
    for k in k_values:
        r = results_by_k[str(k)]
        print(f"{k:>4}  {r['avg_tokens_per_sample']:>20.1f}  {r['accuracy']:>10.4f}")
    print(f"\nN-MARS token budget: {nmars_avg_tokens:.1f}")
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
