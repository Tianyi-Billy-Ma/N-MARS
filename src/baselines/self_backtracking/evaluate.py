"""Evaluation script for the Self-Backtracking baseline.

Supports GSM8K and MATH-500 benchmarks.

Usage
-----
# GSM8K — full backtracking evaluation
python -m baselines.self_backtracking.evaluate \
    --model_path outputs/self-backtrack-llama3.2-1b-gsm8k \
    --output_path outputs/self-backtrack-llama3.2-1b-gsm8k/eval_results.json \
    --task gsm8k \
    --b 1 --n 32 --max_new_tokens 512 --seed 42

# GSM8K — greedy-only (control)
python -m baselines.self_backtracking.evaluate \
    --model_path outputs/self-backtrack-llama3.2-1b-gsm8k \
    --output_path outputs/self-backtrack-llama3.2-1b-gsm8k/eval_greedy.json \
    --task gsm8k --greedy_only --seed 42

# MATH-500 — full backtracking evaluation
python -m baselines.self_backtracking.evaluate \
    --model_path outputs/self-backtrack-llama3.1-8b-math500 \
    --output_path outputs/self-backtrack-llama3.1-8b-math500/eval_results.json \
    --task math500 \
    --b 1 --n 32 --max_new_tokens 1024 --seed 42

# MATH-500 — greedy-only (control)
python -m baselines.self_backtracking.evaluate \
    --model_path outputs/self-backtrack-llama3.1-8b-math500 \
    --output_path outputs/self-backtrack-llama3.1-8b-math500/eval_greedy.json \
    --task math500 --greedy_only --seed 42
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.self_backtracking.decode import SelfBackTrackingDecoder

# ---------------------------------------------------------------------------
# GSM8K answer extraction
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


# ---------------------------------------------------------------------------
# MATH-500 answer extraction
# ---------------------------------------------------------------------------

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
    """Extract answer from MATH-500 model output.

    Tries \\boxed{} first, then 'The answer is: ...' fallback.
    """
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
    s = s.strip().rstrip(".")
    return s


def answers_match_math(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    return normalize_math_answer(pred) == normalize_math_answer(gold)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_path: str):
    """Load model and tokenizer, with optional LoRA adapter via peft."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Ensure <backtrack> is registered as a special token
    if "<backtrack>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<backtrack>"]})

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try to load as a PEFT/LoRA adapter first
    try:
        from peft import PeftModel  # type: ignore[import]

        base_model_name = _read_peft_base_model(model_path)
        if base_model_name:
            print(f"Loading base model '{base_model_name}' + LoRA adapter from '{model_path}'")
            base = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            base.resize_token_embeddings(len(tokenizer))
            model = PeftModel.from_pretrained(base, model_path)
            model = model.merge_and_unload()
            return model, tokenizer
    except (ImportError, Exception):
        pass

    # Plain HF model
    print(f"Loading model from '{model_path}'")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def _read_peft_base_model(model_path: str) -> str | None:
    """Return the base model name from adapter_config.json if it exists."""
    adapter_cfg = Path(model_path) / "adapter_config.json"
    if not adapter_cfg.exists():
        return None
    try:
        with open(adapter_cfg) as fh:
            cfg = json.load(fh)
        return cfg.get("base_model_name_or_path")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Greedy generation
# ---------------------------------------------------------------------------

def greedy_generate(model, tokenizer, prompt: str, max_new_tokens: int) -> dict:
    """Run standard greedy decoding and return result dict."""
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated_ids = out[0][prompt_len:]
    text = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)
    return {
        "text": text,
        "log_prob": 0.0,
        "num_backtracks": 0,
        "tokens_generated": generated_ids.shape[0],
    }


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def format_prompt(question: str) -> str:
    return f"###Question: {question}\n###Response:\n"


def run_evaluation(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.model_path)
    model.eval()

    # Build decoder (only used in backtracking mode)
    if not args.greedy_only:
        decoder = SelfBackTrackingDecoder(
            model=model,
            tokenizer=tokenizer,
            b=args.b,
            n=args.n,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    # Load dataset based on task
    if args.task == "math500":
        print("Loading MATH-500 test set...")
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    else:
        print("Loading GSM8K test set...")
        dataset = load_dataset("gsm8k", "main", split="test")

    results_per_sample: list[dict] = []
    correct = 0
    total_tokens = 0
    total_backtracks = 0

    wall_start = time.time()

    for idx, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        if args.task == "math500":
            question = sample["problem"]
            gold_answer = normalize_math_answer(sample["answer"])
        else:
            question = sample["question"]
            gold_raw = sample["answer"]
            # GSM8K gold answers end with "#### <number>"
            gold_answer = extract_answer_gsm8k(gold_raw) or gold_raw.strip()

        prompt = format_prompt(question)

        if args.greedy_only:
            result = greedy_generate(model, tokenizer, prompt, args.max_new_tokens)
        else:
            result = decoder.generate(prompt)

        if args.task == "math500":
            pred_answer = extract_answer_math(result["text"])
            is_correct = answers_match_math(pred_answer, gold_answer)
        else:
            pred_answer = extract_answer_gsm8k(result["text"])
            is_correct = answers_match_gsm8k(pred_answer, gold_answer)

        if is_correct:
            correct += 1
        total_tokens += result["tokens_generated"]
        total_backtracks += result["num_backtracks"]

        results_per_sample.append(
            {
                "idx": idx,
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": pred_answer,
                "correct": is_correct,
                "generated_text": result["text"],
                "log_prob": result["log_prob"],
                "num_backtracks": result["num_backtracks"],
                "tokens_generated": result["tokens_generated"],
            }
        )

    wall_elapsed = time.time() - wall_start
    n_samples = len(dataset)

    metrics = {
        "accuracy": correct / n_samples if n_samples > 0 else 0.0,
        "correct": correct,
        "total": n_samples,
        "avg_tokens_generated": total_tokens / n_samples if n_samples > 0 else 0.0,
        "avg_backtracks_per_sample": total_backtracks / n_samples if n_samples > 0 else 0.0,
        "total_wall_clock_seconds": wall_elapsed,
        "mode": "greedy" if args.greedy_only else "self_backtracking",
        "config": {
            "task": args.task,
            "model_path": args.model_path,
            "b": args.b if not args.greedy_only else None,
            "n": args.n if not args.greedy_only else None,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature if not args.greedy_only else None,
            "seed": args.seed,
        },
    }

    output = {
        "metrics": metrics,
        "samples": results_per_sample,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(output, fh, indent=2)

    print("\n=== Results ===")
    print(f"Task:                  {args.task}")
    print(f"Accuracy:              {metrics['accuracy']:.4f}  ({correct}/{n_samples})")
    print(f"Avg tokens generated:  {metrics['avg_tokens_generated']:.1f}")
    print(f"Avg backtracks/sample: {metrics['avg_backtracks_per_sample']:.2f}")
    print(f"Wall-clock time:       {wall_elapsed:.1f}s")
    print(f"Results saved to:      {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Self-Backtracking decoder on GSM8K or MATH-500"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model or LoRA adapter directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to write JSON results file",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["gsm8k", "math500"],
        default="gsm8k",
        help="Evaluation benchmark: 'gsm8k' or 'math500' (default: gsm8k)",
    )
    # Backtracking hyperparameters
    parser.add_argument(
        "--b", type=int, default=1, help="Number of backtracking rounds (default: 1)"
    )
    parser.add_argument(
        "--n", type=int, default=32, help="Candidates per round (default: 32)"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, help="Max new tokens per generation"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)"
    )
    # Control flags
    parser.add_argument(
        "--greedy_only",
        action="store_true",
        help="Skip backtracking; use standard greedy decoding",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
