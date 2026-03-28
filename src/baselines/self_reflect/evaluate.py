"""Evaluate Self-Reflect baseline on GSM8K or MATH-500.

Self-Reflect uses standard autoregressive generation — no special tokens,
no backtracking decoder. We generate the full output (which may contain
errors + NL reflection + corrections) and extract the final answer.

Usage:
    python -m baselines.self_reflect.evaluate \
        --model_path outputs/self-reflect-llama3.2-1b-gsm8k \
        --output_path outputs/self-reflect-llama3.2-1b-gsm8k/eval.json \
        --task gsm8k --seed 42
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Answer extraction (same as self_backtracking evaluate)
# ---------------------------------------------------------------------------

_GSM8K_PATTERNS = [
    re.compile(r"[Tt]he answer is:?\s*(\-?[\d,]+)"),
    re.compile(r"####\s*(\-?[\d,]+)"),
]


def extract_answer_gsm8k(text: str) -> str | None:
    for pat in _GSM8K_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).replace(",", "").strip()
    return None


def extract_boxed(text: str) -> str | None:
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


def extract_answer_math(text: str) -> str | None:
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    m = re.search(r"[Tt]he answer is:?\s*(.+?)\.?\s*$", text, re.MULTILINE)
    if m:
        return m.group(1).strip()
    return None


def normalize_answer(answer: str) -> str:
    s = answer.replace(",", "").strip().lstrip("0") or "0"
    return s


def normalize_math_answer(answer: str) -> str:
    s = answer.strip()
    s = s.replace("\\$", "").replace("$", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    return s.strip().rstrip(".")


def answers_match(pred: str | None, gold: str, task: str) -> bool:
    if pred is None:
        return False
    if task == "gsm8k":
        try:
            return float(normalize_answer(pred)) == float(
                normalize_answer(gold)
            )
        except ValueError:
            return normalize_answer(pred) == normalize_answer(gold)
    else:
        return normalize_math_answer(pred) == normalize_math_answer(gold)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try LoRA adapter first
    try:
        from peft import PeftModel

        adapter_cfg = Path(model_path) / "adapter_config.json"
        if adapter_cfg.exists():
            with open(adapter_cfg) as f:
                cfg = json.load(f)
            base_name = cfg.get("base_model_name_or_path")
            if base_name:
                base = AutoModelForCausalLM.from_pretrained(
                    base_name, torch_dtype=torch.float16, device_map="auto",
                )
                model = PeftModel.from_pretrained(base, model_path)
                model = model.merge_and_unload()
                return model, tokenizer
    except (ImportError, Exception):
        pass

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto",
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def format_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:\n"


def run_evaluation(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.model_path)
    model.eval()

    if args.task == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split="test")
        q_col, a_col = "question", "answer"
    else:
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        q_col, a_col = "problem", "answer"

    results_per_sample: list[dict] = []
    correct = 0
    total_tokens = 0
    wall_start = time.time()

    for idx, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        question = sample[q_col]
        gold_raw = sample[a_col]

        if args.task == "gsm8k":
            gold = extract_answer_gsm8k(gold_raw) or gold_raw.strip()
        else:
            gold = gold_raw.strip()

        prompt = format_prompt(question)
        device = next(model.parameters()).device
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generated_ids = out[0][prompt_len:]
        text = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)
        n_tokens = generated_ids.shape[0]
        total_tokens += n_tokens

        if args.task == "gsm8k":
            pred = extract_answer_gsm8k(text)
        else:
            pred = extract_answer_math(text)

        is_correct = answers_match(pred, gold, args.task)
        if is_correct:
            correct += 1

        results_per_sample.append({
            "idx": idx,
            "question": question,
            "gold_answer": gold,
            "predicted_answer": pred,
            "correct": is_correct,
            "generated_text": text,
            "tokens_generated": n_tokens,
        })

    wall_elapsed = time.time() - wall_start
    n_samples = len(dataset)

    metrics = {
        "accuracy": correct / n_samples if n_samples > 0 else 0.0,
        "correct": correct,
        "total": n_samples,
        "avg_tokens_generated": total_tokens / n_samples if n_samples > 0 else 0.0,
        "total_wall_clock_seconds": wall_elapsed,
        "task": args.task,
        "mode": "greedy",
        "config": {
            "model_path": args.model_path,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
    }

    output = {"metrics": metrics, "samples": results_per_sample}
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== Results ({args.task}) ===")
    print(f"Accuracy:             {metrics['accuracy']:.4f}  ({correct}/{n_samples})")
    print(f"Avg tokens generated: {metrics['avg_tokens_generated']:.1f}")
    print(f"Wall-clock time:      {wall_elapsed:.1f}s")
    print(f"Results saved to:     {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Self-Reflect baseline"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--task", type=str, default="gsm8k",
        choices=["gsm8k", "math500"],
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
