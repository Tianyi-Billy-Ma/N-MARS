"""Inference cost analysis for N-MARS vs SFT baseline.

Measures UNDO token usage, token overhead, and latency on GSM8K or MATH-500.
The UNDO token ``<|BACKTRACK|>`` is mapped to ``<|reserved_special_token_0|>``
(id=128002) in Llama tokenizers. Stack post-processing removes tokens that were
"undone": scan left-to-right, push normal tokens, pop on UNDO.

Usage
-----
# N-MARS only
python -m n_mars.scripts.inference_cost \\
    --model_path outputs/nmars-llama3.1-8b-gsm8k \\
    --output_path outputs/nmars-inference-cost.json \\
    --task gsm8k --max_new_tokens 512 --seed 42

# N-MARS + SFT baseline comparison
python -m n_mars.scripts.inference_cost \\
    --model_path outputs/nmars-llama3.1-8b-gsm8k \\
    --baseline_model_path outputs/sft-llama3.1-8b-gsm8k \\
    --output_path outputs/nmars-inference-cost.json \\
    --task gsm8k --max_new_tokens 512 --seed 42
"""

import argparse
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# UNDO token constants
# ---------------------------------------------------------------------------

UNDO_TOKEN_STR = "<|BACKTRACK|>"
UNDO_TOKEN_FALLBACK_ID = 128002  # <|reserved_special_token_0|> in Llama tokenizers

# ---------------------------------------------------------------------------
# Answer extraction (copied from baselines.self_backtracking.evaluate)
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


# ---------------------------------------------------------------------------
# Stack post-processing
# ---------------------------------------------------------------------------


def apply_stack_postprocess(token_ids: list[int], undo_id: int) -> list[int]:
    """Simulate UNDO stack: push normal tokens, pop on UNDO token.

    Returns the final stack contents (tokens that survive post-processing).
    """
    stack: list[int] = []
    for tok in token_ids:
        if tok == undo_id:
            if stack:
                stack.pop()
        else:
            stack.append(tok)
    return stack


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


def get_undo_token_id(tokenizer) -> int:
    """Resolve the UNDO token id from the tokenizer vocabulary."""
    vocab = tokenizer.get_vocab()
    if UNDO_TOKEN_STR in vocab:
        return vocab[UNDO_TOKEN_STR]
    # Fall back to reserved_special_token_0 by id
    return UNDO_TOKEN_FALLBACK_ID


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def greedy_generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> dict:
    """Run greedy decoding, return raw token ids and generated text."""
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    generated_ids = out[0][prompt_len:].tolist()
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {
        "raw_ids": generated_ids,
        "text": text,
        "latency_ms": latency_ms,
    }


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------


def format_prompt(question: str) -> str:
    return f"###Question: {question}\n###Response:\n"


def load_task_dataset(task: str):
    """Load GSM8K or MATH-500 test set."""
    if task == "math500":
        print("Loading MATH-500 test set...")
        return load_dataset("HuggingFaceH4/MATH-500", split="test")
    print("Loading GSM8K test set...")
    return load_dataset("gsm8k", "main", split="test")


def get_gold_answer(sample: dict, task: str) -> str:
    """Extract ground-truth answer from a dataset sample."""
    if task == "math500":
        return normalize_math_answer(sample["answer"])
    raw = sample["answer"]
    return extract_answer_gsm8k(raw) or raw.strip()


def check_correct(pred_text: str, gold: str, task: str) -> bool:
    """Check whether the predicted text matches the gold answer."""
    if task == "math500":
        pred = extract_answer_math(pred_text)
        return answers_match_math(pred, gold)
    pred = extract_answer_gsm8k(pred_text)
    return answers_match_gsm8k(pred, gold)


def evaluate_model(
    model,
    tokenizer,
    dataset,
    task: str,
    max_new_tokens: int,
    is_nmars: bool,
) -> dict:
    """Run greedy evaluation on the full dataset, collecting cost metrics.

    For N-MARS models, counts UNDO tokens and applies stack post-processing.
    For baseline models, UNDO analysis is skipped (undo_count == 0 always).

    Returns a metrics dict matching the output JSON schema.
    """
    undo_id = get_undo_token_id(tokenizer) if is_nmars else -1

    undo_counts: list[int] = []
    total_token_list: list[int] = []
    final_token_list: list[int] = []
    latency_list: list[float] = []
    correct = 0

    for sample in tqdm(dataset, desc="Evaluating"):
        question = sample["problem"] if task == "math500" else sample["question"]
        gold = get_gold_answer(sample, task)
        prompt = format_prompt(question)

        result = greedy_generate(model, tokenizer, prompt, max_new_tokens)
        raw_ids = result["raw_ids"]
        latency_list.append(result["latency_ms"])

        total_token_list.append(len(raw_ids))

        if is_nmars:
            undo_count = raw_ids.count(undo_id)
            final_ids = apply_stack_postprocess(raw_ids, undo_id)
        else:
            undo_count = 0
            final_ids = raw_ids

        undo_counts.append(undo_count)
        final_token_list.append(len(final_ids))

        # Decode final (post-processed) tokens for answer extraction
        final_text = tokenizer.decode(final_ids, skip_special_tokens=True)
        if check_correct(final_text, gold, task):
            correct += 1

    n = len(dataset)
    undo_hist = dict(sorted(Counter(undo_counts).items()))

    avg_total = statistics.mean(total_token_list)
    avg_final = statistics.mean(final_token_list)
    avg_overhead = avg_total / avg_final if avg_final > 0 else 1.0

    return {
        "accuracy": correct / n if n > 0 else 0.0,
        "avg_undo_per_seq": statistics.mean(undo_counts),
        "median_undo_per_seq": statistics.median(undo_counts),
        "max_undo_per_seq": max(undo_counts),
        "pct_seqs_with_undo": sum(1 for c in undo_counts if c > 0) / n if n > 0 else 0.0,
        "avg_total_tokens": avg_total,
        "avg_final_tokens": avg_final,
        "avg_overhead_ratio": avg_overhead,
        "avg_latency_ms": statistics.mean(latency_list),
        "undo_count_histogram": {str(k): v for k, v in undo_hist.items()},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inference cost analysis for N-MARS vs SFT baseline"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained N-MARS model",
    )
    parser.add_argument(
        "--baseline_model_path",
        type=str,
        default=None,
        help="Path to SFT baseline model (optional, for comparison)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to write JSON results",
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
        help="Max new tokens per generation (default: 512)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    dataset = load_task_dataset(args.task)

    # --- N-MARS model ---
    print(f"\n=== Evaluating N-MARS model: {args.model_path} ===")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    nmars_metrics = evaluate_model(
        model, tokenizer, dataset, args.task, args.max_new_tokens, is_nmars=True
    )
    del model

    output: dict = {"nmars": nmars_metrics}

    # --- Optional baseline ---
    if args.baseline_model_path:
        print(f"\n=== Evaluating baseline model: {args.baseline_model_path} ===")
        bl_model, bl_tokenizer = load_model_and_tokenizer(args.baseline_model_path)
        baseline_metrics = evaluate_model(
            bl_model, bl_tokenizer, dataset, args.task, args.max_new_tokens, is_nmars=False
        )
        del bl_model
        output["baseline"] = baseline_metrics

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)

    print("\n=== N-MARS Inference Cost Summary ===")
    m = nmars_metrics
    print(f"Task:                {args.task}")
    print(f"Accuracy:            {m['accuracy']:.4f}")
    print(f"Avg UNDO / seq:      {m['avg_undo_per_seq']:.2f}")
    print(f"Pct seqs with UNDO:  {m['pct_seqs_with_undo']:.2%}")
    print(f"Avg total tokens:    {m['avg_total_tokens']:.1f}")
    print(f"Avg final tokens:    {m['avg_final_tokens']:.1f}")
    print(f"Avg overhead ratio:  {m['avg_overhead_ratio']:.3f}")
    print(f"Avg latency (ms):    {m['avg_latency_ms']:.1f}")
    if "baseline" in output:
        b = output["baseline"]
        print("\n=== Baseline Summary ===")
        print(f"Accuracy:            {b['accuracy']:.4f}")
        print(f"Avg total tokens:    {b['avg_total_tokens']:.1f}")
        print(f"Avg latency (ms):    {b['avg_latency_ms']:.1f}")
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
