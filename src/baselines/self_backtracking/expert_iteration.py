"""Self-Backtracking Stage 2: Expert Iteration.

Iterative self-improvement loop:
1. Generate solutions with backtracking decoder on training prompts
2. Filter to keep only correct solutions
3. SFT on filtered correct paths (no masking, no <backtrack>)
4. Repeat K times

Based on: arXiv:2502.04404 (train_self_improvement.py)

Usage:
    python -m baselines.self_backtracking.expert_iteration \
        --model_path outputs/self-backtrack-llama3.2-1b-gsm8k \
        --output_dir outputs/self-backtrack-ei-llama3.2-1b-gsm8k \
        --num_iterations 3 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from baselines.self_backtracking.decode import SelfBackTrackingDecoder

logger = logging.getLogger(__name__)

BACKTRACK_TOKEN = "<backtrack>"
PROMPT_TEMPLATE = "###Question: {question}\n###Response:\n"

# ---------------------------------------------------------------------------
# Answer extraction (GSM8K)
# ---------------------------------------------------------------------------

_ANSWER_PATTERNS = [
    re.compile(r"[Tt]he answer is:?\s*(\-?[\d,]+)"),
    re.compile(r"####\s*(\-?[\d,]+)"),
]


def extract_answer(text: str) -> str | None:
    for pat in _ANSWER_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).replace(",", "").strip()
    return None


def normalize_answer(a: str) -> str:
    return a.replace(",", "").strip().lstrip("0") or "0"


def answers_match(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return float(normalize_answer(pred)) == float(normalize_answer(gold))
    except ValueError:
        return normalize_answer(pred) == normalize_answer(gold)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(model_path: str, bf16: bool = True):
    """Load model + tokenizer, handling LoRA adapters."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if BACKTRACK_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [BACKTRACK_TOKEN]}
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if bf16 else torch.float32

    # Try LoRA adapter
    adapter_cfg = Path(model_path) / "adapter_config.json"
    if adapter_cfg.exists():
        with open(adapter_cfg) as f:
            cfg = json.load(f)
        base_name = cfg.get("base_model_name_or_path")
        if base_name:
            logger.info("Loading base '%s' + LoRA from '%s'", base_name, model_path)
            base = AutoModelForCausalLM.from_pretrained(
                base_name, torch_dtype=dtype, use_cache=False,
            )
            base.resize_token_embeddings(len(tokenizer))
            model = PeftModel.from_pretrained(base, model_path)
            model = model.merge_and_unload()
            return model, tokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, use_cache=False,
    )
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


# ---------------------------------------------------------------------------
# Stage 2: Generate + Filter
# ---------------------------------------------------------------------------


def generate_and_filter(
    model,
    tokenizer,
    train_data,
    backtrack_id: int,
    args,
    iteration: int,
) -> list[dict]:
    """Generate solutions and filter to correct ones."""
    model.eval()

    decoder = SelfBackTrackingDecoder(
        model=model,
        tokenizer=tokenizer,
        b=args.b,
        n=args.n,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    correct_samples = []
    total = 0
    correct_count = 0

    for sample in tqdm(train_data, desc=f"Iter {iteration} — generating"):
        question = sample["question"]
        gold_raw = sample["answer"]
        gold = extract_answer(gold_raw) or gold_raw.strip()

        prompt = PROMPT_TEMPLATE.format(question=question)
        result = decoder.generate(prompt)

        pred = extract_answer(result["text"])
        total += 1

        if answers_match(pred, gold):
            correct_count += 1
            # Store as clean training sample (no <backtrack>)
            # Strip any <backtrack> tokens from the output
            clean_text = result["text"].replace(BACKTRACK_TOKEN, "")
            full_text = f"{prompt}{clean_text}{tokenizer.eos_token}"
            correct_samples.append({"text": full_text})

    acc = correct_count / total if total > 0 else 0.0
    logger.info(
        "Iter %d: %d/%d correct (%.2f%%), %d training samples",
        iteration, correct_count, total, acc * 100, len(correct_samples),
    )

    return correct_samples


# ---------------------------------------------------------------------------
# Stage 2: Train on filtered correct paths
# ---------------------------------------------------------------------------


def train_on_correct(
    model_name_or_path: str,
    correct_samples: list[dict],
    tokenizer,
    output_dir: Path,
    args,
    iteration: int,
):
    """Standard SFT on filtered correct paths. No masking, no <backtrack>."""
    # Load fresh base model + apply LoRA for this iteration
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=dtype, use_cache=False,
    )
    model.resize_token_embeddings(len(tokenizer))

    if args.lora_rank > 0:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules="all-linear",
            bias="none",
        )
        model = get_peft_model(model, lora_config)

    dataset = Dataset.from_list(correct_samples)

    def tokenize(example):
        text = example["text"]
        enc = tokenizer(
            text, truncation=True, max_length=args.max_length, padding=False,
        )
        labels = list(enc["input_ids"])

        # Mask prompt
        prompt_end = text.find("###Response:\n")
        if prompt_end != -1:
            pre = text[: prompt_end + len("###Response:\n")]
            pre_ids = tokenizer(pre, add_special_tokens=False)["input_ids"]
            for i in range(min(len(pre_ids), len(labels))):
                labels[i] = -100

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

    num_devices = max(torch.cuda.device_count(), 1)
    per_device_bs = 2
    grad_accum = max(1, args.ei_batch_size // (per_device_bs * num_devices))

    run_name = f"self-backtrack-ei-iter{iteration}"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.ei_epochs,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.ei_lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=args.bf16,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        seed=args.seed,
        report_to="wandb",
        run_name=run_name,
        remove_unused_columns=False,
    )

    if "WANDB_PROJECT" not in os.environ:
        os.environ["WANDB_PROJECT"] = "n-mars"

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=8,
        padding=True,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Iter %d model saved to %s", iteration, output_dir)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_greedy(model, tokenizer, output_path: Path):
    """Quick greedy eval on GSM8K test set."""
    model.eval()
    device = next(model.parameters()).device

    test_ds = load_dataset("gsm8k", "main", split="test")
    correct = 0

    for sample in tqdm(test_ds, desc="Evaluating"):
        question = sample["question"]
        gold_raw = sample["answer"]
        gold = extract_answer(gold_raw) or gold_raw.strip()

        prompt = PROMPT_TEMPLATE.format(question=question)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        gen_ids = out[0][input_ids.shape[1]:]
        text = tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)
        pred = extract_answer(text)

        if answers_match(pred, gold):
            correct += 1

    n = len(test_ds)
    acc = correct / n if n > 0 else 0.0
    logger.info("Greedy eval: %d/%d = %.4f", correct, n, acc)

    results = {"accuracy": acc, "correct": correct, "total": n}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Self-Backtracking Stage 2: Expert Iteration"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to Stage 1 trained model",
    )
    parser.add_argument(
        "--base_model", type=str, default="meta-llama/Llama-3.2-1B",
        help="Base model for fresh LoRA in each EI iteration",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--num_iterations", type=int, default=3,
        help="Number of expert iteration rounds",
    )
    # Generation params
    parser.add_argument("--b", type=int, default=1)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    # EI training params
    parser.add_argument("--ei_epochs", type=int, default=3)
    parser.add_argument("--ei_batch_size", type=int, default=16)
    parser.add_argument("--ei_lr", type=float, default=1e-5)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load GSM8K training set
    train_ds = load_dataset("gsm8k", "main", split="train")
    logger.info("Loaded %d training samples", len(train_ds))

    current_model_path = args.model_path

    for iteration in range(1, args.num_iterations + 1):
        logger.info("=== Expert Iteration %d/%d ===", iteration, args.num_iterations)

        # Load current model
        model, tokenizer = load_model_and_tokenizer(current_model_path, bf16=args.bf16)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        backtrack_id = tokenizer.convert_tokens_to_ids(BACKTRACK_TOKEN)

        # Generate + filter
        correct_samples = generate_and_filter(
            model, tokenizer, train_ds, backtrack_id, args, iteration,
        )

        if len(correct_samples) == 0:
            logger.warning("No correct samples at iter %d. Stopping.", iteration)
            break

        # Save generation results
        gen_path = args.output_dir / f"generation_iter{iteration}.json"
        with open(gen_path, "w") as f:
            json.dump({
                "num_correct": len(correct_samples),
                "num_total": len(train_ds),
                "accuracy": len(correct_samples) / len(train_ds),
            }, f, indent=2)

        # Free the generation model
        del model
        torch.cuda.empty_cache()

        # Train on filtered correct paths
        iter_output = args.output_dir / f"model_iteration_{iteration}"
        train_on_correct(
            args.base_model, correct_samples, tokenizer,
            iter_output, args, iteration,
        )

        # Eval this iteration
        model, tokenizer = load_model_and_tokenizer(
            str(iter_output), bf16=args.bf16,
        )
        model = model.to(device)
        eval_path = args.output_dir / f"eval_iter{iteration}.json"
        evaluate_greedy(model, tokenizer, eval_path)

        del model
        torch.cuda.empty_cache()

        current_model_path = str(iter_output)

    logger.info("Expert iteration complete. Final model: %s", current_model_path)


if __name__ == "__main__":
    main()
