"""Natural Language Self-Correction Baseline (EXP-5).

Data-matched baseline for the N-MARS ICML rebuttal (reviewer DLhi-W1).
Tests whether an explicit UNDO token is necessary, or whether natural-language
revision markers (e.g., "Wait, that is incorrect. Let me reconsider.") achieve
the same self-correction effect.

Two variants are trained on NL-augmented GSM8K traces:
  - NL-SFT:  standard SFT, loss on all tokens (errors become "natural" context).
  - NL-mSFT: masked SFT, error tokens masked to -100, loss only on NL marker
             + correction tokens.

Usage
-----
# Step 1: build data
python -m n_mars.scripts.nl_self_correction --stage build_data \\
    --output_dir data/nl_self_correction

# Step 2a: train NL-SFT
python -m n_mars.scripts.nl_self_correction --stage train \\
    --variant sft --data_dir data/nl_self_correction \\
    --output_dir outputs/nl-sft-llama3.2-1b-gsm8k

# Step 2b: train NL-mSFT
python -m n_mars.scripts.nl_self_correction --stage train \\
    --variant msft --data_dir data/nl_self_correction \\
    --output_dir outputs/nl-msft-llama3.2-1b-gsm8k

# Step 3: evaluate
python -m n_mars.scripts.nl_self_correction --stage evaluate \\
    --model_path outputs/nl-sft-llama3.2-1b-gsm8k \\
    --output_path outputs/nl-sft-llama3.2-1b-gsm8k/eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "meta-llama/Llama-3.2-1B"
BACKTRACK_TOKEN = "<|BACKTRACK|>"
NL_MARKER = "\nWait, that is incorrect. Let me reconsider.\n"

_GSM8K_ANSWER_PATTERNS = [
    re.compile(r"[Tt]he answer is\s*(\-?[\d,]+)"),
    re.compile(r"####\s*(\-?[\d,]+)"),
]


# ---------------------------------------------------------------------------
# Stage 1: build data
# ---------------------------------------------------------------------------


def _replace_backtrack_tokens(text: str) -> str:
    """Replace one or more consecutive BACKTRACK tokens with the NL marker."""
    # Replace a run of BACKTRACK tokens (possibly separated by nothing) with a
    # single NL marker so the output reads naturally.
    pattern = r"(?:" + re.escape(BACKTRACK_TOKEN) + r")+"
    return re.sub(pattern, NL_MARKER, text)


def build_data(output_dir: Path) -> None:
    """Load HF dataset, transform to NL traces, save sft and msft splits."""
    from datasets import Dataset, load_dataset  # type: ignore[import]

    logger.info("Loading mtybilly/GSM8K-Random-All (p0.1_n10, train split)...")
    ds = load_dataset("mtybilly/GSM8K-Random-All", "p0.1_n10", split="train")
    logger.info("Loaded %d samples.", len(ds))

    sft_records: list[dict] = []
    msft_records: list[dict] = []

    for sample in ds:
        query: str = sample["query"]
        backtrack_response: str = sample["backtrack_response"]
        backtrack_prefix: str = sample["backtrack_prefix"]

        # Build the NL-augmented response
        nl_response = _replace_backtrack_tokens(backtrack_response)

        # Full formatted text (prompt + response)
        prompt = f"Question: {query}\nAnswer:\n"
        full_text = prompt + nl_response

        # --- NL-SFT record: standard SFT, no masking ---
        sft_records.append(
            {
                "text": full_text,
                "prompt": prompt,
                "response": nl_response,
                "query": query,
                "backtrack_prefix": backtrack_prefix,
            }
        )

        # --- NL-mSFT record: same text, but we store the boundary info so
        # the tokenize function can mask error tokens later ---
        msft_records.append(
            {
                "text": full_text,
                "prompt": prompt,
                "response": nl_response,
                "query": query,
                # backtrack_prefix is the portion of the response that is
                # "wrong" (before the error tokens diverge from gold).  We
                # store both prompt+prefix so the masking function can compute
                # the token boundary precisely.
                "backtrack_prefix": backtrack_prefix,
                "error_end_text": prompt + backtrack_prefix,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    sft_ds = Dataset.from_list(sft_records)
    msft_ds = Dataset.from_list(msft_records)

    sft_path = output_dir / "nl_sft"
    msft_path = output_dir / "nl_msft"
    sft_ds.save_to_disk(str(sft_path))
    msft_ds.save_to_disk(str(msft_path))

    logger.info("NL-SFT dataset (%d samples) saved to %s", len(sft_ds), sft_path)
    logger.info("NL-mSFT dataset (%d samples) saved to %s", len(msft_ds), msft_path)


# ---------------------------------------------------------------------------
# Stage 2: train
# ---------------------------------------------------------------------------


def _build_tokenize_fn_sft(tokenizer, max_length: int):
    """Tokenize for NL-SFT: mask prompt tokens only, loss on full response."""

    def tokenize(example: dict) -> dict:
        text: str = example["text"]
        prompt: str = example["prompt"]

        encoding = tokenizer(text, truncation=True, max_length=max_length, padding=False)
        input_ids = encoding["input_ids"]
        labels = list(input_ids)

        # Mask prompt tokens
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        n_prompt = len(prompt_ids)
        for i in range(min(n_prompt, len(labels))):
            labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"],
            "labels": labels,
        }

    return tokenize


def _build_tokenize_fn_msft(tokenizer, max_length: int):
    """Tokenize for NL-mSFT: mask prompt + error tokens.

    Error tokens are those between the end of backtrack_prefix and the start
    of the NL marker inside the response.  The NL marker and everything after
    (the correction) receive loss.
    """

    def tokenize(example: dict) -> dict:
        text: str = example["text"]
        prompt: str = example["prompt"]
        error_end_text: str = example["error_end_text"]  # prompt + backtrack_prefix

        encoding = tokenizer(text, truncation=True, max_length=max_length, padding=False)
        input_ids = encoding["input_ids"]
        labels = list(input_ids)

        # 1. Mask prompt
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        n_prompt = len(prompt_ids)
        for i in range(min(n_prompt, len(labels))):
            labels[i] = -100

        # 2. Mask error tokens: from end of prompt up to (but not including)
        #    the NL marker.  We locate the NL marker boundary via the
        #    error_end_text token count.
        nl_marker_start_text = error_end_text  # everything before NL marker
        if nl_marker_start_text and nl_marker_start_text != prompt:
            marker_boundary_ids = tokenizer(nl_marker_start_text, add_special_tokens=False)[
                "input_ids"
            ]
            # mask from end of prompt to the marker boundary
            start_mask = n_prompt
            end_mask = len(marker_boundary_ids)
            for i in range(start_mask, min(end_mask, len(labels))):
                labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"],
            "labels": labels,
        }

    return tokenize


def train(
    variant: str,
    data_dir: Path,
    output_dir: Path,
    model_name_or_path: str,
    num_epochs: int,
    batch_size: int,
    per_device_train_batch_size: int,
    learning_rate: float,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    max_length: int,
    seed: int,
    bf16: bool,
    wandb_project: str,
    wandb_run_name: str | None,
) -> None:
    import os

    import torch
    from datasets import load_from_disk  # type: ignore[import]
    from peft import LoraConfig, TaskType, get_peft_model  # type: ignore[import]
    from transformers import (  # type: ignore[import]
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(seed)

    # --- Tokenizer and model ---
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        use_cache=False,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules="all-linear",
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Dataset ---
    split_name = "nl_sft" if variant == "sft" else "nl_msft"
    dataset_path = data_dir / split_name
    dataset = load_from_disk(str(dataset_path))
    logger.info("Loaded %s dataset: %s", split_name, dataset)

    if variant == "sft":
        tokenize_fn = _build_tokenize_fn_sft(tokenizer, max_length)
    else:
        tokenize_fn = _build_tokenize_fn_msft(tokenizer, max_length)

    tokenized = dataset.map(
        tokenize_fn,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    # --- Gradient accumulation ---
    num_devices = max(torch.cuda.device_count(), 1)
    grad_accum = max(1, batch_size // (per_device_train_batch_size * num_devices))
    logger.info(
        "num_devices=%d  per_device_bs=%d  grad_accum=%d  effective_bs=%d",
        num_devices,
        per_device_train_batch_size,
        grad_accum,
        per_device_train_batch_size * num_devices * grad_accum,
    )

    run_name = wandb_run_name or output_dir.name
    if "WANDB_PROJECT" not in os.environ:
        os.environ["WANDB_PROJECT"] = wandb_project

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=bf16,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=seed,
        report_to="wandb",
        run_name=run_name,
        remove_unused_columns=False,
    )

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
    logger.info("Training complete. Model saved to %s", output_dir)


# ---------------------------------------------------------------------------
# Stage 3: evaluate
# ---------------------------------------------------------------------------


def _extract_answer_gsm8k(text: str) -> str | None:
    for pat in _GSM8K_ANSWER_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).replace(",", "").strip()
    return None


def _normalize_answer(answer: str) -> str:
    return answer.replace(",", "").strip().lstrip("0") or "0"


def _answers_match(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return float(_normalize_answer(pred)) == float(_normalize_answer(gold))
    except ValueError:
        return _normalize_answer(pred) == _normalize_answer(gold)


def _load_model_and_tokenizer(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try LoRA adapter
    adapter_cfg = Path(model_path) / "adapter_config.json"
    if adapter_cfg.exists():
        try:
            from peft import PeftModel  # type: ignore[import]

            with open(adapter_cfg) as fh:
                cfg = json.load(fh)
            base_name = cfg.get("base_model_name_or_path", "")
            if base_name:
                logger.info("Loading base '%s' + LoRA adapter '%s'", base_name, model_path)
                base = AutoModelForCausalLM.from_pretrained(
                    base_name, torch_dtype=torch.float16, device_map="auto"
                )
                model = PeftModel.from_pretrained(base, model_path)
                model = model.merge_and_unload()
                return model, tokenizer
        except Exception as exc:
            logger.warning("LoRA load failed (%s), falling back to plain load.", exc)

    logger.info("Loading model from '%s'", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    return model, tokenizer


def evaluate(model_path: str, output_path: Path, max_new_tokens: int, seed: int) -> None:
    import torch
    from datasets import load_dataset  # type: ignore[import]
    from tqdm import tqdm  # type: ignore[import]

    torch.manual_seed(seed)

    model, tokenizer = _load_model_and_tokenizer(model_path)
    model.eval()

    logger.info("Loading GSM8K test set...")
    dataset = load_dataset("gsm8k", "main", split="test")

    results_per_sample: list[dict] = []
    correct = 0
    total_tokens = 0
    wall_start = time.time()

    device = next(model.parameters()).device

    for idx, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        question: str = sample["question"]
        gold_raw: str = sample["answer"]
        gold_answer = _extract_answer_gsm8k(gold_raw) or gold_raw.strip()

        prompt = f"Question: {question}\nAnswer:\n"
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
        generated_text = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)
        pred_answer = _extract_answer_gsm8k(generated_text)
        is_correct = _answers_match(pred_answer, gold_answer)

        if is_correct:
            correct += 1
        total_tokens += generated_ids.shape[0]

        results_per_sample.append(
            {
                "idx": idx,
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": pred_answer,
                "correct": is_correct,
                "generated_text": generated_text,
                "tokens_generated": generated_ids.shape[0],
            }
        )

    wall_elapsed = time.time() - wall_start
    n_samples = len(dataset)

    metrics = {
        "accuracy": correct / n_samples if n_samples > 0 else 0.0,
        "correct": correct,
        "total": n_samples,
        "avg_tokens_generated": total_tokens / n_samples if n_samples > 0 else 0.0,
        "total_wall_clock_seconds": wall_elapsed,
        "config": {
            "model_path": model_path,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
        },
    }

    output = {"metrics": metrics, "samples": results_per_sample}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(output, fh, indent=2)

    print("\n=== Results ===")
    print(f"Accuracy:             {metrics['accuracy']:.4f}  ({correct}/{n_samples})")
    print(f"Avg tokens generated: {metrics['avg_tokens_generated']:.1f}")
    print(f"Wall-clock time:      {wall_elapsed:.1f}s")
    print(f"Results saved to:     {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NL Self-Correction Baseline (EXP-5) — build data, train, or evaluate"
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=["build_data", "train", "evaluate"],
        help="Pipeline stage to run",
    )

    # build_data
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="(build_data) Directory to write NL-SFT and NL-mSFT datasets",
    )

    # train
    parser.add_argument(
        "--variant",
        choices=["sft", "msft"],
        default=None,
        help="(train) Which variant to train: 'sft' (no masking) or 'msft' (error-masked)",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="(train) Directory containing nl_sft / nl_msft HF datasets",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=MODEL_NAME,
        help=f"(train) Base model (default: {MODEL_NAME})",
    )
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Effective batch size (default: 16)",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Per-device batch size (default: 4)",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", help="Train in bfloat16")
    parser.add_argument("--wandb_project", type=str, default="n-mars")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    # evaluate
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="(evaluate) Path to trained model or LoRA adapter directory",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="(evaluate) Path to write JSON results",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="(evaluate) Max new tokens for greedy decoding (default: 512)",
    )

    return parser


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    parser = build_parser()
    args = parser.parse_args()

    if args.stage == "build_data":
        if args.output_dir is None:
            parser.error("--output_dir is required for --stage build_data")
        build_data(args.output_dir)

    elif args.stage == "train":
        if args.variant is None:
            parser.error("--variant is required for --stage train")
        if args.data_dir is None:
            parser.error("--data_dir is required for --stage train")
        if args.output_dir is None:
            parser.error("--output_dir is required for --stage train")
        train(
            variant=args.variant,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model_name_or_path=args.model_name_or_path,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            per_device_train_batch_size=args.per_device_train_batch_size,
            learning_rate=args.learning_rate,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            max_length=args.max_length,
            seed=args.seed,
            bf16=args.bf16,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
        )

    elif args.stage == "evaluate":
        if args.model_path is None:
            parser.error("--model_path is required for --stage evaluate")
        if args.output_path is None:
            parser.error("--output_path is required for --stage evaluate")
        evaluate(
            model_path=args.model_path,
            output_path=args.output_path,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
