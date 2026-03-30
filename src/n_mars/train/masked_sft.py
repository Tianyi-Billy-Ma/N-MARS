"""Masked Supervised Fine-Tuning (mSFT) for N-MARS.

Implements Section 3.2: train on augmented sequences D_aug with error token
positions masked from the loss (labels = -100).

Usage:
    python -m n_mars.train.masked_sft \\
        --model_name_or_path meta-llama/Llama-3.2-1B \\
        --data_path data/nmars/gsm8k/augmented \\
        --output_dir outputs/nmars-llama3.2-1b-gsm8k-msft \\
        --token_init_method semantic \\
        --num_epochs 3 --batch_size 16 --learning_rate 1e-5 \\
        --max_length 2048 --seed 42 --bf16
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from n_mars.train.token_init import initialize_undo_token

logger = logging.getLogger(__name__)

RESPONSE_MARKER = "###Response:\n"


def prepare_msft_dataset(tokenizer, data_path: str | Path, max_length: int):
    """Load and tokenize the augmented dataset D_aug with masking applied.

    Masking rules:
    - Prompt (everything up to and including RESPONSE_MARKER) → labels = -100
    - Error token spans (M_t = 0 in the binary mask) → labels = -100
    - Matched tokens, <UNDO> tokens, correction tokens → kept in labels

    The augmented dataset is expected to have columns:
        - "text": the full sequence (prompt + augmented response)
        - "error_spans": list of (start_char, end_char) tuples for error spans
          OR a pre-computed "mask" column with per-token binary mask values.

    If neither "error_spans" nor "mask" columns are present, only prompt masking
    is applied (fallback to standard SFT).
    """
    dataset = load_from_disk(str(data_path))
    logger.info("Loaded dataset: %s", dataset)

    if hasattr(dataset, "keys"):
        train_dataset = dataset["train"] if "train" in dataset else dataset[list(dataset.keys())[0]]
    else:
        train_dataset = dataset

    has_error_spans = "error_spans" in train_dataset.column_names
    has_mask = "mask" in train_dataset.column_names
    has_undo_positions = "undo_positions" in train_dataset.column_names

    logger.info(
        "Dataset columns: %s | has_error_spans=%s has_mask=%s has_undo_positions=%s",
        train_dataset.column_names,
        has_error_spans,
        has_mask,
        has_undo_positions,
    )

    def tokenize(example: dict) -> dict:
        text: str = example["text"]

        encoding = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        input_ids = encoding["input_ids"]
        labels = list(input_ids)

        # --- 1. Mask the prompt ---
        response_start = text.find(RESPONSE_MARKER)
        if response_start != -1:
            pre_response_text = text[: response_start + len(RESPONSE_MARKER)]
            pre_ids = tokenizer(pre_response_text, add_special_tokens=False)["input_ids"]
            n_prompt = len(pre_ids)
            for i in range(min(n_prompt, len(labels))):
                labels[i] = -100
        else:
            # No response marker: mask nothing (pure response data)
            n_prompt = 0

        # --- 2. Mask error token spans (M_t = 0) ---
        if has_mask:
            # Per-token binary mask aligned with the full tokenized sequence.
            # The mask may be stored as a comma-joined string (from build_dataset)
            # or as a list of ints (native Arrow).
            raw_mask = example["mask"]
            if isinstance(raw_mask, str):
                token_mask = [int(x) for x in raw_mask.split(",")]
            else:
                token_mask = list(raw_mask)
            for i, m in enumerate(token_mask):
                if i < len(labels) and m == 0:
                    labels[i] = -100

        elif has_error_spans:
            # Character-level error spans; find corresponding token positions
            error_spans = example["error_spans"]  # list of [start, end]
            for span_start, span_end in error_spans:
                # Tokenize text up to span boundaries to get token offsets
                before_ids = tokenizer(text[:span_start], add_special_tokens=False)["input_ids"]
                through_ids = tokenizer(text[:span_end], add_special_tokens=False)["input_ids"]
                tok_start = len(before_ids)
                tok_end = len(through_ids)
                for i in range(tok_start, min(tok_end, len(labels))):
                    labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"],
            "labels": labels,
        }

    tokenized = train_dataset.map(
        tokenize,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing + masking",
    )
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="N-MARS Masked SFT (Section 3.2)")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--token_init_method",
        type=str,
        default="semantic",
        choices=["centroid", "context", "semantic"],
    )
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16, help="Effective batch size")
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Per-device batch size; gradient_accumulation computed automatically",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="n-mars")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    set_seed(args.seed)

    if "WANDB_PROJECT" not in os.environ:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    # --- Model and tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        use_cache=False,  # required for gradient checkpointing
    )

    model, tokenizer = initialize_undo_token(model, tokenizer, method=args.token_init_method)

    # --- Dataset ---
    tokenized = prepare_msft_dataset(tokenizer, args.data_path, args.max_length)

    # --- Gradient accumulation ---
    num_devices = max(torch.cuda.device_count(), 1)
    per_device_bs = args.per_device_train_batch_size
    grad_accum = max(1, args.batch_size // (per_device_bs * num_devices))
    logger.info(
        "num_devices=%d  per_device_bs=%d  grad_accum=%d  effective_bs=%d",
        num_devices,
        per_device_bs,
        grad_accum,
        per_device_bs * num_devices * grad_accum,
    )

    wandb_run_name = args.wandb_run_name or Path(args.output_dir).name

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.1,
        max_grad_norm=1.0,
        bf16=args.bf16,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=args.seed,
        report_to="wandb",
        run_name=wandb_run_name,
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
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    logger.info("mSFT complete. Model saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
