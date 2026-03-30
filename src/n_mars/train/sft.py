"""Standard SFT training and evaluation pipeline.

Supports three modes:
  - train: fine-tune model on dataset
  - eval: evaluate model with lm_eval
  - full: train then eval

Usage:
    python -m n_mars.train.sft --config configs/train/sft_gsm8k_llama3.2-1b.yaml --mode full
    python -m n_mars.train.sft --config configs/train/sft_gsm8k_llama3.2-1b.yaml --mode train \\
        --model_name_or_path meta-llama/Llama-3.2-1B
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys

from datasets import load_dataset
from transformers import (
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from n_mars.hparams.sft_args import SFTArguments
from n_mars.models.loader import load_model_and_tokenizer, save_model

logger = logging.getLogger(__name__)


def prepare_dataset(tokenizer, dataset_name: str, max_length: int, seed: int = 42):
    """Load and tokenize a dataset for SFT."""
    if dataset_name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="train")
    else:
        ds = load_dataset(dataset_name, split="train")

    def tokenize(example):
        # Format: Question + Answer
        if "question" in example and "answer" in example:
            text = f"Question: {example['question']}\nAnswer: {example['answer']}"
        else:
            text = example.get("text", "")

        encoding = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        encoding["labels"] = list(encoding["input_ids"])
        return encoding

    tokenized = ds.map(
        tokenize,
        remove_columns=ds.column_names,
        desc="Tokenizing",
    )
    return tokenized


def run_train(args: SFTArguments):
    """Run SFT training."""
    set_seed(args.seed)

    if "WANDB_PROJECT" not in os.environ:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    model, tokenizer = load_model_and_tokenizer(
        args.model_name_or_path,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    dataset = prepare_dataset(tokenizer, args.dataset, args.max_length, args.seed)

    run_name = args.run_name or os.path.basename(args.output_dir)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        report_to=args.report_to,
        run_name=run_name,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        padding=True,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainer.train()
    save_model(model, tokenizer, args.output_dir, use_lora=args.use_lora)
    logger.info("SFT training complete. Model saved to %s", args.output_dir)


def run_eval(args: SFTArguments):
    """Run lm_eval evaluation via subprocess."""
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={args.output_dir}",
        "--tasks", args.eval_tasks,
        "--batch_size", args.eval_batch_size,
        "--output_path", f"{args.output_dir}/eval_{args.eval_tasks}",
        "--seed", str(args.seed),
        "--log_samples",
        "--confirm_run_unsafe_code",
    ]

    if args.wandb_project:
        run_name = args.run_name or os.path.basename(args.output_dir)
        cmd.extend([
            "--wandb_args",
            f"project={args.wandb_project},name={args.eval_tasks}-{run_name},group=lm_eval",
        ])

    if args.num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(args.num_fewshot)])

    if args.max_gen_toks is not None:
        cmd.extend(["--gen_kwargs", f"max_gen_toks={args.max_gen_toks}"])

    logger.info("Running eval: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=True)
    logger.info("Eval complete (exit code %d)", result.returncode)


def main():
    parser = argparse.ArgumentParser(description="N-MARS SFT Training & Evaluation")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--mode", type=str, default=None, choices=["train", "eval", "full"])
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--eval_tasks", type=str, default=None)
    parser.add_argument("--max_gen_toks", type=int, default=None)
    parser.add_argument("--num_fewshot", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_lora", action="store_true", default=None)
    parser.add_argument("--no_lora", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--bf16", action="store_true", default=None)
    cli_args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    # Load config from YAML if provided
    if cli_args.config:
        args = SFTArguments.from_yaml(cli_args.config)
    else:
        args = SFTArguments()

    # Override with CLI args
    cli_dict = {k: v for k, v in vars(cli_args).items() if k != "config" and k != "no_lora"}
    args.override_from_cli(cli_dict)

    if cli_args.no_lora:
        args.use_lora = False

    logger.info(
        "Mode: %s | Model: %s | Dataset: %s | Output: %s",
        args.mode,
        args.model_name_or_path,
        args.dataset,
        args.output_dir,
    )

    if args.mode in ("train", "full"):
        run_train(args)

    if args.mode in ("eval", "full"):
        run_eval(args)


if __name__ == "__main__":
    main()
