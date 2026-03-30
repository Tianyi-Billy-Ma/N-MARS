"""GRPO training with multi-objective reward for N-MARS.

Implements Section 3.3: Group Relative Policy Optimization with R_inc + R_pen.

For each prompt x, G trajectories are sampled from the current policy,
post-processed with stack_postprocess, and rewarded with the multi-objective
reward. Group-normalized advantages drive the policy update.

Usage:
    python -m n_mars.train.grpo \\
        --model_path outputs/nmars-llama3.2-1b-gsm8k-msft \\
        --data_path data/nmars/gsm8k/augmented \\
        --output_dir outputs/nmars-llama3.2-1b-gsm8k-grpo \\
        --kappa 0.2 --num_generations 8 \\
        --num_epochs 1 --batch_size 4 --learning_rate 1e-5 \\
        --seed 42 --bf16
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)

from n_mars.train.reward import compute_total_reward
from n_mars.train.token_init import UNDO_TOKEN

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_grpo_dataset(data_path: str | Path) -> Dataset:
    """Load D_aug and return a Dataset with 'prompt' and 'answer' columns."""
    dataset = load_from_disk(str(data_path))
    logger.info("Loaded dataset: %s", dataset)

    if hasattr(dataset, "keys"):
        ds = dataset["train"] if "train" in dataset else dataset[list(dataset.keys())[0]]
    else:
        ds = dataset

    # Ensure required columns; adapt column names as needed
    if "prompt" not in ds.column_names and "text" in ds.column_names:
        # Split on response marker if only 'text' is present
        response_marker = "###Response:\n"

        def split_prompt(example: dict) -> dict:
            text: str = example["text"]
            idx = text.find(response_marker)
            if idx != -1:
                return {
                    "prompt": text[: idx + len(response_marker)],
                    "answer": example.get("answer", ""),
                }
            return {"prompt": text, "answer": example.get("answer", "")}

        ds = ds.map(split_prompt)

    return ds


# ---------------------------------------------------------------------------
# GRPO loop (trl-based or custom fallback)
# ---------------------------------------------------------------------------

def _try_trl_grpo(
    model,
    tokenizer,
    dataset: Dataset,
    args: argparse.Namespace,
    undo_token_id: int,
) -> bool:
    """Attempt to use trl.GRPOTrainer. Returns True on success."""
    try:
        from trl import GRPOConfig, GRPOTrainer  # type: ignore[import]
    except ImportError:
        logger.info("trl not available; falling back to custom GRPO loop")
        return False

    kappa = args.kappa

    def reward_fn(completions: list[str], prompts: list[str], **kwargs: Any) -> list[float]:  # noqa: ANN001
        rewards = []
        gold = kwargs.get("answer", [""] * len(completions))
        for i, (completion, prompt) in enumerate(zip(completions, prompts)):
            gold_answer = gold[i] if isinstance(gold, list) else gold

            token_ids = tokenizer.encode(completion, add_special_tokens=False)
            r = compute_total_reward(token_ids, gold_answer, undo_token_id, kappa, tokenizer)
            rewards.append(r)
        return rewards

    wandb_run_name = args.wandb_run_name or Path(args.output_dir).name

    grpo_config = GRPOConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.1,
        bf16=args.bf16,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=args.seed,
        report_to="wandb",
        run_name=wandb_run_name,
        num_generations=args.num_generations,
        # GRPO-specific
        epsilon=0.2,
        beta=0.01,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=reward_fn,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    logger.info("trl GRPO training complete. Model saved to %s", args.output_dir)
    return True


def _custom_grpo_loop(
    model,
    tokenizer,
    dataset: Dataset,
    args: argparse.Namespace,
    undo_token_id: int,
) -> None:
    """Custom GRPO training loop (used when trl is unavailable)."""
    import torch.nn.functional as F  # noqa: PLC0415

    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.1
    )

    # Simple warmup scheduler (linear)
    total_steps = len(dataset) // args.batch_size * args.num_epochs
    warmup_steps = int(0.1 * total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return max(0.0, 1.0 - (step - warmup_steps) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    global_step = 0
    kappa = args.kappa
    G = args.num_generations

    prompts = dataset["prompt"]
    answers: list[str] = (
        dataset["answer"] if "answer" in dataset.column_names else [""] * len(prompts)
    )

    for epoch in range(args.num_epochs):
        indices = torch.randperm(len(prompts)).tolist()
        for batch_start in range(0, len(indices), args.batch_size):
            batch_idx = indices[batch_start : batch_start + args.batch_size]
            batch_prompts = [prompts[i] for i in batch_idx]
            batch_answers = [answers[i] for i in batch_idx]

            all_losses = []
            for prompt, gold in zip(batch_prompts, batch_answers):
                prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

                # Sample G trajectories
                with torch.no_grad():
                    outputs = model.generate(
                        prompt_ids,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=1.0,
                        num_return_sequences=G,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                rewards = []
                for g in range(G):
                    response_ids = outputs[g][prompt_ids.shape[-1] :].tolist()
                    r = compute_total_reward(response_ids, gold, undo_token_id, kappa, tokenizer)
                    rewards.append(r)

                rewards_t = torch.tensor(rewards, dtype=torch.float32)
                mean_r = rewards_t.mean()
                std_r = rewards_t.std() + 1e-8
                advantages = (rewards_t - mean_r) / std_r  # shape: (G,)

                # REINFORCE policy gradient (simplified; for full GRPO with
                # clipped importance ratios, use the trl-based path above)
                for g in range(G):
                    response_ids = outputs[g][prompt_ids.shape[-1] :]
                    full_ids = outputs[g].unsqueeze(0)

                    logits = model(full_ids).logits
                    prompt_len = prompt_ids.shape[-1]
                    response_logits = logits[0, prompt_len - 1 : -1, :]
                    log_probs = F.log_softmax(response_logits, dim=-1)

                    token_log_probs = log_probs[
                        torch.arange(len(response_ids)), response_ids
                    ]
                    pg_loss = -(advantages[g] * token_log_probs.sum())
                    all_losses.append(pg_loss)

            if all_losses:
                loss = torch.stack(all_losses).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if global_step % 10 == 0:
                    logger.info(
                        "epoch=%d step=%d loss=%.4f", epoch + 1, global_step, loss.item()
                    )
                global_step += 1

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    logger.info("Custom GRPO training complete. Model saved to %s", args.output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="N-MARS GRPO with multi-objective reward (Section 3.3)"
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to mSFT checkpoint")
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--kappa", type=float, default=0.2, help="<UNDO> budget fraction")
    parser.add_argument("--num_generations", type=int, default=8, help="G: trajectories per prompt")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        use_cache=False,
    )

    undo_token_id = tokenizer.convert_tokens_to_ids(UNDO_TOKEN)
    if undo_token_id == tokenizer.unk_token_id:
        logger.warning(
            "<UNDO> token not found in tokenizer; it may not have been added during mSFT. "
            "Falling back to registering it now."
        )
        from n_mars.train.token_init import initialize_undo_token  # noqa: PLC0415
        model, tokenizer = initialize_undo_token(model, tokenizer, method="semantic")
        undo_token_id = tokenizer.convert_tokens_to_ids(UNDO_TOKEN)

    logger.info("<UNDO> token id: %d", undo_token_id)

    # --- Dataset ---
    dataset = load_grpo_dataset(args.data_path)

    # --- Train ---
    success = _try_trl_grpo(model, tokenizer, dataset, args, undo_token_id)
    if not success:
        _custom_grpo_loop(model, tokenizer, dataset, args, undo_token_id)


if __name__ == "__main__":
    main()
