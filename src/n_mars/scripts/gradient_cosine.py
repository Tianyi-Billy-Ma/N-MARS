"""Gradient cosine similarity analysis for Proposition 2.1 validation.

Measures cosine similarity between error-token gradients and UNDO/correction
gradients during mSFT training to empirically validate the gradient
non-alignment assumption (Assumption B.2).

Uses the mtybilly/GSM8K-Random-All dataset. The dataset's <|BACKTRACK|> token
is mapped to Llama's reserved special token <|reserved_special_token_0|>
(id=128002), which is already in the vocabulary — no embedding resize needed.

Usage:
    python -m n_mars.scripts.gradient_cosine \
        --model_name_or_path meta-llama/Llama-3.2-1B \
        --dataset_config p1_n3 \
        --output_dir outputs/gradient_cosine \
        --num_steps 500 --log_interval 10 --batch_size 2 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

logger = logging.getLogger(__name__)

# Dataset uses <|BACKTRACK|> but Llama tokenizer doesn't recognize it.
# We map it to a pre-existing reserved special token (no resize needed).
SRC_BACKTRACK = "<|BACKTRACK|>"
DST_BACKTRACK = "<|reserved_special_token_0|>"
DATASET_ID = "mtybilly/GSM8K-Random-All"


def build_model_and_tokenizer(
    model_name: str, lora_rank: int = 0,
):
    """Load model. Use a reserved token for UNDO — no resize.

    If lora_rank > 0, apply LoRA. Otherwise, train full parameters.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backtrack_id = tokenizer.convert_tokens_to_ids(DST_BACKTRACK)
    assert backtrack_id != tokenizer.unk_token_id, (
        f"{DST_BACKTRACK} not found in tokenizer vocabulary"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, use_cache=False
    )

    if lora_rank > 0:
        from peft import LoraConfig, TaskType, get_peft_model
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules="all-linear",
            bias="none",
        )
        model = get_peft_model(model, lora_config)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "trainable params: %d / %d (%.2f%%)",
        trainable, total, 100 * trainable / total,
    )

    return model, tokenizer, backtrack_id


def preprocess_text(text: str) -> str:
    """Replace dataset's <|BACKTRACK|> with the reserved Llama token."""
    return text.replace(SRC_BACKTRACK, DST_BACKTRACK)


def tokenize_sample(tokenizer, sample: dict, max_length: int, backtrack_id: int):
    """Tokenize a backtrack_response with mSFT-style masking.

    Returns input_ids and labels where:
    - Prompt tokens -> label = -100
    - Error tokens (before BACKTRACK runs) -> label = -100
    - BACKTRACK tokens -> label = token_id (supervised)
    - Correction tokens (after BACKTRACK runs) -> label = token_id (supervised)
    """
    query = sample["query"]
    response = preprocess_text(sample["backtrack_response"])

    text = f"Question: {query}\nAnswer:\n{response}{tokenizer.eos_token}"

    encoding = tokenizer(
        text, truncation=True, max_length=max_length, return_tensors="pt"
    )
    input_ids = encoding["input_ids"].squeeze(0)
    labels = input_ids.clone()

    # 1. Mask prompt tokens
    prompt_text = f"Question: {query}\nAnswer:\n"
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    n_prompt = min(len(prompt_ids), len(labels))
    labels[:n_prompt] = -100

    # 2. Mask error tokens (tokens before each BACKTRACK run)
    # Structure: ... [error_1..error_k] [BT x k] [correction] ...
    bk_positions = (input_ids == backtrack_id).nonzero(as_tuple=True)[0]

    if len(bk_positions) > 0:
        # Group into contiguous runs
        runs = []
        run_start = bk_positions[0].item()
        run_end = run_start
        for pos in bk_positions[1:]:
            if pos.item() == run_end + 1:
                run_end = pos.item()
            else:
                runs.append((run_start, run_end))
                run_start = pos.item()
                run_end = run_start
        runs.append((run_start, run_end))

        for run_start, run_end in runs:
            n_bk = run_end - run_start + 1
            error_start = max(n_prompt, run_start - n_bk)
            for i in range(error_start, run_start):
                labels[i] = -100

    return input_ids, labels


def build_masks(
    input_ids: torch.Tensor, labels: torch.Tensor, backtrack_id: int
):
    """Build three boolean masks for gradient decomposition.

    Returns:
        mask_e:  error token positions (label=-100, after prompt, before BT)
        mask_bk: BACKTRACK token positions
        mask_c:  correction token positions (supervised, after BT runs)
    """
    seq_len = labels.shape[0]
    device = labels.device

    # Find prompt end (first non -100 label)
    prompt_end = 0
    for i in range(seq_len):
        if labels[i] != -100:
            prompt_end = i
            break

    is_bk = input_ids == backtrack_id
    mask_e = torch.zeros(seq_len, dtype=torch.bool, device=device)
    mask_bk = is_bk.clone()
    mask_c = torch.zeros(seq_len, dtype=torch.bool, device=device)

    # State machine: walk post-prompt tokens
    seen_backtrack = False
    for i in range(prompt_end, seq_len):
        if is_bk[i]:
            seen_backtrack = True
        elif labels[i] == -100:
            # Masked post-prompt = error token
            mask_e[i] = True
        elif seen_backtrack:
            # Supervised token after a backtrack run = correction
            mask_c[i] = True
            # Don't reset seen_backtrack — all subsequent supervised
            # tokens after any BT are corrections

    return mask_e, mask_bk, mask_c


def compute_component_loss(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
):
    """Cross-entropy loss at masked positions only. Returns None if empty."""
    # For error positions, labels are -100 — we need to use input_ids
    # as targets instead. But we actually compute loss against input_ids
    # shifted, so we need the actual token ids.
    # We handle this by passing the original input_ids as labels for
    # error positions in the caller.
    if mask.sum() == 0:
        return None

    shift_logits = logits[:-1, :]
    shift_labels = labels[1:]
    shift_mask = mask[1:]

    if shift_mask.sum() == 0:
        return None

    # For positions where label is -100, cross_entropy would ignore them.
    # We need to compute loss even at error positions for gradient analysis.
    # Use the actual token ids (not -100) for loss computation.
    loss_per_token = F.cross_entropy(
        shift_logits, shift_labels, reduction="none", ignore_index=-100
    )
    valid = shift_mask & (shift_labels != -100)
    if valid.sum() == 0:
        return None
    return (loss_per_token * valid.float()).sum() / valid.float().sum()


def compute_error_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    mask_e: torch.Tensor,
):
    """Compute loss at error positions using actual token ids as targets.

    Error positions have label=-100 in mSFT, but we need their gradients
    for the cosine analysis, so we compute loss against the real token ids.
    """
    if mask_e.sum() == 0:
        return None

    shift_logits = logits[:-1, :]
    shift_targets = input_ids[1:]  # actual token ids, not masked labels
    shift_mask = mask_e[1:]

    if shift_mask.sum() == 0:
        return None

    loss_per_token = F.cross_entropy(
        shift_logits, shift_targets, reduction="none"
    )
    return (
        (loss_per_token * shift_mask.float()).sum()
        / shift_mask.float().sum()
    )


def get_gradient_vector(model, loss):
    """Gradient of loss w.r.t. trainable params, as a flat vector."""
    if loss is None:
        return None
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(
        loss, params, retain_graph=True, allow_unused=True
    )
    parts = []
    for g in grads:
        if g is not None:
            parts.append(g.detach().view(-1))
    if not parts:
        return None
    return torch.cat(parts)


def cosine_sim(v1, v2):
    """Cosine similarity between two flat vectors."""
    if v1 is None or v2 is None:
        return float("nan")
    if v1.norm() == 0 or v2.norm() == 0:
        return 0.0
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


def main():
    parser = argparse.ArgumentParser(
        description="Gradient cosine similarity for Prop 2.1"
    )
    parser.add_argument(
        "--model_name_or_path", type=str,
        default="meta-llama/Llama-3.2-1B",
    )
    parser.add_argument(
        "--dataset_config", type=str, default="p0.1_n10",
        choices=["p0.1_n10", "p1_n1", "p1_n3"],
    )
    parser.add_argument(
        "--output_dir", type=Path,
        default=Path("outputs/gradient_cosine"),
    )
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument(
        "--log_interval", type=int, default=10,
        help="Compute gradient cosines every N training steps",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--lora_rank", type=int, default=0,
                        help="LoRA rank. 0 = full parameter training")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, backtrack_id = build_model_and_tokenizer(
        args.model_name_or_path, args.lora_rank
    )
    model = model.to(device)

    logger.info(
        "Loading dataset %s config=%s", DATASET_ID, args.dataset_config
    )
    ds = load_dataset(DATASET_ID, args.dataset_config, split="train")
    ds = ds.filter(lambda x: SRC_BACKTRACK in x["backtrack_response"])
    logger.info("Filtered to %d backtrack samples", len(ds))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=0.01,
    )

    results = {
        "steps": [], "cos_e_bk": [], "cos_e_c": [],
        "loss_total": [], "loss_e": [], "loss_bk": [], "loss_c": [],
        "config": {k: str(v) for k, v in vars(args).items()},
    }

    model.train()
    data_idx = 0

    for step in range(1, args.num_steps + 1):
        batch_cos_e_bk = []
        batch_cos_e_c = []
        batch_loss_e = []
        batch_loss_bk = []
        batch_loss_c = []
        total_loss = torch.tensor(0.0, device=device)
        valid_samples = 0

        for _ in range(args.batch_size):
            sample = ds[data_idx % len(ds)]
            data_idx += 1

            input_ids, labels = tokenize_sample(
                tokenizer, sample, args.max_length, backtrack_id
            )
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids=input_ids.unsqueeze(0)).logits.squeeze(0)
            mask_e, mask_bk, mask_c = build_masks(
                input_ids, labels, backtrack_id
            )

            # Error loss uses real token ids (not -100 labels)
            loss_e = compute_error_loss(logits, input_ids, mask_e)
            # BK and correction losses use labels directly
            loss_bk = compute_component_loss(logits, labels, mask_bk)
            loss_c = compute_component_loss(logits, labels, mask_c)

            # mSFT training loss (no error gradient)
            msft_loss = torch.tensor(0.0, device=device)
            if loss_bk is not None:
                msft_loss = msft_loss + loss_bk
            if loss_c is not None:
                msft_loss = msft_loss + loss_c
            if msft_loss.item() > 0:
                total_loss = total_loss + msft_loss
                valid_samples += 1

            # Gradient cosine measurement at log intervals
            if step % args.log_interval == 0 and loss_e is not None:
                g_e = get_gradient_vector(model, loss_e)
                if loss_bk is not None:
                    g_bk = get_gradient_vector(model, loss_bk)
                    batch_cos_e_bk.append(cosine_sim(g_e, g_bk))
                if loss_c is not None:
                    g_c = get_gradient_vector(model, loss_c)
                    batch_cos_e_c.append(cosine_sim(g_e, g_c))
                if loss_e is not None:
                    batch_loss_e.append(loss_e.item())
                if loss_bk is not None:
                    batch_loss_bk.append(loss_bk.item())
                if loss_c is not None:
                    batch_loss_c.append(loss_c.item())

        # Backward + update (mSFT loss only)
        if valid_samples > 0:
            avg_loss = total_loss / valid_samples
            optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Log
        if step % args.log_interval == 0:
            m_ebk = (
                sum(batch_cos_e_bk) / len(batch_cos_e_bk)
                if batch_cos_e_bk else float("nan")
            )
            m_ec = (
                sum(batch_cos_e_c) / len(batch_cos_e_c)
                if batch_cos_e_c else float("nan")
            )
            cur_loss = (
                avg_loss.item() if valid_samples > 0 else float("nan")
            )

            results["steps"].append(step)
            results["cos_e_bk"].append(m_ebk)
            results["cos_e_c"].append(m_ec)
            results["loss_total"].append(cur_loss)
            results["loss_e"].append(
                sum(batch_loss_e) / len(batch_loss_e)
                if batch_loss_e else float("nan")
            )
            results["loss_bk"].append(
                sum(batch_loss_bk) / len(batch_loss_bk)
                if batch_loss_bk else float("nan")
            )
            results["loss_c"].append(
                sum(batch_loss_c) / len(batch_loss_c)
                if batch_loss_c else float("nan")
            )

            logger.info(
                "Step %d | loss=%.4f | cos(g_e,g_bk)=%.4f "
                "| cos(g_e,g_c)=%.4f | mask_e=%d mask_bk=%d mask_c=%d",
                step, cur_loss, m_ebk, m_ec,
                mask_e.sum().item(), mask_bk.sum().item(),
                mask_c.sum().item(),
            )

    # Save
    out_file = args.output_dir / "gradient_cosine_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_file)

    # Summary
    valid_ebk = [v for v in results["cos_e_bk"] if v == v]
    valid_ec = [v for v in results["cos_e_c"] if v == v]

    print("\n=== Gradient Cosine Similarity Summary ===")
    if valid_ebk:
        m = sum(valid_ebk) / len(valid_ebk)
        print(f"cos(g_error, g_undo):       mean={m:.4f}  (n={len(valid_ebk)})")
        v = "NEGATIVE (holds)" if m < 0 else "POSITIVE (may not hold)"
        print(f"  {v}")
    if valid_ec:
        m = sum(valid_ec) / len(valid_ec)
        print(f"cos(g_error, g_correction): mean={m:.4f}  (n={len(valid_ec)})")
        v = "NEGATIVE (holds)" if m < 0 else "POSITIVE (may not hold)"
        print(f"  {v}")


if __name__ == "__main__":
    main()
