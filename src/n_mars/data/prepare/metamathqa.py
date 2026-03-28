"""Prepare and push MetaMathQA dataset subsets to HuggingFace Hub.

Creates three subsets from meta-math/MetaMathQA:
  - full:     All 395K samples (unchanged)
  - MATH:     Only MATH_* types (~155K)
  - MATH-50K: Stratified sample of 50K from MATH subset

All subsets have a single 'train' split and include an 'id' column
in the format <type>_<global_idx> (global index shared across all rows).

Usage:
    python -m n_mars.data.prepare.metamathqa \
        --repo_id mtybilly/MetaMathQA \
        --seed 42
"""

from __future__ import annotations

import argparse
import logging
import random
from collections import Counter

from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)

MATH_TYPES = {"MATH_AnsAug", "MATH_Rephrased", "MATH_FOBAR", "MATH_SV"}
SOURCE_DATASET = "meta-math/MetaMathQA"
MATH_50K_SIZE = 50_000


def add_global_ids(ds: Dataset) -> Dataset:
    """Add 'id' column and reorder: id, query, response, then rest."""
    ids = [f"{row['type']}_{i}" for i, row in enumerate(ds)]
    ds = ds.add_column("id", ids)
    # Reorder columns: id, query, response first
    cols = ds.column_names
    ordered = ["id", "query", "response"]
    ordered += [c for c in cols if c not in ordered]
    return ds.select_columns(ordered)


def stratified_sample(
    ds: Dataset, n: int, seed: int,
) -> Dataset:
    """Stratified sample preserving type proportions.

    Samples n rows total, with each type contributing proportionally
    to its share in the original dataset.
    """
    rng = random.Random(seed)
    type_counts = Counter(ds["type"])
    total = len(ds)

    # Compute per-type sample sizes (proportional)
    type_n = {}
    allocated = 0
    sorted_types = sorted(type_counts.keys())
    for i, t in enumerate(sorted_types):
        if i == len(sorted_types) - 1:
            # Last type gets the remainder to ensure exactly n total
            type_n[t] = n - allocated
        else:
            count = round(n * type_counts[t] / total)
            type_n[t] = count
            allocated += count

    logger.info("Stratified sampling: %s", type_n)

    # Group indices by type
    type_indices: dict[str, list[int]] = {t: [] for t in sorted_types}
    for idx, t in enumerate(ds["type"]):
        type_indices[t].append(idx)

    # Sample from each type
    selected = []
    for t in sorted_types:
        pool = type_indices[t]
        k = min(type_n[t], len(pool))
        selected.extend(rng.sample(pool, k))

    selected.sort()
    return ds.select(selected)


def _push_dataset_card(
    repo_id: str,
    full_ds: Dataset,
    math_ds: Dataset,
    math_50k_ds: Dataset,
) -> None:
    """Create and push a dataset card (README.md) to the HuggingFace repo."""
    from huggingface_hub import HfApi

    math_50k_counts = Counter(math_50k_ds["type"])

    card = f"""\
---
license: mit
task_categories:
  - text-generation
  - question-answering
language:
  - en
tags:
  - math
  - reasoning
  - metamath
source_datasets:
  - meta-math/MetaMathQA
configs:
  - config_name: full
    data_files:
      - split: train
        path: full/train-*.parquet
  - config_name: MATH
    data_files:
      - split: train
        path: MATH/train-*.parquet
  - config_name: MATH-50K
    data_files:
      - split: train
        path: MATH-50K/train-*.parquet
default_config_name: full
---

# MetaMathQA Subsets

Curated subsets of [meta-math/MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) \
for mathematical reasoning experiments.

## Subsets

| Subset | Samples | Description |
|--------|---------|-------------|
| `full` | {len(full_ds):,} | All MetaMathQA samples (unchanged) |
| `MATH` | {len(math_ds):,} | MATH_* types only (AnsAug, Rephrased, FOBAR, SV) |
| `MATH-50K` | {len(math_50k_ds):,} | Stratified 50K sample from MATH subset |

### MATH-50K Type Distribution

| Type | Count | Proportion |
|------|-------|------------|
| MATH_AnsAug | {math_50k_counts.get("MATH_AnsAug", 0):,} | \
{math_50k_counts.get("MATH_AnsAug", 0) / len(math_50k_ds) * 100:.1f}% |
| MATH_Rephrased | {math_50k_counts.get("MATH_Rephrased", 0):,} | \
{math_50k_counts.get("MATH_Rephrased", 0) / len(math_50k_ds) * 100:.1f}% |
| MATH_FOBAR | {math_50k_counts.get("MATH_FOBAR", 0):,} | \
{math_50k_counts.get("MATH_FOBAR", 0) / len(math_50k_ds) * 100:.1f}% |
| MATH_SV | {math_50k_counts.get("MATH_SV", 0):,} | \
{math_50k_counts.get("MATH_SV", 0) / len(math_50k_ds) * 100:.1f}% |

## Columns

| Column | Description |
|--------|-------------|
| `id` | Unique identifier in format `<type>_<global_idx>` |
| `type` | Question type (e.g., MATH_AnsAug, GSM_Rephrased) |
| `query` | The question text |
| `original_question` | Original question from the source dataset |
| `response` | Chain-of-thought solution ending with "The answer is: ..." |

## Usage

```python
from datasets import load_dataset

# Load the MATH-50K subset
ds = load_dataset("{repo_id}", "MATH-50K", split="train")

# Load full dataset
ds = load_dataset("{repo_id}", "full", split="train")
```

## Source

Derived from [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) \
(Yu et al., 2024). Stratified sampling uses seed=42 for reproducibility.
"""

    api = HfApi()
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    logger.info("Dataset card pushed to %s", repo_id)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and push MetaMathQA subsets to HuggingFace"
    )
    parser.add_argument(
        "--repo_id", type=str, default="mtybilly/MetaMathQA",
        help="HuggingFace repo ID to push to",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Build datasets but don't push to HuggingFace",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    # ----- Load source dataset -----
    logger.info("Loading %s", SOURCE_DATASET)
    raw = load_dataset(SOURCE_DATASET, split="train")
    logger.info("Loaded %d samples", len(raw))

    # ----- Subset: full -----
    full_ds = add_global_ids(raw)
    logger.info("full: %d samples", len(full_ds))

    # ----- Subset: MATH -----
    math_ds = raw.filter(lambda x: x["type"] in MATH_TYPES)
    math_ds = add_global_ids(math_ds)
    logger.info("MATH: %d samples", len(math_ds))

    # ----- Subset: MATH-50K -----
    math_50k_ds = stratified_sample(math_ds, MATH_50K_SIZE, args.seed)
    # Re-assign IDs to preserve original IDs from the MATH subset
    # (stratified_sample preserves the rows, so 'id' column is kept)
    logger.info("MATH-50K: %d samples", len(math_50k_ds))

    # Log type distributions
    for name, ds in [
        ("full", full_ds), ("MATH", math_ds), ("MATH-50K", math_50k_ds),
    ]:
        counts = Counter(ds["type"])
        logger.info(
            "%s type distribution: %s",
            name,
            dict(sorted(counts.items())),
        )

    if args.dry_run:
        logger.info("Dry run — not pushing to HuggingFace")
        return

    # ----- Push to HuggingFace -----
    logger.info("Pushing to %s", args.repo_id)

    # Push each subset as a separate config
    for config_name, ds in [
        ("full", full_ds),
        ("MATH", math_ds),
        ("MATH-50K", math_50k_ds),
    ]:
        logger.info("Pushing config '%s' (%d rows)", config_name, len(ds))
        ds_dict = DatasetDict({"train": ds})
        ds_dict.push_to_hub(
            args.repo_id,
            config_name=config_name,
            private=False,
        )

    # ----- Push dataset card -----
    _push_dataset_card(args.repo_id, full_ds, math_ds, math_50k_ds)

    logger.info("Done. Dataset at https://huggingface.co/datasets/%s", args.repo_id)


if __name__ == "__main__":
    main()
