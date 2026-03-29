"""Analyze lm_eval log_samples output for compute budget comparison.

Reads lm_eval logged samples and computes per-sample and aggregate statistics:
- Average / median / max tokens generated
- Total tokens generated
- Estimated FLOPs (tokens × model_params × 2)
- Wall-clock time (if available)

Usage:
    python -m n_mars.scripts.compute_budget \
        --results_dir outputs/llama3.1-8b-gsm8k-ar-256tok \
        --model_params 8e9

    # Compare multiple runs
    python -m n_mars.scripts.compute_budget \
        --results_dir outputs/llama3.1-8b-gsm8k-ar-256tok \
                      outputs/llama3.1-8b-gsm8k-nmars-256tok \
        --model_params 8e9
"""

import argparse
import json
import statistics
from pathlib import Path


def load_samples(results_dir: Path) -> list[dict]:
    """Load logged samples from lm_eval output directory."""
    samples = []
    for f in sorted(results_dir.rglob("samples_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                samples.append(json.loads(line))
    return samples


def count_tokens_from_sample(sample: dict) -> int:
    """Extract generated token count from a logged sample.

    Approximates token count via whitespace splitting.
    For exact counts, use the model's tokenizer.
    """
    resps = sample.get("resps", [])
    if not resps:
        return 0
    # lm_eval stores resps as list of lists; each inner list contains
    # either a string or a tuple (string, is_greedy)
    first_resp = resps[0][0] if isinstance(resps[0], list) else resps[0]
    if isinstance(first_resp, (list, tuple)):
        first_resp = first_resp[0]
    if not isinstance(first_resp, str):
        return 0
    return len(first_resp.split())


def analyze_run(results_dir: Path, model_params: float | None = None) -> dict:
    """Analyze a single evaluation run."""
    samples = load_samples(results_dir)
    if not samples:
        print(f"  No samples found in {results_dir}")
        return {}

    token_counts = [count_tokens_from_sample(s) for s in samples]

    stats = {
        "run": results_dir.name,
        "num_samples": len(samples),
        "avg_tokens": statistics.mean(token_counts),
        "median_tokens": statistics.median(token_counts),
        "max_tokens": max(token_counts),
        "min_tokens": min(token_counts),
        "total_tokens": sum(token_counts),
    }

    if model_params:
        # Rough FLOPs estimate: 2 * params * tokens (forward pass only)
        stats["est_flops"] = 2 * model_params * stats["total_tokens"]
        stats["est_flops_per_sample"] = 2 * model_params * stats["avg_tokens"]

    # Try to load results.json for wall-clock time and accuracy
    for f in results_dir.rglob("results.json"):
        with open(f) as fh:
            results = json.loads(fh.read())
        if "total_evaluation_time_seconds" in results:
            stats["wall_clock_s"] = results["total_evaluation_time_seconds"]
        if "results" in results:
            for task_name, task_results in results["results"].items():
                for metric, value in task_results.items():
                    if "acc" in metric or "exact_match" in metric:
                        stats[f"{task_name}/{metric}"] = value
        break

    return stats


def format_flops(flops: float) -> str:
    """Format FLOPs in human-readable form."""
    if flops >= 1e18:
        return f"{flops / 1e18:.2f} EFLOPs"
    elif flops >= 1e15:
        return f"{flops / 1e15:.2f} PFLOPs"
    elif flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    return f"{flops:.2f} FLOPs"


def print_report(all_stats: list[dict]) -> None:
    """Print a comparison table."""
    if not all_stats:
        print("No results to display.")
        return

    # Header
    has_flops = any("est_flops" in s for s in all_stats)
    # Collect accuracy keys across all runs
    accuracy_keys = sorted({k for s in all_stats for k in s if "/" in k})

    print("\n" + "=" * 100)
    print("COMPUTE BUDGET ANALYSIS")
    print("=" * 100)

    has_time = any("wall_clock_s" in s for s in all_stats)

    header = f"{'Run':<40} {'Samples':>8} {'Avg Tok':>8} {'Med Tok':>8} {'Max Tok':>8} {'Total Tok':>10}"
    if has_time:
        header += f" {'Wall-clock':>12}"
    if has_flops:
        header += f" {'Est FLOPs':>14}"
    for key in accuracy_keys:
        header += f" {key:>20}"
    print(header)
    print("-" * len(header))

    for stats in all_stats:
        row = (
            f"{stats['run']:<40} "
            f"{stats['num_samples']:>8} "
            f"{stats['avg_tokens']:>8.1f} "
            f"{stats['median_tokens']:>8.1f} "
            f"{stats['max_tokens']:>8} "
            f"{stats['total_tokens']:>10}"
        )
        if has_time:
            secs = stats.get("wall_clock_s")
            if secs is not None:
                mins, sec = divmod(int(secs), 60)
                row += f" {mins:>5}m{sec:02d}s"
            else:
                row += f" {'N/A':>12}"
        if has_flops:
            row += f" {format_flops(stats['est_flops']):>14}"
        for key in accuracy_keys:
            val = stats.get(key, "N/A")
            if isinstance(val, float):
                row += f" {val:>20.4f}"
            else:
                row += f" {str(val):>20}"
        print(row)

    print("=" * len(header))
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze lm_eval compute budget")
    parser.add_argument(
        "--results_dir",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to lm_eval output directories",
    )
    parser.add_argument(
        "--model_params",
        type=float,
        default=None,
        help="Number of model parameters (e.g., 8e9 for 8B) for FLOPs estimation",
    )
    args = parser.parse_args()

    all_stats = []
    for results_dir in args.results_dir:
        print(f"Analyzing: {results_dir}")
        stats = analyze_run(results_dir, args.model_params)
        if stats:
            all_stats.append(stats)

    print_report(all_stats)


if __name__ == "__main__":
    main()
