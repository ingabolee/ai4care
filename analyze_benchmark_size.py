"""
analyze_benchmark_size.py -- Bootstrap confidence interval analysis for benchmark adequacy.

Usage:
    python analyze_benchmark_size.py

Reads results/evaluation_results.csv and calculates bootstrap CI width for
primary metrics as a function of sample size.

Output: results/benchmark_size_analysis.csv
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

RESULTS_CSV = Path("results/evaluation_results.csv")
OUTPUT_CSV = Path("results/benchmark_size_analysis.csv")

METRICS = [
    "schema_completeness",
    "reference_fact_recall",
    "reference_fact_f1",
    "unsupported_fact_rate",
    "gap_targeting_rate",
    "redundancy_rate",
]

N_BOOTSTRAP = 1000
CONFIDENCE = 0.95
SAMPLE_SIZES = [10, 25, 50, 100, 200, 300, 500]


def bootstrap_ci(data: list[float], n_boot: int = N_BOOTSTRAP, conf: float = CONFIDENCE, seed: int = 42) -> tuple[float, float, float, float]:
    """Return (mean, lower_ci, upper_ci, ci_width) using bootstrap."""
    if not data:
        return 0.0, 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [rng.choice(data) for _ in range(len(data))]
        means.append(sum(sample) / len(sample))
    means.sort()
    alpha = (1 - conf) / 2
    lo_idx = int(alpha * n_boot)
    hi_idx = int((1 - alpha) * n_boot)
    mean = sum(data) / len(data)
    lower = means[lo_idx]
    upper = means[min(hi_idx, n_boot - 1)]
    return round(mean, 4), round(lower, 4), round(upper, 4), round(upper - lower, 4)


def load_results() -> list[dict]:
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Results not found at {RESULTS_CSV}. Run evaluate.py first.")
    rows = []
    with open(RESULTS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def main():
    rows = load_results()
    print(f"Loaded {len(rows)} result rows.")

    output_rows = []
    rng = random.Random(42)

    for metric in METRICS:
        values = []
        for row in rows:
            try:
                v = float(row[metric])
                values.append(v)
            except (KeyError, ValueError):
                pass

        if not values:
            print(f"  [SKIP] {metric}: no data")
            continue

        for n in SAMPLE_SIZES:
            if n > len(values):
                n_sample = len(values)
            else:
                n_sample = n

            # Subsample n times and compute CI at each subsample size
            mean, lower, upper, width = bootstrap_ci(
                rng.sample(values, n_sample),
                seed=rng.randint(0, 10000),
            )
            output_rows.append({
                "sample_size": n_sample,
                "metric": metric,
                "mean": mean,
                "lower_ci": lower,
                "upper_ci": upper,
                "ci_width": width,
            })
            print(f"  {metric} n={n_sample}: mean={mean:.4f}, CI=[{lower:.4f}, {upper:.4f}], width={width:.4f}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if output_rows:
        cols = list(output_rows[0].keys())
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(output_rows)
        print(f"\nBenchmark size analysis saved → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
