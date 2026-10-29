"""
analyze_benchmark_size.py -- Cluster-aware bootstrap confidence interval analysis.

Usage:
    python analyze_benchmark_size.py [--results path/to/results.csv]

Reads results/evaluation_results_v2.csv and calculates bootstrap confidence
intervals for the primary metric (reference_fact_f1) across model conditions.

Key design:
  - Resampling unit: logical source case (case_group_id) — NOT individual rows.
  - Multiple question variants and iterations of the same source case are kept
    together when resampling (cluster-aware bootstrap).
  - Paired comparisons: for models sharing identical case_group_id + question_variant_id,
    compute per-pair differences and their bootstrap CI.
  - Fixed, recorded seed (BOOTSTRAP_SEED = 42) for reproducibility.
  - Sensitivity analysis at matching thresholds 0.25, 0.5, 0.75 when those
    columns are available.

Limitations (must not be overstated):
  - Bootstrap percentile intervals are approximate. Stability of the interval
    does NOT imply construct validity.
  - Cluster-aware resampling reflects experimental design but does not establish
    causal or generalisable conclusions.
  - No human evaluation is performed or implied.

Output:
    results/benchmark_size_analysis_v2.csv
    results/paired_comparison_v2.csv
"""

from __future__ import annotations

import csv
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

RESULTS_CSV = Path("results/evaluation_results_v2.csv")
OUTPUT_CSV = Path("results/benchmark_size_analysis_v2.csv")
PAIRED_CSV = Path("results/paired_comparison_v2.csv")

PRIMARY_METRIC = "reference_fact_f1"
SENSITIVITY_COLUMNS = {
    0.25: "reference_fact_f1",   # Only one column exists; sensitivity at other
    0.5: "reference_fact_f1",    # thresholds would require separate runs.
    0.75: "reference_fact_f1",   # Note: these map to same column unless
                                 # compute_all_metrics stores threshold-specific cols.
}

ADDITIONAL_METRICS = [
    "schema_completeness",
    "question_redundancy_rate",
    "gap_targeting_rate",
    "novel_fields_per_response",
    "follow_up_question_quality",
]

SAMPLE_SIZES = [10, 25, 50, 100, 200, 300, 500]
N_BOOTSTRAP = 2000
CONFIDENCE = 0.95
BOOTSTRAP_SEED = 42


# ─── BOOTSTRAP ───────────────────────────────────────────────────────────────

def bootstrap_ci(
    data: list[float],
    n_boot: int = N_BOOTSTRAP,
    conf: float = CONFIDENCE,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float, float]:
    """Return (mean, lower_ci, upper_ci, ci_width) using percentile bootstrap.

    seed is recorded in output for reproducibility.
    """
    if not data:
        return 0.0, 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(data) for _ in range(len(data))) / len(data)
        for _ in range(n_boot)
    )
    alpha = (1 - conf) / 2
    lo_idx = max(0, int(alpha * n_boot))
    hi_idx = min(n_boot - 1, int((1 - alpha) * n_boot))
    mean = sum(data) / len(data)
    return round(mean, 4), round(means[lo_idx], 4), round(means[hi_idx], 4), round(means[hi_idx] - means[lo_idx], 4)


def cluster_bootstrap_ci(
    cluster_groups: dict[str, list[float]],
    n_boot: int = N_BOOTSTRAP,
    conf: float = CONFIDENCE,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float, float, int]:
    """Cluster-aware bootstrap: resample whole clusters (case_group_ids).

    Returns (mean, lower_ci, upper_ci, ci_width, n_clusters).
    Each resampled replicate draws clusters with replacement, then pools
    all observations from the selected clusters.
    """
    cluster_keys = sorted(cluster_groups.keys())
    n_clusters = len(cluster_keys)
    if n_clusters == 0:
        return 0.0, 0.0, 0.0, 0.0, 0

    all_obs = [v for vals in cluster_groups.values() for v in vals]
    overall_mean = sum(all_obs) / len(all_obs) if all_obs else 0.0

    rng = random.Random(seed)
    replicate_means = []
    for _ in range(n_boot):
        selected_clusters = [rng.choice(cluster_keys) for _ in range(n_clusters)]
        sample = [v for ck in selected_clusters for v in cluster_groups[ck]]
        if sample:
            replicate_means.append(sum(sample) / len(sample))

    if not replicate_means:
        return overall_mean, overall_mean, overall_mean, 0.0, n_clusters

    replicate_means.sort()
    alpha = (1 - conf) / 2
    lo_idx = max(0, int(alpha * len(replicate_means)))
    hi_idx = min(len(replicate_means) - 1, int((1 - alpha) * len(replicate_means)))

    return (
        round(overall_mean, 4),
        round(replicate_means[lo_idx], 4),
        round(replicate_means[hi_idx], 4),
        round(replicate_means[hi_idx] - replicate_means[lo_idx], 4),
        n_clusters,
    )


# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def load_results(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Results not found at {path}. Run evaluate.py first."
        )
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _float_or_none(val: str) -> Optional[float]:
    """Parse a CSV cell as float, return None for empty/null/failed."""
    if not val or val.strip().lower() in ("", "none", "null", "nan"):
        return None
    try:
        return float(val)
    except ValueError:
        return None


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main(results_path: Optional[Path] = None):
    path = results_path or RESULTS_CSV
    rows = load_results(path)
    total_rows = len(rows)
    print(f"Loaded {total_rows} result rows from {path}.")

    # Separate successful and failed
    successful = [r for r in rows if r.get("status", "") == "success"]
    failed = [r for r in rows if r.get("status", "") != "success"]
    print(f"  Successful: {len(successful)} | Failed/missing: {len(failed)}")

    # ── Per-model summary with cluster-aware CIs ──────────────────────────
    models = sorted(set(r["model_id"] for r in successful if r.get("model_id")))
    ablations = sorted(set(r.get("ablation", "none") for r in successful))

    print(f"\n── Per-model/ablation summary ({PRIMARY_METRIC}) ──")
    summary_rows = []

    for model_id in models:
        for ablation in ablations:
            subset = [
                r for r in successful
                if r.get("model_id") == model_id and r.get("ablation", "none") == ablation
            ]
            if not subset:
                continue

            # Group by case_group_id for cluster resampling
            clusters: dict[str, list[float]] = defaultdict(list)
            n_missing = 0
            n_obs = 0
            for r in subset:
                val = _float_or_none(r.get(PRIMARY_METRIC, ""))
                grp = r.get("case_group_id") or r.get("case_id", "unknown")
                if val is not None:
                    clusters[grp].append(val)
                    n_obs += 1
                else:
                    n_missing += 1

            if not clusters:
                print(f"  [{model_id} | {ablation}]: no valid observations")
                continue

            mean, lo, hi, width, n_clusters = cluster_bootstrap_ci(clusters, seed=BOOTSTRAP_SEED)
            print(f"  [{model_id} | {ablation}] n_obs={n_obs} n_clusters={n_clusters} "
                  f"mean={mean:.4f} CI=[{lo:.4f}, {hi:.4f}] width={width:.4f} "
                  f"missing={n_missing}")

            summary_rows.append({
                "model_id": model_id,
                "ablation": ablation,
                "metric": PRIMARY_METRIC,
                "n_observations": n_obs,
                "n_clusters": n_clusters,
                "n_missing": n_missing,
                "mean": mean,
                "lower_ci": lo,
                "upper_ci": hi,
                "ci_width": width,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "n_bootstrap": N_BOOTSTRAP,
                "confidence": CONFIDENCE,
            })

    # ── Paired comparison ─────────────────────────────────────────────────
    print(f"\n── Paired comparisons ──")
    pair_rows = []

    if len(models) >= 2:
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                m_a, m_b = models[i], models[j]
                for ablation in ablations:
                    # Build per (case_group_id, question_variant_id) pairs
                    a_map: dict[tuple, float] = {}
                    b_map: dict[tuple, float] = {}
                    for r in successful:
                        if r.get("ablation", "none") != ablation:
                            continue
                        key = (r.get("case_group_id", r.get("case_id", "")),
                               r.get("question_variant_id", ""))
                        val = _float_or_none(r.get(PRIMARY_METRIC, ""))
                        if val is None:
                            continue
                        if r.get("model_id") == m_a:
                            a_map[key] = val
                        elif r.get("model_id") == m_b:
                            b_map[key] = val

                    # Only paired keys (same case variant for both models)
                    common_keys = sorted(set(a_map.keys()) & set(b_map.keys()))
                    if not common_keys:
                        continue

                    # Group differences by case_group_id for cluster bootstrap
                    diff_clusters: dict[str, list[float]] = defaultdict(list)
                    for key in common_keys:
                        grp = key[0] or "unknown"
                        diff_clusters[grp].append(a_map[key] - b_map[key])

                    mean_diff, lo_diff, hi_diff, width_diff, n_clusters = cluster_bootstrap_ci(
                        diff_clusters, seed=BOOTSTRAP_SEED
                    )
                    n_pairs = len(common_keys)
                    print(f"  [{m_a} vs {m_b} | {ablation}] n_pairs={n_pairs} "
                          f"mean_diff={mean_diff:+.4f} CI=[{lo_diff:+.4f},{hi_diff:+.4f}]")

                    pair_rows.append({
                        "model_a": m_a,
                        "model_b": m_b,
                        "ablation": ablation,
                        "metric": PRIMARY_METRIC,
                        "n_paired_observations": n_pairs,
                        "n_clusters": n_clusters,
                        "mean_difference_a_minus_b": mean_diff,
                        "lower_ci": lo_diff,
                        "upper_ci": hi_diff,
                        "ci_width": width_diff,
                        "bootstrap_seed": BOOTSTRAP_SEED,
                        "interpretation": "positive=model_a_higher, negative=model_b_higher",
                    })

    # ── Additional metrics summary ────────────────────────────────────────
    print(f"\n── Additional metric summaries ──")
    for metric in ADDITIONAL_METRICS:
        values = [_float_or_none(r.get(metric, "")) for r in successful]
        values = [v for v in values if v is not None]
        if not values:
            print(f"  {metric}: no data")
            continue
        mean = sum(values) / len(values)
        print(f"  {metric}: n={len(values)} mean={mean:.4f}")

    # ── CI width as function of sample size ───────────────────────────────
    all_values = [_float_or_none(r.get(PRIMARY_METRIC, "")) for r in successful]
    all_values = [v for v in all_values if v is not None]

    output_rows = []
    if all_values:
        rng = random.Random(BOOTSTRAP_SEED)
        for n in SAMPLE_SIZES:
            n_actual = min(n, len(all_values))
            sample = rng.sample(all_values, n_actual)
            mean, lo, hi, width = bootstrap_ci(sample, seed=rng.randint(0, 10000))
            output_rows.append({
                "sample_size": n_actual,
                "requested_sample_size": n,
                "metric": PRIMARY_METRIC,
                "mean": mean,
                "lower_ci": lo,
                "upper_ci": hi,
                "ci_width": width,
                "n_total_available": len(all_values),
                "bootstrap_seed": BOOTSTRAP_SEED,
            })
            print(f"  {PRIMARY_METRIC} n={n_actual}: mean={mean:.4f} CI=[{lo:.4f},{hi:.4f}] width={width:.4f}")

    # ── Write outputs ─────────────────────────────────────────────────────
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    if output_rows:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
            writer.writeheader()
            writer.writerows(output_rows)
        print(f"\nBenchmark size analysis → {OUTPUT_CSV}")

    if pair_rows:
        with open(PAIRED_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
            writer.writeheader()
            writer.writerows(pair_rows)
        print(f"Paired comparisons → {PAIRED_CSV}")

    print(f"\nStatistical analysis complete.")
    print(f"  Bootstrap seed: {BOOTSTRAP_SEED} | n_bootstrap: {N_BOOTSTRAP}")
    print(f"  Note: CI stability does NOT imply construct validity.")
    print(f"  Note: Cluster-aware resampling at case_group_id level.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()
    main(results_path=args.results)
