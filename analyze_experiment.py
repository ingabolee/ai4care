"""Analyze one AI4CARE experiment against the thesis research questions.

Inputs are the run-specific evaluation_results.csv and judges/judge_scores_long.csv.
Outputs are written beneath <experiment-dir>/analysis so pilot and full analyses
remain separate and reproducible.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


RQ1_METRICS = ("schema_completeness", "reference_fact_f1")
RQ2_METRICS = (
    "novel_fields_per_response",
    "question_redundancy_rate",
    "gap_targeting_rate",
    "turn_count",
)
def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def number(value):
    if value is None or str(value).strip() in ("", "None", "null", "NA", "nan"):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def mean(values: list[float]):
    return statistics.mean(values) if values else None


def percentile(values: list[float], p: float):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def cluster_key(row: dict) -> str:
    return str(row.get("case_group_id") or row.get("case_id") or "missing_group")


def clustered_mean_ci(rows: list[dict], metric: str, reps: int, seed: int):
    """Percentile CI resampling source-case clusters with their rows intact."""
    clusters = defaultdict(list)
    for row in rows:
        value = number(row.get(metric))
        if value is not None:
            clusters[cluster_key(row)].append(value)
    keys = sorted(clusters)
    if not keys:
        return {"estimate": None, "ci95_low": None, "ci95_high": None, "clusters": 0, "n": 0}
    observed = mean([v for key in keys for v in clusters[key]])
    if len(keys) < 2 or reps <= 0:
        return {"estimate": observed, "ci95_low": None, "ci95_high": None,
                "clusters": len(keys), "n": sum(map(len, clusters.values()))}
    rng = random.Random(seed)
    boot = []
    for _ in range(reps):
        sampled = [rng.choice(keys) for _ in keys]
        values = [v for key in sampled for v in clusters[key]]
        if values:
            boot.append(mean(values))
    return {
        "estimate": observed,
        "ci95_low": percentile(boot, 0.025),
        "ci95_high": percentile(boot, 0.975),
        "clusters": len(keys),
        "n": sum(map(len, clusters.values())),
    }


def paired_cluster_difference(rows: list[dict], model_a: str, model_b: str,
                              metric: str, reps: int, seed: int):
    """Paired A-B differences on common assignments, bootstrapped by source case."""
    paired_by_group = defaultdict(list)
    a_by_key, b_by_key = {}, {}
    for row in rows:
        value = number(row.get(metric))
        if value is None:
            continue
        key = (cluster_key(row), row.get("question_variant_id", ""),
               row.get("iteration_id", ""), row.get("ablation", ""))
        if row.get("model_id") == model_a:
            a_by_key[key] = value
        elif row.get("model_id") == model_b:
            b_by_key[key] = value
    for key in sorted(a_by_key.keys() & b_by_key.keys()):
        paired_by_group[key[0]].append(a_by_key[key] - b_by_key[key])
    groups = sorted(paired_by_group)
    if not groups:
        return {"model_a": model_a, "model_b": model_b, "metric": metric,
                "estimate_a_minus_b": None, "ci95_low": None, "ci95_high": None,
                "paired_clusters": 0, "paired_assignments": 0}
    observed = mean([d for g in groups for d in paired_by_group[g]])
    if len(groups) < 2 or reps <= 0:
        low = high = None
    else:
        rng = random.Random(seed)
        boot = []
        for _ in range(reps):
            sampled = [rng.choice(groups) for _ in groups]
            boot.append(mean([d for g in sampled for d in paired_by_group[g]]))
        low, high = percentile(boot, 0.025), percentile(boot, 0.975)
    return {
        "model_a": model_a, "model_b": model_b, "metric": metric,
        "estimate_a_minus_b": observed, "ci95_low": low, "ci95_high": high,
        "paired_clusters": len(groups),
        "paired_assignments": sum(map(len, paired_by_group.values())),
    }


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for position in range(i, j):
            ranks[order[position]] = rank
        i = j
    return ranks


def spearman(xs: list[float], ys: list[float]):
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx, ry = average_ranks(xs), average_ranks(ys)
    mx, my = mean(rx), mean(ry)
    dx, dy = [x - mx for x in rx], [y - my for y in ry]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if denom == 0:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / denom


def judge_analysis(judge_rows: list[dict]):
    scores = defaultdict(dict)
    candidate_by_run = {}
    case_group_by_run = {}
    judge_ids = sorted({str(r.get("judge_id", "")) for r in judge_rows if r.get("judge_id")})
    for row in judge_rows:
        if str(row.get("parse_status", "success")).lower() not in ("success", "ok", ""):
            continue
        score = number(row.get("score"))
        if score is None:
            continue
        run_id = str(row.get("run_id", ""))
        judge_id = str(row.get("judge_id", ""))
        if run_id and judge_id:
            scores[run_id][judge_id] = score
            candidate_by_run[run_id] = row.get("candidate_model_id", "")
            case_group_by_run[run_id] = row.get("case_group_id") or row.get("case_id") or run_id

    pairs = []
    for judge_a, judge_b in itertools.combinations(judge_ids, 2):
        common = sorted(run for run, per_judge in scores.items()
                        if judge_a in per_judge and judge_b in per_judge)
        rho = spearman([scores[r][judge_a] for r in common],
                       [scores[r][judge_b] for r in common])
        pairs.append({
            "judge_a": judge_a,
            "judge_b": judge_b,
            "spearman_rho": rho,
            "n_shared_traces": len(common),
            "rho_gt_0_7": rho is not None and rho > 0.7,
        })

    per_trace = []
    mad_values = []
    coverage = defaultdict(int)
    for run_id, per_judge in scores.items():
        valid = list(per_judge.values())
        if len(valid) >= 2:
            med = statistics.median(valid)
            mad = statistics.median([abs(v - med) for v in valid])
            mad_values.append(mad)
        else:
            med, mad = (statistics.median(valid) if valid else None), None
        for jid in per_judge:
            coverage[jid] += 1
        per_trace.append({
            "run_id": run_id,
            "candidate_model_id": candidate_by_run.get(run_id, ""),
            "panel_median": med,
            "disagreement_mad": mad,
            "successful_judges": len(valid),
        })

    model_scores = defaultdict(lambda: defaultdict(list))
    for run_id, per_judge in scores.items():
        candidate = candidate_by_run.get(run_id, "")
        if not candidate:
            continue
        for jid, score in per_judge.items():
            model_scores[jid][candidate].append(score)
    mean_by_model = {
        jid: {mid: mean(vals) for mid, vals in candidates.items()}
        for jid, candidates in model_scores.items()
    }
    model_rank_correlations = []
    for judge_a, judge_b in itertools.combinations(judge_ids, 2):
        common_models = sorted(set(mean_by_model.get(judge_a, {})) & set(mean_by_model.get(judge_b, {})))
        rho = spearman([mean_by_model[judge_a][m] for m in common_models],
                       [mean_by_model[judge_b][m] for m in common_models])
        model_rank_correlations.append({
            "judge_a": judge_a, "judge_b": judge_b, "spearman_rho": rho,
            "n_candidate_models": len(common_models),
        })

    group_scores = defaultdict(lambda: defaultdict(list))
    for run_id, per_judge in scores.items():
        group = case_group_by_run.get(run_id, run_id)
        for jid, score in per_judge.items():
            group_scores[jid][group].append(score)
    case_group_correlations = []
    for judge_a, judge_b in itertools.combinations(judge_ids, 2):
        common_groups = sorted(set(group_scores.get(judge_a, {})) & set(group_scores.get(judge_b, {})))
        rho = spearman(
            [mean(group_scores[judge_a][g]) for g in common_groups],
            [mean(group_scores[judge_b][g]) for g in common_groups],
        )
        case_group_correlations.append({
            "judge_a": judge_a, "judge_b": judge_b, "spearman_rho": rho,
            "n_independent_case_groups": len(common_groups),
        })

    panel_by_candidate = defaultdict(lambda: {"medians": [], "mads": [], "traces": 0})
    for row in per_trace:
        candidate = row.get("candidate_model_id")
        if not candidate:
            continue
        item = panel_by_candidate[candidate]
        item["traces"] += 1
        if row.get("panel_median") is not None:
            item["medians"].append(row["panel_median"])
        if row.get("disagreement_mad") is not None:
            item["mads"].append(row["disagreement_mad"])
    candidate_panel_summary = [
        {
            "candidate_model_id": candidate,
            "traces_with_scores": vals["traces"],
            "mean_panel_median": mean(vals["medians"]),
            "median_panel_median": statistics.median(vals["medians"]) if vals["medians"] else None,
            "mean_episode_mad": mean(vals["mads"]),
            "traces_with_estimable_mad": len(vals["mads"]),
        }
        for candidate, vals in sorted(panel_by_candidate.items())
    ]

    valid_pairs = [p for p in pairs if p["spearman_rho"] is not None]
    threshold_pairs = sum(p["rho_gt_0_7"] for p in valid_pairs)
    return {
        "judge_ids": judge_ids,
        "valid_score_rows": sum(len(v) for v in scores.values()),
        "traces_with_scores": len(scores),
        "successful_scores_per_judge": dict(coverage),
        "pairwise_episode_spearman": pairs,
        "pairwise_candidate_model_rank_spearman": model_rank_correlations,
        "pairwise_case_group_spearman": case_group_correlations,
        "candidate_panel_summary": candidate_panel_summary,
        "episode_panel_mad": {
            "n_traces_with_two_or_more_judges": len(mad_values),
            "mean": mean(mad_values),
            "median": statistics.median(mad_values) if mad_values else None,
        },
        "rho_threshold": 0.7,
        "judge_pairs_above_threshold": threshold_pairs,
        "meets_thesis_reliability_rule": threshold_pairs >= 2,
        "per_trace": per_trace,
        "note": "The thesis threshold is Spearman rho > 0.7; rho is not a p-value.",
    }


def write_csv(path: Path, rows: list[dict], columns: list[str]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value, digits=3):
    return "NA" if value is None else f"{value:.{digits}f}"


def run_analysis(experiment_dir: Path, results_path: Path, judges_path: Path,
                 reps: int, seed: int) -> dict:
    rows = read_csv(results_path)
    judge_rows = read_csv(judges_path)
    successful = [r for r in rows if r.get("status", "success") == "success"]
    models = sorted({r.get("model_id", "") for r in successful if r.get("model_id")})
    groups = sorted({cluster_key(r) for r in successful})
    ablations = sorted({r.get("ablation", "none") for r in successful})

    rq1_summary, rq1_pairs = [], []
    for model_id in models:
        model_rows = [r for r in successful if r.get("model_id") == model_id]
        for metric in RQ1_METRICS:
            estimate = clustered_mean_ci(model_rows, metric, reps, seed)
            rq1_summary.append({"model_id": model_id, "metric": metric, **estimate,
                                "bootstrap_replicates": reps, "bootstrap_seed": seed})
    for metric in RQ1_METRICS:
        for a, b in itertools.combinations(models, 2):
            result = paired_cluster_difference(successful, a, b, metric, reps, seed)
            result["bootstrap_replicates"] = reps
            result["bootstrap_seed"] = seed
            rq1_pairs.append(result)

    rq2_summary = []
    for ablation in ablations:
        condition_rows = [r for r in successful if r.get("ablation", "none") == ablation]
        for model_id in models:
            model_rows = [r for r in condition_rows if r.get("model_id") == model_id]
            for metric in RQ2_METRICS:
                values = [number(r.get(metric)) for r in model_rows]
                values = [v for v in values if v is not None]
                rq2_summary.append({
                    "ablation": ablation, "model_id": model_id, "metric": metric,
                    "n": len(values), "mean": mean(values),
                    "median": statistics.median(values) if values else None,
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                })

    judges = judge_analysis(judge_rows)
    manifest_path = experiment_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    profile = manifest.get("profile") or (successful[0].get("run_profile") if successful else "unknown")
    expected_episodes = manifest.get("expected_episode_count")
    observed_episodes = len(rows)
    coverage = (observed_episodes / expected_episodes) if expected_episodes else None
    generation_status = manifest.get("stage_status", {}).get("generation", "unknown")
    failed_by_model = defaultdict(int)
    for row in rows:
        if row.get("status") == "failed":
            failed_by_model[row.get("model_id", "unknown")] += 1
    schema_pair_differences = [r for r in rq1_pairs if r["metric"] == "schema_completeness"]
    clear_schema_differences = [r for r in schema_pair_differences
                                if r["ci95_low"] is not None and
                                (r["ci95_low"] > 0 or r["ci95_high"] < 0)]
    selected_variants = len({str(r.get("question_variant_id", "")) for r in successful
                             if r.get("question_variant_id")})
    llm_extraction_counts = [number(r.get("llm_extraction_count")) or 0 for r in successful]
    fallback_extraction_counts = [number(r.get("fallback_extraction_count")) or 0 for r in successful]
    capability_check = {
        "episodes_with_llm_extractions": sum(count > 0 for count in llm_extraction_counts),
        "episodes_with_fallback_extractions": sum(count > 0 for count in fallback_extraction_counts),
        "total_llm_extractions": sum(llm_extraction_counts),
        "total_fallback_extractions": sum(fallback_extraction_counts),
        "mean_schema_completeness": mean([
            value for row in successful
            if (value := number(row.get("schema_completeness"))) is not None
        ]),
        "mean_reference_fact_f1": mean([
            value for row in successful
            if (value := number(row.get("reference_fact_f1"))) is not None
        ]),
    }
    if generation_status == "incomplete" or (coverage is not None and coverage < 1):
        rq1_finding = "inconclusive: the experiment does not contain every planned episode"
    elif len(models) < 2:
        rq1_finding = "inconclusive: at least two candidate models are needed for a comparison"
    elif len(groups) < 2:
        rq1_finding = (
            f"descriptive proof-of-concept only: {len(groups)} independent source-case group; "
            "there are too few clusters for a meaningful cluster-bootstrap comparison"
        )
    elif clear_schema_differences:
        rq1_finding = (
            "observed model differences on this finite proof-of-concept benchmark; "
            "the small number of source documents does not support population-level claims"
        )
    else:
        rq1_finding = (
            "no clear model difference detected on this finite proof-of-concept benchmark; "
            "this does not establish equivalence or population-level performance"
        )
    if len(judges["judge_ids"]) < 2:
        rq3_finding = "inconclusive: fewer than two judges produced scores"
    elif judges["meets_thesis_reliability_rule"]:
        rq3_finding = (
            "meets the prespecified Spearman rule on this proof-of-concept benchmark; "
            f"scores are clustered in {len(groups)} source-case groups, so generalization is limited"
        )
    else:
        rq3_finding = "does not meet the prespecified Spearman rule on this finite benchmark"

    summary = {
        "analysis_schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": manifest.get("experiment_id", experiment_dir.name),
        "profile": profile,
        "input_files": {"results": str(results_path), "judges": str(judges_path)},
        "data_coverage": {
            "successful_episode_rows": len(successful),
            "benchmark_variants_evaluated": selected_variants,
            "benchmark_variants_available": manifest.get("benchmark_cases_available"),
            "failed_episode_rows": sum(r.get("status") == "failed" for r in rows),
            "failed_by_candidate_model": dict(failed_by_model),
            "expected_episode_rows": expected_episodes,
            "episode_coverage_fraction": coverage,
            "generation_status": generation_status,
            "candidate_models": models,
            "ablation_conditions": ablations,
            "independent_case_groups": len(groups),
            "analysis_scope": "finite_proof_of_concept_benchmark",
            "bootstrap_replicates": reps,
            "bootstrap_seed": seed,
        },
        "proof_of_concept_capability_check": capability_check,
        "rq1": {
            "primary_metric": "schema_completeness",
            "finding": rq1_finding,
            "cluster_bootstrap_model_means": rq1_summary,
            "paired_model_differences": rq1_pairs,
        },
        "rq2": {
            "finding": "descriptive comparison of novel-fact gain, redundancy, gap targeting, and turns by model and ablation",
            "descriptive_statistics": rq2_summary,
        },
        "rq3": {"finding": rq3_finding, **judges},
        "interpretation_limits": [
            "The full profile evaluates benchmark variants from a small finite proof-of-concept corpus; it is not a representative enterprise sample.",
            "Question variants from one case_group_id are repeated measures, not independent source documents; report the source-group count with all results.",
            f"Cluster-bootstrap intervals are based on {len(groups)} source groups; with three groups they are coarse and should be interpreted descriptively.",
            "Lexical reference-fact F1 is a proxy and judge agreement does not establish human validity.",
            "RQ2 summaries are descriptive; the thesis does not specify a formal significance test for those outcomes.",
        ],
    }

    analysis_dir = experiment_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(analysis_dir / "rq1_model_estimates.csv", rq1_summary,
              ["model_id", "metric", "estimate", "ci95_low", "ci95_high", "clusters", "n", "bootstrap_replicates", "bootstrap_seed"])
    write_csv(analysis_dir / "rq1_paired_differences.csv", rq1_pairs,
              ["model_a", "model_b", "metric", "estimate_a_minus_b", "ci95_low", "ci95_high", "paired_clusters", "paired_assignments", "bootstrap_replicates", "bootstrap_seed"])
    write_csv(analysis_dir / "rq2_descriptive_statistics.csv", rq2_summary,
              ["ablation", "model_id", "metric", "n", "mean", "median", "min", "max"])
    write_csv(analysis_dir / "rq3_judge_pairs.csv", judges["pairwise_episode_spearman"],
              ["judge_a", "judge_b", "spearman_rho", "n_shared_traces", "rho_gt_0_7"])
    write_csv(analysis_dir / "rq3_case_group_correlations.csv", judges["pairwise_case_group_spearman"],
              ["judge_a", "judge_b", "spearman_rho", "n_independent_case_groups"])
    write_csv(analysis_dir / "rq3_panel_by_trace.csv", judges["per_trace"],
              ["run_id", "candidate_model_id", "panel_median", "disagreement_mad", "successful_judges"])
    write_csv(analysis_dir / "rq3_candidate_panel_summary.csv", judges["candidate_panel_summary"],
              ["candidate_model_id", "traces_with_scores", "mean_panel_median", "median_panel_median", "mean_episode_mad", "traces_with_estimable_mad"])

    report = [
        f"# Experiment analysis: {summary['experiment_id']}", "",
        f"Profile: **{profile}**", "",
        "## Data coverage", "",
        f"- Successful episodes: {len(successful)}; failed episodes: {summary['data_coverage']['failed_episode_rows']}",
        f"- Failed episodes by model: {dict(failed_by_model) if failed_by_model else 'none'}",
        f"- Planned episode coverage: {_fmt(coverage * 100, 1) + '%' if coverage is not None else 'unknown'}",
        f"- Benchmark variants evaluated: {selected_variants} of {manifest.get('benchmark_cases_available', 'unknown')}",
        f"- Candidate models: {', '.join(models) if models else 'none'}",
        f"- Independent source-case groups: {len(groups)}",
        f"- Judge score rows: {judges['valid_score_rows']} across {judges['traces_with_scores']} traces",
        "- Scope: finite proof of concept; findings are limited to these source documents.",
        "", "## Proof-of-concept extraction check", "",
        f"LLM extractions occurred in {capability_check['episodes_with_llm_extractions']} of {len(successful)} successful episodes; total LLM extractions: {capability_check['total_llm_extractions']}.",
        f"Mean schema completeness: {_fmt(capability_check['mean_schema_completeness'])}; mean reference-fact F1: {_fmt(capability_check['mean_reference_fact_f1'])}.",
        "", "## RQ1 — Schema completeness and reference-fact F1", "",
        rq1_finding, "",
        "Model estimates and paired cluster-bootstrap intervals are in `rq1_model_estimates.csv` and `rq1_paired_differences.csv`.",
        "A confidence interval excluding zero is evidence of a difference; an interval including zero does not prove the models equivalent.",
        "", "## RQ2 — Elicitation efficiency", "",
        "Compare novel facts per response, redundancy rate, gap-targeting rate, and turns across models and ablations in `rq2_descriptive_statistics.csv`.",
        "These are descriptive summaries, as specified in the thesis analysis plan.",
        "", "## RQ3 — Judge reliability", "",
        rq3_finding,
        f"- Judge pairs with episode-level Spearman rho > 0.7: {judges['judge_pairs_above_threshold']}.",
        f"- Mean episode panel MAD: {_fmt(judges['episode_panel_mad']['mean'])} over {judges['episode_panel_mad']['n_traces_with_two_or_more_judges']} traces.",
        "- The thesis criterion is Spearman rho > 0.7; rho is not a p-value.",
        "- Candidate-model panel score summaries are in `rq3_candidate_panel_summary.csv`; independent source-case correlations are in `rq3_case_group_correlations.csv`.",
        "- Candidate-model rank correlations are also in `analysis_summary.json`. Interpret correlations cautiously when there are few models or independent source cases.",
        "", "## Interpretation limits", "",
    ]
    report.extend(f"- {note}" for note in summary["interpretation_limits"])
    (analysis_dir / "analysis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if manifest_path.exists():
        manifest.setdefault("stage_status", {})["analysis"] = "complete"
        manifest.setdefault("paths", {})["analysis_output"] = str(analysis_dir)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze one AI4CARE experiment against the thesis RQs")
    parser.add_argument("--experiment-dir", required=True,
                        help="Run folder containing evaluation_results.csv and judges/ outputs")
    parser.add_argument("--results", default=None, help="Override evaluation CSV input")
    parser.add_argument("--judges", default=None, help="Override judge_scores_long.csv input")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    experiment_dir = Path(args.experiment_dir)
    results_path = Path(args.results) if args.results else experiment_dir / "evaluation_results.csv"
    judges_path = Path(args.judges) if args.judges else experiment_dir / "judges" / "judge_scores_long.csv"
    if not results_path.exists():
        parser.error(f"Evaluation CSV not found: {results_path}")
    if args.bootstrap_replicates < 0:
        parser.error("--bootstrap-replicates cannot be negative")
    summary = run_analysis(experiment_dir, results_path, judges_path,
                           args.bootstrap_replicates, args.seed)
    print(f"Analysis saved to {experiment_dir / 'analysis'}")
    print(f"RQ1: {summary['rq1']['finding']}")
    print(f"RQ3: {summary['rq3']['finding']}")


if __name__ == "__main__":
    main()
