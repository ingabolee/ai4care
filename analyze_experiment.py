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
    "turns_to_completion",
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


def _cluster_bootstrap_rate(trace_rows: list[dict], reps: int, seed: int):
    """Bootstrap a trace-level binary outcome by independent source group."""
    by_group = defaultdict(list)
    for row in trace_rows:
        rate = number(row.get("good_or_excellent_rate"))
        if rate is not None:
            by_group[cluster_key(row)].append(rate)
    groups = sorted(by_group)
    observed = mean([value for group in groups for value in by_group[group]])
    if len(groups) < 2 or reps <= 0:
        return {"estimate": observed, "ci95_low": None, "ci95_high": None,
                "clusters": len(groups)}
    rng = random.Random(seed)
    boot = []
    for _ in range(reps):
        sampled = [rng.choice(groups) for _ in groups]
        values = [value for group in sampled for value in by_group[group]]
        if values:
            boot.append(mean(values))
    return {"estimate": observed, "ci95_low": percentile(boot, 0.025),
            "ci95_high": percentile(boot, 0.975), "clusters": len(groups)}


def judge_analysis(judge_rows: list[dict], reps: int, seed: int):
    """Summarize ordinal ratings as descriptive judge-assessed quality.

    Scores 3–4 are the rubric's good/excellent categories. This outcome is not
    a measure of inter-rater reliability.
    """
    attempts_by_judge = defaultdict(int)
    valid_by_judge = defaultdict(int)
    trace_scores = defaultdict(list)
    metadata_by_run = {}
    judge_ids = sorted({str(row.get("judge_id", "")) for row in judge_rows if row.get("judge_id")})
    valid_rows = []
    for row in judge_rows:
        judge_id = str(row.get("judge_id", ""))
        run_id = str(row.get("run_id", ""))
        if judge_id:
            attempts_by_judge[judge_id] += 1
        if str(row.get("parse_status", "success")).lower() not in ("success", "ok", ""):
            continue
        score = number(row.get("score"))
        if score not in (0, 1, 2, 3, 4) or not run_id or not judge_id:
            continue
        clean_row = {**row, "score": int(score)}
        valid_rows.append(clean_row)
        valid_by_judge[judge_id] += 1
        trace_scores[run_id].append(int(score))
        metadata_by_run[run_id] = {
            "candidate_model_id": row.get("candidate_model_id", ""),
            "case_group_id": row.get("case_group_id") or row.get("case_id") or run_id,
        }

    per_trace = []
    for run_id, scores in trace_scores.items():
        metadata = metadata_by_run[run_id]
        good_count = sum(score >= 3 for score in scores)
        per_trace.append({
            "run_id": run_id,
            "candidate_model_id": metadata["candidate_model_id"],
            "case_group_id": metadata["case_group_id"],
            "valid_judges": len(scores),
            "good_or_excellent_count": good_count,
            "good_or_excellent_rate": good_count / len(scores),
            "panel_median": statistics.median(scores),
        })

    by_candidate = defaultdict(list)
    for row in per_trace:
        if row["candidate_model_id"]:
            by_candidate[row["candidate_model_id"]].append(row)

    candidate_quality = []
    score_distribution = []
    for candidate, traces in sorted(by_candidate.items()):
        run_ids = {row["run_id"] for row in traces}
        scores = [row["score"] for row in valid_rows
                  if str(row.get("run_id", "")) in run_ids]
        quality_ci = _cluster_bootstrap_rate(traces, reps, seed)
        counts = {score: scores.count(score) for score in range(5)}
        candidate_quality.append({
            "candidate_model_id": candidate,
            "traces_scored": len(traces),
            "judge_ratings": len(scores),
            "good_or_excellent_ratings": counts[3] + counts[4],
            "good_or_excellent_rate": quality_ci["estimate"],
            "ci95_low": quality_ci["ci95_low"],
            "ci95_high": quality_ci["ci95_high"],
            "independent_source_groups": quality_ci["clusters"],
            "mean_judge_score_descriptive": mean(scores),
            "median_judge_score": statistics.median(scores) if scores else None,
        })
        for score in range(5):
            score_distribution.append({
                "candidate_model_id": candidate,
                "score": score,
                "rating_count": counts[score],
                "rating_share": counts[score] / len(scores) if scores else None,
            })

    all_scores = [row["score"] for row in valid_rows]
    overall_ci = _cluster_bootstrap_rate(per_trace, reps, seed)
    judge_coverage = [
        {"judge_id": judge_id,
         "attempts": attempts_by_judge[judge_id],
         "valid_ratings": valid_by_judge[judge_id],
         "failed_or_invalid": attempts_by_judge[judge_id] - valid_by_judge[judge_id]}
        for judge_id in judge_ids
    ]
    return {
        "judge_ids": judge_ids,
        "valid_score_rows": len(valid_rows),
        "traces_with_scores": len(per_trace),
        "judge_coverage": judge_coverage,
        "overall_good_or_excellent_rate": overall_ci["estimate"],
        "overall_good_or_excellent_ci95_low": overall_ci["ci95_low"],
        "overall_good_or_excellent_ci95_high": overall_ci["ci95_high"],
        "overall_score_counts": {str(score): all_scores.count(score) for score in range(5)},
        "candidate_quality_summary": candidate_quality,
        "score_distribution": score_distribution,
        "per_trace": per_trace,
        "amended_question": (
            "How frequently were interview follow-up questions rated good or excellent, "
            "and how did this vary across candidate models?"
        ),
        "analysis_amendment": (
            "This completed-run analysis reframes RQ3 as descriptive judge-assessed quality. "
            "It does not answer the original inter-rater reliability question."
        ),
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

    judges = judge_analysis(judge_rows, reps, seed)
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
    fact_f1_pairs = [r for r in rq1_pairs if r["metric"] == "reference_fact_f1"]
    clear_fact_f1_differences = [r for r in fact_f1_pairs
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
    schema_estimates = [row for row in rq1_summary if row["metric"] == "schema_completeness"]
    schema_pairs = [row for row in rq1_pairs if row["metric"] == "schema_completeness"]
    fact_f1_estimates = [row for row in rq1_summary if row["metric"] == "reference_fact_f1"]
    highest_schema = max(schema_estimates, key=lambda row: row["estimate"] or -math.inf) if schema_estimates else None
    lowest_schema = min(schema_estimates, key=lambda row: row["estimate"] if row["estimate"] is not None else math.inf) if schema_estimates else None
    highest_fact_f1 = max(fact_f1_estimates, key=lambda row: row["estimate"] or -math.inf) if fact_f1_estimates else None
    lowest_fact_f1 = min(fact_f1_estimates, key=lambda row: row["estimate"] if row["estimate"] is not None else math.inf) if fact_f1_estimates else None
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
            f"{len(clear_schema_differences)} of {len(schema_pairs)} pairwise schema-completeness "
            "source-cluster intervals exclude zero in this finite benchmark; "
            f"the highest mean was {highest_schema['model_id']} ({highest_schema['estimate']:.3f}) "
            f"and the lowest was {lowest_schema['model_id']} ({lowest_schema['estimate']:.3f}). "
            f"For the reference-fact F1 proxy, {len(clear_fact_f1_differences)} of "
            f"{len(fact_f1_pairs)} paired intervals exclude zero; its highest mean was "
            f"{highest_fact_f1['model_id']} ({highest_fact_f1['estimate']:.3f}) and lowest was "
            f"{lowest_fact_f1['model_id']} ({lowest_fact_f1['estimate']:.3f}). The measures rank "
            "models differently. Treat these as benchmark-specific evidence: only three independent source groups "
            "were sampled, and pairwise intervals are unadjusted for multiple comparisons."
        )
    else:
        rq1_finding = (
            "no clear model difference detected on this finite proof-of-concept benchmark; "
            "this does not establish equivalence or population-level performance"
        )
    novelty_rows = [row for row in rq2_summary if row["metric"] == "novel_fields_per_response"]
    targeting_rows = [row for row in rq2_summary if row["metric"] == "gap_targeting_rate"]
    redundancy_rows = [row for row in rq2_summary if row["metric"] == "question_redundancy_rate"]
    best_novelty = max(novelty_rows, key=lambda row: row["mean"] if row["mean"] is not None else -math.inf) if novelty_rows else None
    best_targeting = max(targeting_rows, key=lambda row: row["mean"] if row["mean"] is not None else -math.inf) if targeting_rows else None
    highest_redundancy = max(redundancy_rows, key=lambda row: row["mean"] if row["mean"] is not None else -math.inf) if redundancy_rows else None
    lowest_redundancy = min(redundancy_rows, key=lambda row: row["mean"] if row["mean"] is not None else math.inf) if redundancy_rows else None
    turn_values = sorted({number(row.get("turn_count")) for row in successful if number(row.get("turn_count")) is not None})
    completion_turns = [number(row.get("turns_to_completion")) for row in successful]
    completion_turns = [value for value in completion_turns if value is not None]
    schema_completed = sum(row.get("termination_reason") == "schema_threshold" for row in successful)
    reply_limited = sum(row.get("termination_reason") == "reply_limit" for row in successful)
    if best_novelty and best_targeting and highest_redundancy and lowest_redundancy:
        rq2_finding = (
            f"Descriptively, {best_novelty['model_id']} had the highest novel-field gain "
            f"({_fmt(best_novelty['mean'], 2)} per response), {best_targeting['model_id']} had the "
            f"highest gap-targeting rate ({_fmt(best_targeting['mean'], 2)}). Redundancy was highest "
            f"for {highest_redundancy['model_id']} ({_fmt(highest_redundancy['mean'], 2)}) and lowest "
            f"for {lowest_redundancy['model_id']} ({_fmt(lowest_redundancy['mean'], 2)}). "
            f"The schema-completion threshold was reached in {schema_completed}/{len(successful)} "
            f"episodes (mean {_fmt(mean(completion_turns), 2)} replies among completers; "
            f"{reply_limited} reached the reply limit). Turn counts ranged from "
            f"{turn_values[0]:g} to {turn_values[-1]:g}. No single efficiency measure has a "
            "prespecified pass threshold."
        )
    else:
        rq2_finding = "insufficient: no successful episode metrics are available for comparison"

    good_rate = judges["overall_good_or_excellent_rate"]
    highest_quality = max(
        judges["candidate_quality_summary"],
        key=lambda row: row["good_or_excellent_rate"] if row["good_or_excellent_rate"] is not None else -math.inf,
        default=None,
    )
    lowest_quality = min(
        judges["candidate_quality_summary"],
        key=lambda row: row["good_or_excellent_rate"] if row["good_or_excellent_rate"] is not None else math.inf,
        default=None,
    )
    if good_rate is None or not highest_quality or not lowest_quality:
        rq3_finding = "insufficient: no valid judge ratings matched to this experiment"
    else:
        rq3_finding = (
            f"Under the amended descriptive quality question, {good_rate:.1%} of valid ratings "
            "were good or excellent. By candidate model the observed range was "
            f"{lowest_quality['good_or_excellent_rate']:.1%} ({lowest_quality['candidate_model_id']}) "
            f"to {highest_quality['good_or_excellent_rate']:.1%} ({highest_quality['candidate_model_id']}). "
            "This does not answer the original inter-rater reliability question or establish a pass mark."
        )

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
            "finding": rq2_finding,
            "descriptive_statistics": rq2_summary,
            "turn_count_unique_values": turn_values,
            "schema_completed_episodes": schema_completed,
            "reply_limited_episodes": reply_limited,
            "turns_to_completion_n": len(completion_turns),
            "turns_to_completion_mean": mean(completion_turns),
            "formal_threshold_specified": False,
        },
        "rq3": {"finding": rq3_finding, **judges},
        "interpretation_limits": [
            "The full profile evaluates benchmark variants from a small finite proof-of-concept corpus; it is not a representative enterprise sample.",
            "Question variants from one case_group_id are repeated measures, not independent source documents; report the source-group count with all results.",
            f"Cluster-bootstrap intervals are based on {len(groups)} source groups; with three groups they are coarse and should be interpreted descriptively.",
            "Lexical reference-fact F1 is a proxy and judge-assessed quality does not establish human validity.",
            "RQ2 summaries are descriptive; the thesis does not specify a formal significance test for those outcomes.",
            "The completed-run RQ3 analysis was explicitly reframed from inter-rater reliability to descriptive judge-assessed quality; the original reliability question remains unanswered by this outcome.",
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
    write_csv(analysis_dir / "rq3_good_excellent_by_candidate.csv", judges["candidate_quality_summary"],
              ["candidate_model_id", "traces_scored", "judge_ratings", "good_or_excellent_ratings", "good_or_excellent_rate", "ci95_low", "ci95_high", "independent_source_groups", "mean_judge_score_descriptive", "median_judge_score"])
    write_csv(analysis_dir / "rq3_score_distribution.csv", judges["score_distribution"],
              ["candidate_model_id", "score", "rating_count", "rating_share"])
    write_csv(analysis_dir / "rq3_judge_coverage.csv", judges["judge_coverage"],
              ["judge_id", "attempts", "valid_ratings", "failed_or_invalid"])
    write_csv(analysis_dir / "rq3_panel_by_trace.csv", judges["per_trace"],
              ["run_id", "candidate_model_id", "case_group_id", "valid_judges", "good_or_excellent_count", "good_or_excellent_rate", "panel_median"])
    for obsolete in ("rq3_judge_pairs.csv", "rq3_case_group_correlations.csv", "rq3_candidate_panel_summary.csv"):
        obsolete_path = analysis_dir / obsolete
        if obsolete_path.exists():
            obsolete_path.unlink()

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
        f"For the reference-fact F1 proxy, {len(clear_fact_f1_differences)} of {len(fact_f1_pairs)} paired source-group intervals exclude zero; the highest model mean was {highest_fact_f1['model_id']} ({highest_fact_f1['estimate']:.3f}) and the lowest was {lowest_fact_f1['model_id']} ({lowest_fact_f1['estimate']:.3f}). Schema completeness and reference-fact F1 rank models differently. Reference-fact F1 is a lexical proxy applied only to full-source facts disclosed by the synthetic respondent.",
        "A paired interval excluding zero is benchmark-specific evidence of a difference; an interval including zero leaves that comparison unresolved, not proof of equivalence. These pairwise intervals are unadjusted and based on only three source groups.",
        "", "## RQ2 — Elicitation efficiency", "",
        rq2_finding,
        "Model-level descriptive values are in `rq2_descriptive_statistics.csv`. `turns_to_completion` includes only episodes that reached the schema-completion threshold; episodes ending at the reply limit are counted separately because their completion time is censored. Read turn counts with schema completeness and reference-fact F1.",
        "", "## RQ3 — Amended question: judge-assessed follow-up quality", "",
        rq3_finding,
        "This post-completion amendment uses the rubric's good/excellent categories (scores 3–4) to describe judge-assessed quality. It does not measure judge reliability, and the original reliability question remains unanswered. The overall and per-model rating shares have no pass threshold; see `rq3_good_excellent_by_candidate.csv`, `rq3_score_distribution.csv`, and `rq3_judge_coverage.csv`.",
        "", "## Interpretation limits", "",
    ]
    report.extend(f"- {note}" for note in summary["interpretation_limits"])
    (analysis_dir / "analysis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if manifest_path.exists():
        manifest.setdefault("stage_status", {})["analysis"] = "complete"
        manifest.setdefault("paths", {})["analysis_output"] = str(analysis_dir)
        manifest.setdefault("analysis_amendments", {})["rq3"] = (
            "Post-completion descriptive quality analysis using rubric scores 3–4; "
            "this does not answer the original inter-rater reliability question."
        )
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
