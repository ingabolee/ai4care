"""
panel_aggregator.py -- Evaluation MoE Panel Aggregator.

Provides a single shared aggregation function used by both evaluate.py and
judge_benchmark.py to prevent silent drift between the two paths.

Metric schema version: v2
Metric: follow_up_question_quality (0–4 ordinal integer)

Disagreement measure: median absolute deviation (MAD) across valid scores.
Disagreement is returned as None (not estimable) when fewer than 2 judges
provide valid scores.

IMPORTANT:
  - Panel agreement does NOT establish correctness.
  - Judges sharing a model family or API endpoint are not statistically independent.
"""

from __future__ import annotations

import statistics
from typing import Optional, List
from llm_judge import LLMJudge, METRIC


def aggregate_judge_results(results: list[dict]) -> dict:
    """Aggregate judge results into a panel summary.

    Inputs:
        results: list of result dicts from LLMJudge.evaluate_interview(), each with:
                 "success": bool
                 "score": int or None
                 (other fields preserved for diagnostics)

    Returns:
        panel_median         : float or None  (None if no valid scores)
        disagreement_mad     : float or None  (None if fewer than 2 valid scores)
        successful_judges    : int
        total_judges         : int
        valid_scores         : list[int]
        status               : str

    Disagreement interpretation:
        MAD = median absolute deviation from the panel median.
        0.0 MAD means all valid judges agreed exactly.
        None MAD means disagreement is not estimable (only 0 or 1 valid scores).
        Do NOT interpret None as zero disagreement.
    """
    total_judges = len(results)

    valid_scores: list[int] = []
    for r in results:
        if r.get("success", False) and r.get("score") is not None:
            score = r["score"]
            # Extra guard: only include values that are already-validated integers in range
            if isinstance(score, int) and 0 <= score <= 4:
                valid_scores.append(score)

    successful_judges = len(valid_scores)

    if successful_judges == 0:
        return {
            "panel_median": None,
            "disagreement_mad": None,
            "successful_judges": 0,
            "total_judges": total_judges,
            "valid_scores": [],
            "status": "no_valid_scores",
        }

    panel_median = statistics.median(valid_scores)

    if successful_judges < 2:
        disagreement_mad = None  # Not estimable from a single score
        status = "single_valid_score_disagreement_not_estimable"
    else:
        # Median absolute deviation
        abs_deviations = [abs(s - panel_median) for s in valid_scores]
        disagreement_mad = statistics.median(abs_deviations)
        status = "ok"

    return {
        "panel_median": panel_median,
        "disagreement_mad": disagreement_mad,
        "successful_judges": successful_judges,
        "total_judges": total_judges,
        "valid_scores": valid_scores,
        "status": status,
    }


class PanelAggregator:
    """Wraps a panel of LLMJudge instances and aggregates their results.

    Use aggregate_judge_results() directly if you have pre-collected results.
    """

    def __init__(self, judges: List[LLMJudge]):
        self.judges = judges

    def evaluate_with_panel(self, trace: dict) -> dict:
        """Evaluate a trace with all judges and aggregate results.

        Returns the aggregate_judge_results() dict plus the raw judge_responses list.
        """
        results = [judge.evaluate_interview(trace) for judge in self.judges]
        agg = aggregate_judge_results(results)
        agg["judge_responses"] = results
        return agg
