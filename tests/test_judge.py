"""
tests/test_judge.py -- Unit tests for LLMJudge and PanelAggregator (no real LLM required).
Uses mock ChatRunner to test parsing and aggregation logic.

Updated for metric schema v2:
  - Single follow_up_question_quality metric (0-4 ordinal integer)
  - Strict validation: no midpoint defaults, no defaults for missing keys
  - Distinct failure types: parse_failure, inference_failure, invalid_output
  - Panel MAD = None when fewer than 2 valid judges

Run with:
    python -m pytest tests/test_judge.py -v
"""

import json
import math
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from inference import ChatRunner
from llm_judge import (
    LLMJudge,
    METRIC,
    METRIC_VERSION,
    FAILURE_INFERENCE,
    FAILURE_PARSE,
    FAILURE_INVALID,
    _scrub_json,
    sanitize_trace,
    format_transcript_text,
)
from panel_aggregator import PanelAggregator, aggregate_judge_results


# ─── MOCK RUNNER ─────────────────────────────────────────────────────────────

class MockRunner(ChatRunner):
    """Returns a fixed text response for testing."""

    def __init__(self, response_text: str):
        self._response_text = response_text

    def generate(self, messages, max_tokens=1000, temperature=0.7, seed=1234, response_format=None):
        return {
            "text": self._response_text,
            "model": "mock",
            "backend": "mock",
            "prompt_tokens": 5,
            "completion_tokens": 10,
            "total_tokens": 15,
            "latency_ms": 1,
            "finish_reason": "stop",
        }


class FailingRunner(ChatRunner):
    """Simulates an inference failure (exception during generate)."""

    def generate(self, messages, max_tokens=1000, temperature=0.7, seed=1234, response_format=None):
        raise RuntimeError("Simulated inference failure")


SAMPLE_TRACE = {
    "turns": [
        {"role": "interviewer", "text": "Tell me about a challenge your team faced."},
        {"role": "participant", "text": "We had a major server outage that lasted 3 days."},
        {"role": "interviewer", "text": "What caused the outage?"},
        {"role": "participant", "text": "It was a cascading failure in our load balancers."},
    ],
    "turn_count": 2,
    "completeness": 0.4,
}


# ─── sanitize_trace ───────────────────────────────────────────────────────────

def test_sanitize_removes_system_turns():
    trace = {
        "turns": [
            {"role": "interviewer", "text": "Hello"},
            {"role": "participant", "text": "World"},
            {"role": "system", "text": "Interview done"},
        ]
    }
    result = sanitize_trace(trace)
    assert len(result) == 2
    assert all(t["role"] in ("interviewer", "participant") for t in result)


def test_sanitize_strips_metadata():
    trace = {
        "turns": [
            {"role": "interviewer", "text": "Question?", "generation_latency_ms": 1234, "model_id": "secret_model"},
            {"role": "participant", "text": "Answer.", "extraction": [{"field": "x", "value": "y"}]},
        ]
    }
    result = sanitize_trace(trace)
    assert "generation_latency_ms" not in result[0]
    assert "model_id" not in result[0]
    assert "extraction" not in result[1]
    assert result[0]["text"] == "Question?"


def test_sanitize_preserves_text():
    result = sanitize_trace(SAMPLE_TRACE)
    assert result[0]["text"] == "Tell me about a challenge your team faced."


def test_sanitize_candidate_model_absent():
    """Candidate model identifiers must not appear in sanitized output."""
    trace = {
        "turns": [
            {"role": "interviewer", "text": "Q?", "model_id": "tinyllama_1.1b",
             "target_field": "root_cause"},
        ]
    }
    result = sanitize_trace(trace)
    for turn in result:
        assert "model_id" not in turn
        assert "target_field" not in turn


# ─── _scrub_json ──────────────────────────────────────────────────────────────

def test_scrub_json_clean_object():
    raw = '{"follow_up_question_quality": 3}'
    assert _scrub_json(raw) == raw


def test_scrub_json_extracts_from_prose():
    raw = 'Here is my assessment: {"follow_up_question_quality": 2, "supporting_evidence": "see Q3"} Thanks.'
    result = _scrub_json(raw)
    parsed = json.loads(result)
    assert parsed["follow_up_question_quality"] == 2


# ─── LLMJudge: valid scores ───────────────────────────────────────────────────

def test_judge_accepts_valid_score_3():
    resp = json.dumps({"follow_up_question_quality": 3, "supporting_evidence": "Q2 was well-targeted."})
    judge = LLMJudge("test_judge", MockRunner(resp))
    result = judge.evaluate_interview(SAMPLE_TRACE)
    assert result["success"] is True
    assert result["score"] == 3
    assert result["failure_type"] is None
    assert result["metric_version"] == METRIC_VERSION


def test_judge_accepts_all_valid_scores():
    for score in [0, 1, 2, 3, 4]:
        resp = json.dumps({"follow_up_question_quality": score, "supporting_evidence": "test"})
        judge = LLMJudge("j", MockRunner(resp))
        result = judge.evaluate_interview(SAMPLE_TRACE)
        assert result["success"] is True
        assert result["score"] == score


def test_judge_id_in_result():
    resp = json.dumps({"follow_up_question_quality": 2, "supporting_evidence": "ok"})
    judge = LLMJudge("my_judge_123", MockRunner(resp))
    result = judge.evaluate_interview(SAMPLE_TRACE)
    assert result["judge_id"] == "my_judge_123"


def test_judge_latency_ms_recorded():
    resp = json.dumps({"follow_up_question_quality": 2, "supporting_evidence": "ok"})
    judge = LLMJudge("j", MockRunner(resp))
    result = judge.evaluate_interview(SAMPLE_TRACE)
    assert isinstance(result["latency_ms"], int)
    assert result["latency_ms"] >= 0


# ─── LLMJudge: invalid scores (no defaults) ──────────────────────────────────

def test_judge_rejects_missing_key():
    """Missing required key must produce failure, not default score."""
    resp = json.dumps({"other_key": 3})
    judge = LLMJudge("j", MockRunner(resp))
    result = judge.evaluate_interview(SAMPLE_TRACE)
    assert result["success"] is False
    assert result["score"] is None
    assert result["failure_type"] == FAILURE_PARSE


def test_judge_rejects_bad_json():
    judge = LLMJudge("j", MockRunner("This is not JSON at all!"))
    result = judge.evaluate_interview(SAMPLE_TRACE)
    assert result["success"] is False
    assert result["score"] is None
    assert result["failure_type"] == FAILURE_PARSE


def test_judge_rejects_out_of_range_score_5():
    resp = json.dumps({"follow_up_question_quality": 5, "supporting_evidence": "x"})
    judge = LLMJudge("j", MockRunner(resp))
    result = judge.evaluate_interview(SAMPLE_TRACE)
    assert result["success"] is False
    assert result["score"] is None
    assert result["failure_type"] == FAILURE_INVALID


def test_judge_rejects_negative_score():
    resp = json.dumps({"follow_up_question_quality": -1, "supporting_evidence": "x"})
    judge = LLMJudge("j", MockRunner(resp))
    result = judge.evaluate_interview(SAMPLE_TRACE)
    assert result["success"] is False
    assert result["score"] is None
    assert result["failure_type"] == FAILURE_INVALID


def test_judge_rejects_float_score():
    """Float scores (e.g. 2.5) are not valid on the ordinal scale."""
    resp = json.dumps({"follow_up_question_quality": 2.5, "supporting_evidence": "x"})
    judge = LLMJudge("j", MockRunner(resp))
    result = judge.evaluate_interview(SAMPLE_TRACE)
    assert result["success"] is False
    assert result["failure_type"] == FAILURE_INVALID


def test_judge_rejects_nan_score():
    resp = '{"follow_up_question_quality": NaN, "supporting_evidence": "x"}'
    judge = LLMJudge("j", MockRunner(resp))
    result = judge.evaluate_interview(SAMPLE_TRACE)
    # NaN is not valid JSON so this triggers FAILURE_PARSE before FAILURE_INVALID
    assert result["success"] is False
    assert result["score"] is None


def test_judge_rejects_string_score():
    resp = json.dumps({"follow_up_question_quality": "good", "supporting_evidence": "x"})
    judge = LLMJudge("j", MockRunner(resp))
    result = judge.evaluate_interview(SAMPLE_TRACE)
    assert result["success"] is False
    assert result["failure_type"] == FAILURE_INVALID


def test_judge_inference_failure_distinct_from_parse_failure():
    """Inference errors (runner exception) produce FAILURE_INFERENCE, not FAILURE_PARSE."""
    judge = LLMJudge("j", FailingRunner())
    result = judge.evaluate_interview(SAMPLE_TRACE)
    assert result["success"] is False
    assert result["failure_type"] == FAILURE_INFERENCE
    assert result["score"] is None


def test_judge_no_midpoint_default_for_partial_json():
    """Partial JSON that is missing the required key must NOT get score=2 as default."""
    partial = json.dumps({"supporting_evidence": "some evidence without score"})
    judge = LLMJudge("j", MockRunner(partial))
    result = judge.evaluate_interview(SAMPLE_TRACE)
    assert result["score"] is None
    assert result["success"] is False


# ─── aggregate_judge_results ──────────────────────────────────────────────────

def _make_result(score=None, success=True, failure_type=None):
    return {
        "success": success,
        "score": score,
        "failure_type": failure_type,
        "failure_reason": None if success else "test failure",
        "judge_id": "test",
        "metric_version": METRIC_VERSION,
        "supporting_evidence": "",
        "latency_ms": 1,
    }


def test_aggregate_three_valid_judges_median():
    results = [_make_result(2), _make_result(3), _make_result(4)]
    agg = aggregate_judge_results(results)
    assert agg["successful_judges"] == 3
    assert agg["panel_median"] == 3.0
    assert agg["status"] == "ok"


def test_aggregate_single_valid_judge_null_mad():
    """One valid judge → MAD not estimable → should be None, not 0.0."""
    results = [_make_result(2), _make_result(success=False, failure_type=FAILURE_PARSE)]
    agg = aggregate_judge_results(results)
    assert agg["successful_judges"] == 1
    assert agg["panel_median"] == 2.0
    assert agg["disagreement_mad"] is None  # NOT 0.0
    assert "single_valid_score" in agg["status"]


def test_aggregate_no_valid_judges():
    results = [
        _make_result(success=False, failure_type=FAILURE_PARSE),
        _make_result(success=False, failure_type=FAILURE_INFERENCE),
    ]
    agg = aggregate_judge_results(results)
    assert agg["panel_median"] is None
    assert agg["disagreement_mad"] is None
    assert agg["successful_judges"] == 0
    assert agg["status"] == "no_valid_scores"


def test_aggregate_failed_scores_excluded_from_median():
    results = [_make_result(4), _make_result(0), _make_result(success=False, failure_type=FAILURE_PARSE)]
    agg = aggregate_judge_results(results)
    # Median of [4, 0] = 2.0; failed result must not be included
    assert agg["successful_judges"] == 2
    assert agg["panel_median"] == 2.0


def test_aggregate_mad_two_judges():
    results = [_make_result(2), _make_result(4)]
    agg = aggregate_judge_results(results)
    # Median = 3, deviations = [1, 1], MAD = 1.0
    assert agg["panel_median"] == 3.0
    assert agg["disagreement_mad"] == pytest.approx(1.0)


def test_aggregate_perfect_agreement_zero_mad():
    results = [_make_result(3), _make_result(3), _make_result(3)]
    agg = aggregate_judge_results(results)
    assert agg["disagreement_mad"] == pytest.approx(0.0)


# ─── PanelAggregator class ───────────────────────────────────────────────────

def test_panel_aggregator_uses_shared_function():
    resp = json.dumps({"follow_up_question_quality": 3, "supporting_evidence": "ok"})
    judges = [
        LLMJudge("j1", MockRunner(resp)),
        LLMJudge("j2", MockRunner(resp)),
    ]
    panel = PanelAggregator(judges)
    result = panel.evaluate_with_panel(SAMPLE_TRACE)
    assert result["successful_judges"] == 2
    assert result["panel_median"] == 3.0
    assert "judge_responses" in result


def test_panel_all_failed_returns_null_panel():
    judges = [
        LLMJudge("j1", MockRunner("not json")),
        LLMJudge("j2", MockRunner("also not json")),
    ]
    panel = PanelAggregator(judges)
    result = panel.evaluate_with_panel(SAMPLE_TRACE)
    assert result["panel_median"] is None
    assert result["disagreement_mad"] is None
    assert result["successful_judges"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
