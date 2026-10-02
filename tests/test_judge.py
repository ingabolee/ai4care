"""
tests/test_judge.py -- Unit tests for LLMJudge and PanelAggregator (no real LLM required).
Uses mock ChatRunner to test parsing and aggregation logic.
"""

import pytest
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from inference import ChatRunner
from llm_judge import LLMJudge, METRICS, _scrub_json, sanitize_trace, format_transcript_text
from panel_aggregator import PanelAggregator


# ─── MOCK RUNNER ─────────────────────────────────────────────────────────────

class MockRunner(ChatRunner):
    """Returns a fixed JSON response for testing."""

    def __init__(self, response_text: str):
        self._response_text = response_text

    def generate(self, messages, max_tokens=1000, temperature=0.7, seed=1234, response_format=None):
        return {"text": self._response_text, "model": "mock", "backend": "mock",
                "prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15,
                "latency_ms": 1, "finish_reason": "stop"}


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


# ─── Unit: sanitize_trace ────────────────────────────────────────────────────

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


# ─── Unit: _scrub_json ───────────────────────────────────────────────────────

def test_scrub_json_clean_object():
    raw = '{"turn_level_completeness_efficiency": 8.0}'
    assert _scrub_json(raw) == raw


def test_scrub_json_extract_from_prose():
    raw = 'Here are my ratings: {"turn_level_completeness_efficiency": 7.5, "redundancy_penalty": 1.0} Done!'
    result = _scrub_json(raw)
    parsed = json.loads(result)
    assert parsed["turn_level_completeness_efficiency"] == 7.5


# ─── Unit: LLMJudge.evaluate_interview ───────────────────────────────────────

def test_judge_parses_valid_json():
    mock_scores = {m: 7.0 for m in METRICS}
    runner = MockRunner(json.dumps(mock_scores))
    judge = LLMJudge(judge_id="test_judge", runner=runner)
    result = judge.evaluate_interview(SAMPLE_TRACE)

    assert result["success"] is True
    assert result["error"] is None
    for m in METRICS:
        assert m in result["metrics"]
        assert result["metrics"][m] == 7.0


def test_judge_falls_back_to_defaults_on_bad_json():
    runner = MockRunner("This is not JSON at all, sorry!")
    judge = LLMJudge(judge_id="bad_judge", runner=runner)
    result = judge.evaluate_interview(SAMPLE_TRACE)
    assert result["success"] is False
    assert result["error"] is not None


def test_judge_handles_partial_json():
    # Only some metrics present
    partial = {"turn_level_completeness_efficiency": 9.0, "redundancy_penalty": 0.5}
    runner = MockRunner(json.dumps(partial))
    judge = LLMJudge(judge_id="partial_judge", runner=runner)
    result = judge.evaluate_interview(SAMPLE_TRACE)
    assert result["success"] is True
    # Missing metrics get default 5.0
    assert result["metrics"]["participant_engagement"] == 5.0
    assert result["metrics"]["turn_level_completeness_efficiency"] == 9.0


def test_judge_id_included_in_result():
    mock_scores = {m: 6.0 for m in METRICS}
    runner = MockRunner(json.dumps(mock_scores))
    judge = LLMJudge(judge_id="my_judge", runner=runner)
    result = judge.evaluate_interview(SAMPLE_TRACE)
    assert result["judge_id"] == "my_judge"


# ─── Unit: PanelAggregator ───────────────────────────────────────────────────

def test_panel_takes_median():
    scores_1 = {m: 6.0 for m in METRICS}
    scores_2 = {m: 8.0 for m in METRICS}
    scores_3 = {m: 10.0 for m in METRICS}

    judges = [
        LLMJudge("j1", MockRunner(json.dumps(scores_1))),
        LLMJudge("j2", MockRunner(json.dumps(scores_2))),
        LLMJudge("j3", MockRunner(json.dumps(scores_3))),
    ]
    panel = PanelAggregator(judges)
    result = panel.evaluate_with_panel(SAMPLE_TRACE)

    assert result["total_judges"] == 3
    assert result["successful_judges"] == 3
    for m in METRICS:
        # Median of [6, 8, 10] = 8
        assert result["aggregated_metrics"][m] == pytest.approx(8.0)


def test_panel_handles_one_failed_judge():
    good_scores = {m: 7.0 for m in METRICS}
    judges = [
        LLMJudge("good", MockRunner(json.dumps(good_scores))),
        LLMJudge("bad", MockRunner("not json")),
    ]
    panel = PanelAggregator(judges)
    result = panel.evaluate_with_panel(SAMPLE_TRACE)

    # Should still aggregate from the 1 good judge
    assert result["successful_judges"] == 1
    for m in METRICS:
        # Only one valid score so median = 7.0
        assert result["aggregated_metrics"][m] == pytest.approx(7.0)


def test_panel_all_failed_returns_none_metrics():
    judges = [
        LLMJudge("j1", MockRunner("not json")),
        LLMJudge("j2", MockRunner("also bad")),
    ]
    panel = PanelAggregator(judges)
    result = panel.evaluate_with_panel(SAMPLE_TRACE)
    for m in METRICS:
        assert result["aggregated_metrics"][m] is None
