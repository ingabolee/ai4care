"""
tests/test_engine.py -- Unit tests for core engine logic (no LLM required).
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine import (
    _empty_knowledge_state,
    _empty_field_state,
    apply_extraction_update,
    _values_agree,
    schema_completeness,
    select_next_gap,
    _scrub_json,
    _sentence_level_fallback,
    SCHEMA_FIELDS,
)


# ─── Unit: _values_agree ────────────────────────────────────────────────────

def test_values_agree_identical():
    assert _values_agree("system failure due to overload", "system failure due to overload") is True


def test_values_agree_partial():
    assert _values_agree("network outage in data center", "network outage crash") is True


def test_values_agree_disagreement():
    assert _values_agree("CPU bottleneck", "storage disk failure") is False


def test_values_agree_empty():
    assert _values_agree(None, "anything") is False
    assert _values_agree("", "something") is False


# ─── Unit: apply_extraction_update ──────────────────────────────────────────

def test_unknown_to_candidate():
    state = _empty_knowledge_state()
    state = apply_extraction_update(state, {
        "field": "primary_challenge",
        "value": "server overload",
        "evidence": "The server was overloaded",
        "confidence": 0.8,
    }, turn_id=1)
    assert state["primary_challenge"]["status"] == "candidate"
    assert state["primary_challenge"]["value"] == "server overload"
    assert state["primary_challenge"]["support_count"] == 1


def test_candidate_to_supported_on_agreement():
    state = _empty_knowledge_state()
    state = apply_extraction_update(state, {
        "field": "primary_challenge", "value": "server overload", "evidence": "e1", "confidence": 0.7,
    }, turn_id=1)
    state = apply_extraction_update(state, {
        "field": "primary_challenge", "value": "server overload problem", "evidence": "e2", "confidence": 0.8,
    }, turn_id=2)
    assert state["primary_challenge"]["status"] == "supported"
    assert state["primary_challenge"]["support_count"] == 2


def test_candidate_to_conflicting_on_disagreement():
    state = _empty_knowledge_state()
    state = apply_extraction_update(state, {
        "field": "root_cause", "value": "hardware failure", "evidence": "e1", "confidence": 0.7,
    }, turn_id=1)
    state = apply_extraction_update(state, {
        "field": "root_cause", "value": "software bug in authentication", "evidence": "e2", "confidence": 0.7,
    }, turn_id=2)
    assert state["root_cause"]["status"] == "conflicting"
    assert len(state["root_cause"]["conflicts"]) == 1


def test_supported_to_verified_at_three_supports():
    state = _empty_knowledge_state()
    # 1st update: unknown -> candidate
    state = apply_extraction_update(state, {
        "field": "domain_area", "value": "healthcare IT", "evidence": "e1", "confidence": 0.8,
    }, turn_id=1)
    # 2nd update: candidate -> supported
    state = apply_extraction_update(state, {
        "field": "domain_area", "value": "healthcare IT systems", "evidence": "e2", "confidence": 0.8,
    }, turn_id=2)
    # 3rd update: two supports means support_count >= 3 -> verified
    state = apply_extraction_update(state, {
        "field": "domain_area", "value": "healthcare IT services", "evidence": "e3", "confidence": 0.9,
    }, turn_id=3)
    assert state["domain_area"]["status"] == "verified"


def test_unknown_field_ignored():
    state = _empty_knowledge_state()
    original = dict(state)
    state = apply_extraction_update(state, {
        "field": "nonexistent_field", "value": "anything", "evidence": "e1", "confidence": 0.5,
    }, turn_id=1)
    assert state == original


# ─── Unit: schema_completeness ──────────────────────────────────────────────

def test_completeness_empty():
    state = _empty_knowledge_state()
    assert schema_completeness(state) == 0.0


def test_completeness_all_candidate():
    state = _empty_knowledge_state()
    for field in SCHEMA_FIELDS:
        state[field]["status"] = "candidate"
    # all candidate = 1 point each / 3 max = 1/3
    expected = round(len(SCHEMA_FIELDS) / (len(SCHEMA_FIELDS) * 3), 4)
    assert schema_completeness(state) == pytest.approx(expected)


def test_completeness_all_verified():
    state = _empty_knowledge_state()
    for field in SCHEMA_FIELDS:
        state[field]["status"] = "verified"
    assert schema_completeness(state) == 1.0


# ─── Unit: select_next_gap ──────────────────────────────────────────────────

def test_gap_prefers_conflicting_over_unknown():
    state = _empty_knowledge_state()
    state["root_cause"]["status"] = "conflicting"
    state["domain_area"]["status"] = "unknown"
    gap = select_next_gap(state)
    assert gap["field"] == "root_cause"
    assert gap["status"] == "conflicting"


def test_gap_prefers_unknown_over_candidate():
    state = _empty_knowledge_state()
    for field in SCHEMA_FIELDS:
        state[field]["status"] = "candidate"
    state["lessons_learned"]["status"] = "unknown"
    gap = select_next_gap(state)
    assert gap["field"] == "lessons_learned"


def test_gap_returns_none_when_all_verified():
    state = _empty_knowledge_state()
    for field in SCHEMA_FIELDS:
        state[field]["status"] = "verified"
    gap = select_next_gap(state)
    assert gap is None


# ─── Unit: _scrub_json ──────────────────────────────────────────────────────

def test_scrub_json_clean():
    raw = '{"updates": [{"field": "domain_area", "value": "healthcare"}]}'
    assert _scrub_json(raw) == raw


def test_scrub_json_strips_markdown():
    raw = '```json\n{"updates": []}\n```'
    result = _scrub_json(raw)
    assert result == '{"updates": []}'


def test_scrub_json_extracts_nested():
    raw = 'Here is the JSON:\n{"key": "val"}\nHope this helps!'
    assert _scrub_json(raw) == '{"key": "val"}'


# ─── Unit: _sentence_level_fallback ─────────────────────────────────────────

def test_fallback_finds_challenge():
    text = "Our team struggled with a major system outage that lasted two days."
    updates = _sentence_level_fallback(text, turn_id=1)
    fields = [u["field"] for u in updates]
    assert "primary_challenge" in fields


def test_fallback_short_sentences_ignored():
    text = "OK. Yes."
    updates = _sentence_level_fallback(text, turn_id=1)
    assert updates == []


def test_fallback_no_double_field():
    text = (
        "The team had a major challenge with downtime. "
        "The problem was caused by a network issue."
    )
    updates = _sentence_level_fallback(text, turn_id=1)
    fields = [u["field"] for u in updates]
    # Each field should appear at most once
    assert len(fields) == len(set(fields))
