"""
tests/test_evaluation_metrics.py -- Unit tests for the production evaluation_metrics module.

All tests use small constructed examples with known deterministic results.
These tests verify implementation behaviour, NOT semantic truth or generalisation.

Run with:
    python -m pytest tests/test_evaluation_metrics.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation_metrics import (
    DEFAULT_MATCH_THRESHOLD,
    EVALUATION_SCHEMA_VERSION,
    MATCHING_ALGORITHM,
    SENSITIVITY_THRESHOLDS,
    _bipartite_match,
    _normalise,
    _overlap_coefficient,
    _split_into_claims,
    compute_all_metrics,
    compute_conflicting_field_rate,
    compute_evidence_presence_rate,
    compute_gap_targeting_rate,
    compute_novel_fields,
    compute_redundancy_rate,
    compute_reference_claim_metrics,
    compute_schema_completeness,
    sensitivity_sweep,
)


# ─── Helper fixtures ──────────────────────────────────────────────────────────

def _make_ref(case_id: str, texts: list[str]) -> list[dict]:
    return [{"fact_id": f"{case_id}_f{i}", "case_id": case_id, "text": t}
            for i, t in enumerate(texts)]


def _make_atomic(fields_values: list[tuple[str, str]]) -> list[dict]:
    return [{"field": f, "value": v} for f, v in fields_values]


def _make_ks(status_map: dict[str, str]) -> dict:
    """Build a minimal knowledge state from field→status mapping."""
    ks = {}
    for field, status in status_map.items():
        ks[field] = {
            "field": field,
            "status": status,
            "value": "some value" if status != "unknown" else None,
            "evidence": [{"text": "evidence"}] if status not in ("unknown", "candidate") else [],
            "confidence": 0.8 if status != "unknown" else 0.0,
            "conflicts": [],
        }
    return ks


# ─── Text normalisation ───────────────────────────────────────────────────────

class TestNormalise:
    def test_lowercases(self):
        assert "hello" in _normalise("HELLO")

    def test_strips_punctuation(self):
        tokens = _normalise("hello, world!")
        assert "hello" in tokens and "world" in tokens

    def test_returns_set(self):
        assert isinstance(_normalise("hello"), set)

    def test_empty_string(self):
        assert _normalise("") == set()

    def test_deduplicates_tokens(self):
        tokens = _normalise("the the the")
        assert tokens == {"the"}


# ─── Overlap coefficient ──────────────────────────────────────────────────────

class TestOverlapCoefficient:
    def test_identical_sets(self):
        s = {"a", "b", "c"}
        assert _overlap_coefficient(s, s) == 1.0

    def test_disjoint_sets(self):
        assert _overlap_coefficient({"a"}, {"b"}) == 0.0

    def test_partial_overlap(self):
        score = _overlap_coefficient({"a", "b", "c"}, {"b", "c", "d"})
        # shared = {b,c}=2, max(3,3)=3 → 2/3
        assert abs(score - 2 / 3) < 1e-9

    def test_empty_set_returns_zero(self):
        assert _overlap_coefficient(set(), {"a"}) == 0.0
        assert _overlap_coefficient({"a"}, set()) == 0.0


# ─── Claim splitting ──────────────────────────────────────────────────────────

class TestSplitIntoClaims:
    def test_single_sentence(self):
        claims = _split_into_claims("The system failed due to overload.")
        assert len(claims) >= 1

    def test_multi_sentence(self):
        text = "The system failed. The team resolved it quickly. Lessons were learned."
        claims = _split_into_claims(text)
        assert len(claims) >= 2

    def test_empty_string(self):
        assert _split_into_claims("") == []

    def test_short_fragments_filtered(self):
        # "Hi." is fewer than 3 tokens and should be filtered
        claims = _split_into_claims("Hi. The main challenge was database overload due to volume.")
        assert all(len(c.split()) >= 3 for c in claims)


# ─── Bipartite matching ───────────────────────────────────────────────────────

class TestBipartiteMatch:
    def test_exact_match(self):
        claims = ["the system failed due to overload"]
        refs = ["the system failed due to overload"]
        n, pairs = _bipartite_match(claims, refs, threshold=0.5)
        assert n == 1
        assert len(pairs) == 1

    def test_unrelated_no_match(self):
        claims = ["purple elephant quantum bicycle"]
        refs = ["the system experienced a major database outage"]
        n, pairs = _bipartite_match(claims, refs, threshold=0.5)
        assert n == 0

    def test_one_to_one_constraint(self):
        # One claim should not match multiple references
        claims = ["the system failed"]
        refs = ["the system failed", "the system failed completely"]
        n, pairs = _bipartite_match(claims, refs, threshold=0.5)
        assert n <= 1
        # Each claim can match at most one ref
        claim_indices = [p[0] for p in pairs]
        assert len(claim_indices) == len(set(claim_indices))

    def test_duplicate_claim_not_double_counted(self):
        claims = ["system failure", "system failure"]
        refs = ["system failure event occurred"]
        n, pairs = _bipartite_match(claims, refs, threshold=0.5)
        # Each ref can only be matched once
        ref_indices = [p[1] for p in pairs]
        assert len(ref_indices) == len(set(ref_indices))

    def test_multiple_claims_multiple_refs(self):
        claims = ["database overload caused outage", "team migrated to cloud infrastructure"]
        refs = ["database overload event caused the system outage", "team successfully migrated to cloud"]
        n, pairs = _bipartite_match(claims, refs, threshold=0.3)
        assert n == 2  # Both should match at this threshold

    def test_threshold_boundary_at_exactly_threshold(self):
        # Create a pair with known overlap score exactly at threshold
        # claims = ["a b c"], refs = ["a b d e f"] → shared={a,b}, max=5 → 2/5=0.4
        claims = ["a b c"]
        refs = ["a b d e f"]
        n_below, _ = _bipartite_match(claims, refs, threshold=0.5)    # 0.4 < 0.5
        n_at, _ = _bipartite_match(claims, refs, threshold=0.4)       # 0.4 >= 0.4
        assert n_below == 0
        assert n_at == 1

    def test_deterministic_across_calls(self):
        claims = ["a b c", "d e f"]
        refs = ["a b c x", "d e f y"]
        n1, pairs1 = _bipartite_match(claims, refs, threshold=0.4)
        n2, pairs2 = _bipartite_match(claims, refs, threshold=0.4)
        assert n1 == n2
        assert pairs1 == pairs2


# ─── Reference-claim metrics ──────────────────────────────────────────────────

class TestReferenceClaimMetrics:
    CASE_ID = "case_test01"
    CASE_ID_OTHER = "case_other99"

    def test_exact_match_perfect_scores(self):
        refs = _make_ref(self.CASE_ID, ["the system experienced a major database failure"])
        atoms = _make_atomic([("primary_challenge", "the system experienced a major database failure")])
        m = compute_reference_claim_metrics(atoms, refs, self.CASE_ID, threshold=0.5)
        assert m["reference_fact_f1"] == pytest.approx(1.0, abs=0.01)
        assert m["status"] == "ok"

    def test_unrelated_claims_zero_f1(self):
        refs = _make_ref(self.CASE_ID, ["system experienced database failure"])
        atoms = _make_atomic([("domain_area", "quantum cheese purple bicycle nonsense")])
        m = compute_reference_claim_metrics(atoms, refs, self.CASE_ID, threshold=0.5)
        assert m["reference_fact_f1"] == pytest.approx(0.0)
        assert m["matched_pair_count"] == 0

    def test_empty_extraction_gives_zero_recall_null_precision(self):
        refs = _make_ref(self.CASE_ID, ["the system had a database outage"])
        m = compute_reference_claim_metrics([], refs, self.CASE_ID, threshold=0.5)
        assert m["reference_fact_recall"] == pytest.approx(0.0)
        assert m["reference_fact_precision"] is None
        assert m["reference_fact_f1"] == pytest.approx(0.0)
        assert m["status"] == "no_extracted_claims"

    def test_empty_reference_facts_returns_null(self):
        atoms = _make_atomic([("domain_area", "healthcare database systems")])
        m = compute_reference_claim_metrics(atoms, [], self.CASE_ID, threshold=0.5)
        assert m["reference_fact_f1"] is None
        assert m["reference_fact_precision"] is None
        assert m["reference_fact_recall"] is None
        assert m["status"] == "no_reference_facts"

    def test_none_reference_facts_returns_null(self):
        atoms = _make_atomic([("domain_area", "healthcare database systems")])
        m = compute_reference_claim_metrics(atoms, None or [], self.CASE_ID, threshold=0.5)
        assert m["reference_fact_f1"] is None

    def test_cross_case_isolation(self):
        # Reference facts from another case must not match
        refs = _make_ref(self.CASE_ID_OTHER, ["the system experienced a major database failure"])
        atoms = _make_atomic([("primary_challenge", "the system experienced a major database failure")])
        m = compute_reference_claim_metrics(atoms, refs, self.CASE_ID, threshold=0.5)
        assert m["status"] in ("no_reference_facts", "no_reference_facts_for_case")
        assert m["reference_fact_f1"] is None

    def test_precision_recall_f1_consistency(self):
        refs = _make_ref(self.CASE_ID, [
            "database system failed due to overload",
            "team decided to implement cloud migration",
        ])
        atoms = _make_atomic([
            ("primary_challenge", "database system failed due to overload"),
            ("key_decisions", "cloud migration was adopted by the team"),
            ("domain_area", "completely unrelated purple elephant fact"),
        ])
        m = compute_reference_claim_metrics(atoms, refs, self.CASE_ID, threshold=0.4)
        p = m["reference_fact_precision"]
        r = m["reference_fact_recall"]
        f1 = m["reference_fact_f1"]
        if p is not None and p + r > 0:
            expected_f1 = 2 * p * r / (p + r)
            assert abs(f1 - expected_f1) < 1e-5

    def test_duplicate_claims_not_inflated(self):
        refs = _make_ref(self.CASE_ID, ["database failure caused system outage"])
        atoms = _make_atomic([
            ("primary_challenge", "database failure caused system outage"),
            ("root_cause", "database failure caused system outage"),  # duplicate
        ])
        # Deduplication by (field, value.lower) removes one
        m = compute_reference_claim_metrics(atoms, refs, self.CASE_ID, threshold=0.5)
        assert m["matched_pair_count"] == 1

    def test_matching_metadata_recorded(self):
        refs = _make_ref(self.CASE_ID, ["anything"])
        atoms = _make_atomic([("domain_area", "anything")])
        m = compute_reference_claim_metrics(atoms, refs, self.CASE_ID)
        assert m["matching_algorithm"] == MATCHING_ALGORITHM
        assert m["matching_threshold"] == DEFAULT_MATCH_THRESHOLD


# ─── Sensitivity sweep ────────────────────────────────────────────────────────

class TestSensitivitySweep:
    CASE_ID = "case_sens01"

    def test_returns_all_three_thresholds(self):
        refs = _make_ref(self.CASE_ID, ["the system failed"])
        atoms = _make_atomic([("domain_area", "the system failed")])
        result = sensitivity_sweep(atoms, refs, self.CASE_ID)
        for t in SENSITIVITY_THRESHOLDS:
            assert t in result

    def test_reproducible(self):
        refs = _make_ref(self.CASE_ID, ["the system experienced major failure events"])
        atoms = _make_atomic([("domain_area", "major failure events in the system")])
        r1 = sensitivity_sweep(atoms, refs, self.CASE_ID)
        r2 = sensitivity_sweep(atoms, refs, self.CASE_ID)
        for t in SENSITIVITY_THRESHOLDS:
            assert r1[t]["reference_fact_f1"] == r2[t]["reference_fact_f1"]

    def test_lower_threshold_not_lower_recall(self):
        # With a lower threshold, more matches are possible → recall should be >=
        refs = _make_ref(self.CASE_ID, ["system database failure event occurred"])
        atoms = _make_atomic([("primary_challenge", "system database failure occurred")])
        result = sensitivity_sweep(atoms, refs, self.CASE_ID)
        recall_025 = result[0.25]["reference_fact_recall"] or 0
        recall_075 = result[0.75]["reference_fact_recall"] or 0
        assert recall_025 >= recall_075


# ─── Schema completeness ──────────────────────────────────────────────────────

class TestSchemaCompleteness:
    def test_all_unknown_is_zero(self):
        ks = {f: {"status": "unknown"} for f in [
            "domain_area", "primary_challenge", "root_cause", "key_decisions",
            "technologies_used", "people_involved", "processes_affected",
            "operational_impact", "lessons_learned"
        ]}
        assert compute_schema_completeness(ks) == 0.0

    def test_all_verified_is_one(self):
        ks = {f: {"status": "verified"} for f in [
            "domain_area", "primary_challenge", "root_cause", "key_decisions",
            "technologies_used", "people_involved", "processes_affected",
            "operational_impact", "lessons_learned"
        ]}
        assert compute_schema_completeness(ks) == pytest.approx(1.0)

    def test_conflicting_gives_partial_credit(self):
        ks = {"domain_area": {"status": "conflicting"}}
        score = compute_schema_completeness(ks)
        assert 0 < score < 1.0 / 9


# ─── Novel fields ─────────────────────────────────────────────────────────────

class TestNovelFields:
    def test_counts_newly_surfaced(self):
        turns = [
            {"role": "participant", "newly_surfaced_fields": ["root_cause", "domain_area"], "novel_facts_count": 2},
            {"role": "participant", "newly_surfaced_fields": ["technologies_used"], "novel_facts_count": 1},
            {"role": "interviewer", "text": "Q?", "turn_id": 1},
        ]
        result = compute_novel_fields(turns)
        assert result["novel_fields_total"] == 3
        assert result["n_participant_turns"] == 2
        assert result["novel_fields_per_response"] == pytest.approx(1.5)

    def test_zero_participant_turns(self):
        turns = [{"role": "interviewer", "text": "Opening", "turn_id": 0}]
        result = compute_novel_fields(turns)
        assert result["novel_fields_total"] == 0
        assert result["novel_fields_per_response"] == 0.0


# ─── Redundancy rate ──────────────────────────────────────────────────────────

class TestRedundancyRate:
    def test_no_followup_questions_returns_none(self):
        turns = [{"role": "interviewer", "text": "Opening", "turn_id": 0}]
        result = compute_redundancy_rate(turns)
        assert result["question_redundancy_rate"] is None

    def test_counts_redundant_flag(self):
        turns = [
            {"role": "interviewer", "turn_id": 0, "text": "Opening"},
            {"role": "interviewer", "turn_id": 1, "text": "Q1", "redundant": True},
            {"role": "interviewer", "turn_id": 1, "text": "Q2", "redundant": False},
            {"role": "interviewer", "turn_id": 2, "text": "Q3", "redundant": False},
        ]
        result = compute_redundancy_rate(turns)
        assert result["redundant_count"] == 1
        assert result["eligible_question_count"] == 3
        assert result["question_redundancy_rate"] == pytest.approx(1 / 3)

    def test_missing_redundant_flag_counts_as_non_redundant(self):
        turns = [
            {"role": "interviewer", "turn_id": 0, "text": "Opening"},
            {"role": "interviewer", "turn_id": 1, "text": "Q1"},  # no 'redundant' key
        ]
        result = compute_redundancy_rate(turns)
        assert result["redundant_count"] == 0
        assert result["question_redundancy_rate"] == pytest.approx(0.0)


# ─── Gap targeting rate ───────────────────────────────────────────────────────

class TestGapTargetingRate:
    def test_state_machine_ablation_returns_null(self):
        result = compute_gap_targeting_rate([], ablation="state_machine")
        assert result["gap_targeting_rate"] is None
        assert "state_machine" in result["status"]

    def test_no_followup_questions_returns_null(self):
        turns = [{"role": "interviewer", "turn_id": 0, "text": "Opening"}]
        result = compute_gap_targeting_rate(turns, ablation="none")
        assert result["gap_targeting_rate"] is None

    def test_aligned_question_counted(self):
        turns = [
            {"role": "interviewer", "turn_id": 0, "text": "Opening"},
            {
                "role": "interviewer",
                "turn_id": 1,
                "text": "Q1",
                "target_field": "root_cause",
                "knowledge_gap_targeted": "root_cause",
                "knowledge_state_before_question": {
                    "root_cause": {"status": "unknown"},
                },
            },
        ]
        result = compute_gap_targeting_rate(turns)
        assert result["aligned_count"] == 1
        assert result["gap_targeting_rate"] == pytest.approx(1.0)

    def test_already_verified_field_not_counted_as_aligned(self):
        turns = [
            {"role": "interviewer", "turn_id": 0, "text": "Opening"},
            {
                "role": "interviewer",
                "turn_id": 1,
                "text": "Q1",
                "target_field": "root_cause",
                "knowledge_gap_targeted": "root_cause",
                "knowledge_state_before_question": {
                    "root_cause": {"status": "verified"},  # already resolved
                },
            },
        ]
        result = compute_gap_targeting_rate(turns)
        assert result["aligned_count"] == 0

    def test_missing_target_field_included_in_denominator(self):
        turns = [
            {"role": "interviewer", "turn_id": 0, "text": "Opening"},
            {"role": "interviewer", "turn_id": 1, "text": "Q1"},  # no target_field
        ]
        result = compute_gap_targeting_rate(turns)
        assert result["eligible_count"] == 1
        assert result["aligned_count"] == 0


# ─── Conflict and evidence diagnostics ───────────────────────────────────────

class TestConflictingFieldRate:
    def test_all_unknown_returns_null(self):
        ks = {"domain_area": {"status": "unknown"}}
        result = compute_conflicting_field_rate(ks)
        assert result["conflicting_field_rate"] is None

    def test_correct_fraction(self):
        ks = {
            "domain_area": {"status": "supported"},
            "root_cause": {"status": "conflicting"},
            "key_decisions": {"status": "candidate"},
        }
        result = compute_conflicting_field_rate(ks)
        # 1 conflicting out of 3 non-unknown
        assert result["conflicting_field_rate"] == pytest.approx(1 / 3)
        assert result["non_unknown_field_count"] == 3


class TestEvidencePresenceRate:
    def test_no_populated_fields_returns_null(self):
        ks = {"domain_area": {"status": "unknown", "value": None, "evidence": []}}
        result = compute_evidence_presence_rate(ks)
        assert result["evidence_presence_rate"] is None

    def test_correct_fraction(self):
        ks = {
            "domain_area": {"status": "supported", "value": "healthcare", "evidence": [{"text": "e"}]},
            "root_cause": {"status": "candidate", "value": "failure", "evidence": []},
        }
        result = compute_evidence_presence_rate(ks)
        # 1 of 2 populated fields has evidence
        assert result["evidence_presence_rate"] == pytest.approx(0.5)


# ─── Compute all metrics integration test ────────────────────────────────────

class TestComputeAllMetrics:
    CASE_ID = "case_int01"

    def _make_trace(self):
        return {
            "turns": [
                {"role": "interviewer", "turn_id": 0, "text": "Opening"},
                {
                    "role": "participant",
                    "turn_id": 1,
                    "text": "Answer",
                    "newly_surfaced_fields": ["root_cause"],
                    "novel_facts_count": 1,
                    "extraction": [],
                },
                {
                    "role": "interviewer",
                    "turn_id": 1,
                    "text": "Follow-up?",
                    "target_field": "root_cause",
                    "knowledge_gap_targeted": "root_cause",
                    "redundant": False,
                    "knowledge_state_before_question": {
                        "root_cause": {"status": "unknown"},
                    },
                },
            ],
            "knowledge_state": {
                "root_cause": {
                    "status": "candidate", "value": "system overload",
                    "evidence": [{"text": "system overload"}], "conflicts": [],
                },
                **{f: {"status": "unknown", "value": None, "evidence": [], "conflicts": []}
                   for f in ["domain_area", "primary_challenge", "key_decisions",
                             "technologies_used", "people_involved", "processes_affected",
                             "operational_impact", "lessons_learned"]}
            },
            "atomic_facts": [
                {"field": "root_cause", "value": "system overload caused the outage"},
            ],
        }

    def test_returns_required_keys(self):
        trace = self._make_trace()
        refs = _make_ref(self.CASE_ID, ["system overload caused the critical outage"])
        result = compute_all_metrics(trace, refs, self.CASE_ID)
        required = [
            "reference_fact_f1", "reference_fact_precision", "reference_fact_recall",
            "schema_completeness", "novel_fields_total", "novel_fields_per_response",
            "question_redundancy_rate", "gap_targeting_rate",
            "conflicting_field_rate", "evidence_presence_rate",
            "evaluation_schema_version",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_schema_version_recorded(self):
        trace = self._make_trace()
        result = compute_all_metrics(trace, None, self.CASE_ID)
        assert result["evaluation_schema_version"] == EVALUATION_SCHEMA_VERSION

    def test_null_refs_yields_null_f1(self):
        trace = self._make_trace()
        result = compute_all_metrics(trace, None, self.CASE_ID)
        assert result["reference_fact_f1"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
