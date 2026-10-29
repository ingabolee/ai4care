"""
evaluation_metrics.py -- Production evaluation metrics for AI4CARE.

Implements all deterministic metrics used in the evaluation pipeline.
This module is the single source of truth for metric definitions.

Metric schema version: v2
Matching algorithm:     lexical_bipartite_v1
Default threshold:      0.5  (shared unique token overlap coefficient)

IMPORTANT LIMITATIONS:
- Lexical reference-claim matching is a proxy, NOT verified factual accuracy.
- Schema and process diagnostics are not ground truth.
- Results are simulation-based and reference-based.
- Agreement among automated judges does not establish human-preference alignment
  or real-world correctness.
- No human evaluation is performed or required.
"""

from __future__ import annotations

import re
from typing import Optional

# ─── VERSION ─────────────────────────────────────────────────────────────────

EVALUATION_SCHEMA_VERSION = "v2"
MATCHING_ALGORITHM = "lexical_bipartite_v1"
DEFAULT_MATCH_THRESHOLD = 0.5   # at least half of unique tokens in the longer set must be shared
SENSITIVITY_THRESHOLDS = (0.25, 0.5, 0.75)

# Schema field statuses and their completeness weights (from engine.py)
STATUS_WEIGHTS = {"unknown": 0, "candidate": 1, "supported": 2, "verified": 3, "conflicting": 1}
SCHEMA_FIELD_COUNT = 9
MAX_COMPLETENESS_SCORE = SCHEMA_FIELD_COUNT * 3  # 9 fields × 3 points maximum


# ─── TEXT NORMALISATION ───────────────────────────────────────────────────────

def _normalise(text: str) -> set[str]:
    """Lowercase, strip punctuation, normalise whitespace, return unique token set.

    This is a deterministic transformation used in all lexical comparisons.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return set(text.split())


def _overlap_coefficient(a_tokens: set[str], b_tokens: set[str]) -> float:
    """Shared unique tokens / max(|A|, |B|).

    This is the overlap coefficient (not Jaccard, not cosine similarity).
    Returns 0.0 if either token set is empty.
    """
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(len(a_tokens), len(b_tokens))


# ─── CLAIM SPLITTING ─────────────────────────────────────────────────────────

def _split_into_claims(text: str) -> list[str]:
    """Split multi-sentence text into individual claim units using punctuation rules.

    Deterministic: uses sentence-boundary punctuation only.
    Returns at least one claim even for single-sentence text.
    """
    if not text or not text.strip():
        return []
    # Split on sentence-ending punctuation followed by whitespace or end of string
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    # Further split on semicolons used for list-style facts
    claims = []
    for part in parts:
        sub = [s.strip() for s in part.split(";") if s.strip()]
        claims.extend(sub)
    # Filter very short fragments (fewer than 3 tokens) — not informative claims
    return [c for c in claims if len(c.split()) >= 3]


# ─── BIPARTITE ONE-TO-ONE MATCHING ───────────────────────────────────────────

def _bipartite_match(
    extracted_claims: list[str],
    reference_facts: list[str],
    threshold: float,
) -> tuple[int, list[tuple[int, int, float]]]:
    """Find maximum-cardinality one-to-one matching between claims and facts.

    Each extracted claim and each reference fact participates in at most one match.
    Pairs are qualified if their overlap coefficient >= threshold.

    Tie-breaking: deterministic — sort candidate pairs by (score DESC, claim_idx ASC,
    fact_idx ASC) to ensure reproducibility.

    Returns:
        (matched_pair_count, list of (claim_idx, fact_idx, score) triples)
    """
    # Build all qualifying candidate pairs
    candidates: list[tuple[float, int, int]] = []
    ec_tokens = [_normalise(c) for c in extracted_claims]
    rf_tokens = [_normalise(r) for r in reference_facts]

    for ci, ct in enumerate(ec_tokens):
        for ri, rt in enumerate(rf_tokens):
            score = _overlap_coefficient(ct, rt)
            if score >= threshold:
                # Negate score for sort: we want DESC by score, then ASC by indices
                candidates.append((score, ci, ri))

    # Sort: highest score first; ties broken by claim_idx then fact_idx (deterministic)
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))

    matched_claims: set[int] = set()
    matched_facts: set[int] = set()
    matched_pairs: list[tuple[int, int, float]] = []

    for score, ci, ri in candidates:
        if ci not in matched_claims and ri not in matched_facts:
            matched_claims.add(ci)
            matched_facts.add(ri)
            matched_pairs.append((ci, ri, score))

    return len(matched_pairs), matched_pairs


# ─── REFERENCE-CLAIM METRICS (D1) ─────────────────────────────────────────────

def compute_reference_claim_metrics(
    atomic_facts: list[dict],
    reference_facts: list[dict],
    case_id: str,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> dict:
    """Compute precision, recall, and F1 for reference-claim matching.

    Inputs:
        atomic_facts: list of dicts from trace["atomic_facts"], each having
                      "case_id" (or derivable), "field", "value".
        reference_facts: list of dicts from benchmark_facts.jsonl, each having
                         "case_id" and "text".
        case_id: the current benchmark case_id. Facts from other cases are
                 silently excluded.
        threshold: overlap coefficient threshold for a pair to qualify as a match.

    Formula:
        precision = matched / extracted_claim_count  (0.0 if no extracted claims)
        recall    = matched / reference_fact_count   (0.0 if no reference facts)
        f1        = 2*p*r / (p+r)                   (0.0 if denominator is 0)

    Missing-value behaviour:
        If reference_facts is None or an empty list, all three metrics are None
        (unavailable). This must NOT be replaced with 0.0.

    Returns a dict with keys:
        reference_fact_precision, reference_fact_recall, reference_fact_f1,
        extracted_claim_count, reference_fact_count, matched_pair_count,
        matching_algorithm, matching_threshold, case_id, status
    """
    # Guard: no reference facts available → return None, not 0
    if not reference_facts:
        return {
            "reference_fact_precision": None,
            "reference_fact_recall": None,
            "reference_fact_f1": None,
            "extracted_claim_count": None,
            "reference_fact_count": None,
            "matched_pair_count": None,
            "matching_algorithm": MATCHING_ALGORITHM,
            "matching_threshold": threshold,
            "case_id": case_id,
            "status": "no_reference_facts",
        }

    # Isolate reference facts for this exact case_id
    case_ref_facts = [f for f in reference_facts if f.get("case_id") == case_id]
    if not case_ref_facts:
        return {
            "reference_fact_precision": None,
            "reference_fact_recall": None,
            "reference_fact_f1": None,
            "extracted_claim_count": None,
            "reference_fact_count": None,
            "matched_pair_count": None,
            "matching_algorithm": MATCHING_ALGORITHM,
            "matching_threshold": threshold,
            "case_id": case_id,
            "status": "no_reference_facts_for_case",
        }

    ref_texts = [f["text"] for f in case_ref_facts]
    ref_count = len(ref_texts)

    # Build extracted claim units from atomic_facts for this case
    # Deduplicate by (field, normalised_value) pair before splitting
    seen: set[tuple[str, str]] = set()
    raw_values: list[str] = []
    for af in atomic_facts:
        # Accept facts that have matching case context or no case field (legacy)
        field = af.get("field", "")
        value = str(af.get("value", "")).strip()
        if not value:
            continue
        key = (field, value.lower())
        if key not in seen:
            seen.add(key)
            raw_values.append(value)

    # Split multi-sentence values into individual claim units
    extracted_claims: list[str] = []
    for v in raw_values:
        parts = _split_into_claims(v)
        # If splitting produces nothing (very short text), keep original as a single claim
        extracted_claims.extend(parts if parts else [v])

    ext_count = len(extracted_claims)

    if ext_count == 0:
        # No extracted claims → precision undefined (no claims to be precise about),
        # recall = 0 (nothing was recovered), F1 = 0
        return {
            "reference_fact_precision": None,
            "reference_fact_recall": 0.0,
            "reference_fact_f1": 0.0,
            "extracted_claim_count": 0,
            "reference_fact_count": ref_count,
            "matched_pair_count": 0,
            "matching_algorithm": MATCHING_ALGORITHM,
            "matching_threshold": threshold,
            "case_id": case_id,
            "status": "no_extracted_claims",
        }

    matched, _ = _bipartite_match(extracted_claims, ref_texts, threshold)

    precision = matched / ext_count
    recall = matched / ref_count if ref_count > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "reference_fact_precision": round(precision, 6),
        "reference_fact_recall": round(recall, 6),
        "reference_fact_f1": round(f1, 6),
        "extracted_claim_count": ext_count,
        "reference_fact_count": ref_count,
        "matched_pair_count": matched,
        "matching_algorithm": MATCHING_ALGORITHM,
        "matching_threshold": threshold,
        "case_id": case_id,
        "status": "ok",
    }


def sensitivity_sweep(
    atomic_facts: list[dict],
    reference_facts: list[dict],
    case_id: str,
) -> dict[float, dict]:
    """Run reference-claim matching at all documented sensitivity thresholds.

    Returns a mapping {threshold: metrics_dict} for thresholds 0.25, 0.5, 0.75.
    This analysis must not be used to tune the default threshold.
    """
    return {
        t: compute_reference_claim_metrics(atomic_facts, reference_facts, case_id, threshold=t)
        for t in SENSITIVITY_THRESHOLDS
    }


# ─── SCHEMA COMPLETENESS (D2) ─────────────────────────────────────────────────

def compute_schema_completeness(knowledge_state: dict) -> float:
    """Fraction of maximum possible completeness score across all schema fields.

    Weighting (from engine.py, preserved exactly):
        unknown:    0 points
        candidate:  1 point
        supported:  2 points
        verified:   3 points
        conflicting: 1 point (partial credit, not verification)

    Denominator: SCHEMA_FIELD_COUNT × 3 = 27 points
    Returns a float in [0.0, 1.0].

    This metric reflects progress through the knowledge state machine,
    NOT ground-truth correctness.
    """
    score = sum(STATUS_WEIGHTS.get(fstate.get("status", "unknown"), 0)
                for fstate in knowledge_state.values())
    return round(score / MAX_COMPLETENESS_SCORE, 6) if MAX_COMPLETENESS_SCORE > 0 else 0.0


# ─── NOVEL INFORMATION YIELD (D2) ────────────────────────────────────────────

def compute_novel_fields(turns: list[dict]) -> dict:
    """Calculate novel information yield from per-turn records.

    Inputs per participant turn:
        turns[i]["newly_surfaced_fields"]: list of field names first extracted this turn
        turns[i]["novel_facts_count"]: count of newly surfaced fields

    Denominator for novel_fields_per_response: number of participant-role turns.
    Missing-value behaviour: returns 0 if no participant turns exist.

    Returns:
        novel_fields_total: total number of fields first surfaced across all turns
        novel_fields_per_response: average per participant response
    """
    participant_turns = [t for t in turns if t.get("role") == "participant"]
    n_responses = len(participant_turns)

    total = sum(t.get("novel_facts_count", 0) for t in participant_turns)
    per_response = round(total / n_responses, 6) if n_responses > 0 else 0.0

    return {
        "novel_fields_total": total,
        "novel_fields_per_response": per_response,
        "n_participant_turns": n_responses,
    }


# ─── QUESTION REDUNDANCY (D2) ─────────────────────────────────────────────────

def compute_redundancy_rate(turns: list[dict]) -> dict:
    """Calculate question redundancy rate from per-turn redundancy flags.

    Uses the `redundant` boolean set by engine._is_redundant(), which applies
    an overlap-coefficient heuristic (threshold 0.85) against prior questions.
    This detects LEXICAL overlap, NOT semantic repetition.

    Denominator: all follow-up interviewer turns (turn_id > 0, role=interviewer).
    The opening question (turn_id = 0) is excluded from the denominator.

    If no eligible follow-up questions exist, returns None (not-applicable).
    If the `redundant` flag is missing from a turn, that turn is included in
    the denominator but treated as non-redundant.

    Returns:
        question_redundancy_rate: float in [0.0, 1.0] or None
        redundant_count: int
        eligible_question_count: int
    """
    follow_up_qs = [
        t for t in turns
        if t.get("role") == "interviewer" and t.get("turn_id", 0) > 0
    ]
    n_eligible = len(follow_up_qs)

    if n_eligible == 0:
        return {
            "question_redundancy_rate": None,
            "redundant_count": 0,
            "eligible_question_count": 0,
            "status": "not_applicable_no_followup_questions",
        }

    redundant_count = sum(1 for t in follow_up_qs if t.get("redundant", False))
    rate = round(redundant_count / n_eligible, 6)

    return {
        "question_redundancy_rate": rate,
        "redundant_count": redundant_count,
        "eligible_question_count": n_eligible,
        "status": "ok",
    }


# ─── GAP TARGETING RATE (D2) ─────────────────────────────────────────────────

def compute_gap_targeting_rate(turns: list[dict], ablation: str = "none") -> dict:
    """Calculate time-consistent gap targeting rate for follow-up questions.

    For the `state_machine` ablation, returns None/not-applicable because gap
    selection is intentionally absent.

    For each follow-up interviewer turn (turn_id > 0), a successful gap alignment
    requires BOTH:
      1. The turn has a `knowledge_gap_targeted` field matching its `target_field`.
      2. The target field's status in `knowledge_state_before_question` was
         "unknown", "candidate", or "conflicting" (i.e., unresolved at question time).

    Follow-up questions with missing or invalid target_field metadata ARE included
    in the denominator — they are not successful alignments.

    Returns:
        gap_targeting_rate: float or None
        aligned_count: int or None
        eligible_count: int or None
        status: str
    """
    if ablation == "state_machine":
        return {
            "gap_targeting_rate": None,
            "aligned_count": None,
            "eligible_count": None,
            "status": "not_applicable_state_machine_ablation",
        }

    follow_up_qs = [
        t for t in turns
        if t.get("role") == "interviewer" and t.get("turn_id", 0) > 0
    ]
    n_eligible = len(follow_up_qs)

    if n_eligible == 0:
        return {
            "gap_targeting_rate": None,
            "aligned_count": 0,
            "eligible_count": 0,
            "status": "not_applicable_no_followup_questions",
        }

    UNRESOLVED_STATUSES = {"unknown", "candidate", "conflicting"}
    aligned_count = 0

    for turn in follow_up_qs:
        target_field = turn.get("target_field")
        gap_targeted = turn.get("knowledge_gap_targeted")

        # Must have a target field and it must match the selected gap
        if not target_field or not gap_targeted:
            continue
        if target_field != gap_targeted:
            continue

        # Read the knowledge state BEFORE this question was generated
        state_before = turn.get("knowledge_state_before_question", {})
        field_state = state_before.get(target_field, {})
        field_status = field_state.get("status", "unknown")

        if field_status in UNRESOLVED_STATUSES:
            aligned_count += 1

    rate = round(aligned_count / n_eligible, 6)

    return {
        "gap_targeting_rate": rate,
        "aligned_count": aligned_count,
        "eligible_count": n_eligible,
        "status": "ok",
    }


# ─── CONFLICT AND EVIDENCE DIAGNOSTICS (D2) ──────────────────────────────────

def compute_conflicting_field_rate(knowledge_state: dict) -> dict:
    """Fraction of non-unknown fields in the 'conflicting' state.

    Denominator: fields whose status is not 'unknown'.
    If all fields are unknown, returns None (not-applicable).

    This diagnostic does NOT establish actual logical contradiction — it reflects
    the state machine's conflict-detection heuristic (lexical overlap < threshold).
    """
    non_unknown = [f for f, s in knowledge_state.items() if s.get("status") != "unknown"]
    conflicting = [f for f in non_unknown if knowledge_state[f].get("status") == "conflicting"]

    n_non_unknown = len(non_unknown)
    n_conflicting = len(conflicting)

    if n_non_unknown == 0:
        return {
            "conflicting_field_rate": None,
            "conflicting_count": 0,
            "non_unknown_field_count": 0,
            "status": "not_applicable_all_unknown",
        }

    return {
        "conflicting_field_rate": round(n_conflicting / n_non_unknown, 6),
        "conflicting_count": n_conflicting,
        "non_unknown_field_count": n_non_unknown,
        "status": "ok",
    }


def compute_evidence_presence_rate(knowledge_state: dict) -> dict:
    """Fraction of fields with extracted values that also have non-empty evidence metadata.

    Denominator: fields with a non-None, non-empty value (populated fields).
    If no fields have extracted values, returns None (not-applicable).

    A non-empty evidence list does NOT establish that the evidence semantically
    supports the extracted value. This is a presence diagnostic only.
    """
    populated = [
        (f, s) for f, s in knowledge_state.items()
        if s.get("value") and str(s.get("value", "")).strip()
    ]
    with_evidence = [
        (f, s) for f, s in populated
        if s.get("evidence") and len(s["evidence"]) > 0
    ]

    n_populated = len(populated)
    n_with_evidence = len(with_evidence)

    if n_populated == 0:
        return {
            "evidence_presence_rate": None,
            "populated_field_count": 0,
            "fields_with_evidence_count": 0,
            "status": "not_applicable_no_populated_fields",
        }

    return {
        "evidence_presence_rate": round(n_with_evidence / n_populated, 6),
        "populated_field_count": n_populated,
        "fields_with_evidence_count": n_with_evidence,
        "status": "ok",
    }


# ─── EXTRACTION DIAGNOSTICS ───────────────────────────────────────────────────

def compute_extraction_diagnostics(turns: list[dict]) -> dict:
    """Count extraction method usage (llm / fallback / empty) across all participant turns.

    These are implementation diagnostics, not scoring metrics.
    Only included when reliable extraction metadata exists in turns.
    """
    llm_count = 0
    fallback_count = 0
    empty_count = 0
    total_turns = 0

    for t in turns:
        if t.get("role") != "participant":
            continue
        total_turns += 1
        for upd in t.get("extraction", []):
            # extraction method is tracked per-update when available
            pass
        # Per-turn extraction method is not currently stored directly in participant turn records.
        # It was returned from InterviewSession._extract() but not written to the participant
        # turn dict. We return absent counts rather than fabricate zeros.

    return {
        "llm_extraction_count": None,
        "fallback_extraction_count": None,
        "empty_extraction_count": None,
        "status": "unavailable_extraction_method_not_stored_per_turn",
        "n_participant_turns": total_turns,
    }


# ─── AGGREGATE ENTRYPOINT ─────────────────────────────────────────────────────

def compute_all_metrics(
    trace: dict,
    reference_facts: Optional[list[dict]],
    case_id: str,
    ablation: str = "none",
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> dict:
    """Compute all production evaluation metrics for a single completed interview trace.

    Inputs:
        trace: the full interview trace dict from InterviewSession._state()
        reference_facts: list of benchmark fact dicts for this case (or None)
        case_id: the benchmark case_id to filter reference facts
        ablation: experiment ablation condition ("none", "no_rag", "state_machine")
        threshold: lexical matching threshold (default DEFAULT_MATCH_THRESHOLD = 0.5)

    Returns a flat dict of all metric values, with:
        - null values for unavailable or not-applicable metrics
        - diagnostics counts for transparency
        - matching metadata (algorithm version, threshold, counts)

    Primary outcome: reference_fact_f1
    Secondary outcomes: all other keys in this dict
    """
    turns = trace.get("turns", [])
    knowledge_state = trace.get("knowledge_state", {})
    atomic_facts = trace.get("atomic_facts", [])

    # Primary: reference-claim matching
    ref_metrics = compute_reference_claim_metrics(
        atomic_facts, reference_facts or [], case_id, threshold=threshold
    )

    # Secondary: structural and process measures
    completeness = compute_schema_completeness(knowledge_state)
    novelty = compute_novel_fields(turns)
    redundancy = compute_redundancy_rate(turns)
    gap = compute_gap_targeting_rate(turns, ablation=ablation)
    conflict = compute_conflicting_field_rate(knowledge_state)
    evidence = compute_evidence_presence_rate(knowledge_state)
    extraction_diag = compute_extraction_diagnostics(turns)

    return {
        # Primary reference-claim metrics
        "reference_fact_precision": ref_metrics["reference_fact_precision"],
        "reference_fact_recall": ref_metrics["reference_fact_recall"],
        "reference_fact_f1": ref_metrics["reference_fact_f1"],
        # Matching metadata
        "matching_algorithm": ref_metrics["matching_algorithm"],
        "matching_threshold": ref_metrics["matching_threshold"],
        "extracted_claim_count": ref_metrics["extracted_claim_count"],
        "reference_fact_count": ref_metrics["reference_fact_count"],
        "matched_pair_count": ref_metrics["matched_pair_count"],
        "reference_metrics_status": ref_metrics["status"],
        # Secondary structural metrics
        "schema_completeness": completeness,
        "novel_fields_total": novelty["novel_fields_total"],
        "novel_fields_per_response": novelty["novel_fields_per_response"],
        # Process diagnostics
        "question_redundancy_rate": redundancy["question_redundancy_rate"],
        "redundant_question_count": redundancy["redundant_count"],
        "eligible_question_count": redundancy["eligible_question_count"],
        "gap_targeting_rate": gap["gap_targeting_rate"],
        "gap_aligned_count": gap["aligned_count"],
        "gap_eligible_count": gap["eligible_count"],
        # Knowledge state diagnostics
        "conflicting_field_rate": conflict["conflicting_field_rate"],
        "conflicting_field_count": conflict["conflicting_count"],
        "evidence_presence_rate": evidence["evidence_presence_rate"],
        "fields_with_evidence_count": evidence["fields_with_evidence_count"],
        # Extraction diagnostics (currently unavailable — stored for future use)
        "extraction_diag_status": extraction_diag["status"],
        # Version
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
    }
"""
Metric definitions summary
--------------------------
reference_fact_precision   = matched_pairs / extracted_claim_count
                             None if no extracted claims (no precision denominator)
reference_fact_recall      = matched_pairs / reference_fact_count
                             None if no reference facts available
reference_fact_f1          = 2*P*R / (P+R), 0 if P+R=0
                             None if reference facts unavailable
schema_completeness        = sum(status_weights) / (9 * 3)
novel_fields_total         = sum of newly_surfaced_fields counts across participant turns
novel_fields_per_response  = novel_fields_total / n_participant_turns
question_redundancy_rate   = redundant_follow_up_questions / eligible_follow_up_questions
gap_targeting_rate         = aligned_follow_up_questions / eligible_follow_up_questions
                             null for state_machine ablation
conflicting_field_rate     = conflicting_fields / non_unknown_fields
evidence_presence_rate     = fields_with_evidence / populated_fields

All rates return None (null) when their denominator is zero or the measure is
not-applicable. Null must never be replaced with zero in downstream aggregation.
"""
