# Evaluation Metrics Reference

**Metric schema version**: v3
**Matching algorithm**: `lexical_bipartite_exposed_v2`
**Default matching threshold**: `0.5`

> **Important limitations**: All metrics are simulation-based and reference-based. Lexical matching is a proxy, not verified factual accuracy. Panel agreement does not establish human-preference alignment or real-world correctness. No human evaluation is performed.

---

## Primary Outcome

### `reference_fact_f1`

**Type**: float [0, 1] or `null`  
**Formula**: `2 × P × R / (P + R)`, zero if `P + R = 0`  
**Null when**: Reference facts are unavailable or the synthetic respondent disclosed no source sentences in that episode.

The primary extraction outcome. Balances precision and recall against the full-source sentences that the deterministic synthetic respondent actually disclosed during that episode. This exposure-conditioned denominator fixes the previous mismatch where the benchmark expected only the first 12 sentences but the respondent could disclose any source sentence. Claims are still matched lexically, so this is a reproducible proxy, not semantic or human-verified accuracy.

---

## Reference-Claim Metrics

These metrics quantify how well the model extracted information that can be lexically matched to known reference facts.

### Matching algorithm

Each extracted claim is compared to each exposed reference sentence using the **overlap coefficient**:

```
overlap(A, B) = |tokens(A) ∩ tokens(B)| / max(|tokens(A)|, |tokens(B)|)
```

A bipartite one-to-one matching is then computed (greedy, deterministic, score-descending). Each extracted claim and each reference fact can participate in at most one match.

Tokens are lowercased, punctuation stripped, and unique (set-based).

### `reference_fact_precision`

**Formula**: `matched_pairs / extracted_claim_count`  
**Null when**: No extracted claims (no precision denominator).

Fraction of extracted claim units that matched at least one reference fact.

### `reference_fact_recall`

**Formula**: `matched_pairs / reference_fact_count`  
**Null when**: Reference facts unavailable.  
**Zero when**: No extracted claims.

Fraction of reference facts covered by at least one extracted claim.

### `extracted_claim_count`

Number of unique extracted claim units sent into matching. Multi-sentence field values are split into individual claims; duplicates by (field, value) are removed before splitting.

### `reference_fact_count` and `exposed_reference_fact_count`

`reference_fact_count` is the number of source sentences disclosed in the episode and used as the recall denominator. `exposed_reference_fact_count` is the same quantity for clarity. `source_reference_fact_count` is the complete sentence inventory for the case; it is reported for context but does not count undisclosed facts as misses.

### `reference_fact_exposure_rate`

`exposed_reference_fact_count / source_reference_fact_count`. This describes how much of the case was disclosed by the respondent. It is not an extraction-quality score. Compare it alongside F1 because low exposure means the interviewer had less source content available to extract.

### `reference_metrics_scope`

Production traces use `respondent_exposed_facts`. Traces missing respondent exposure metadata are scored with an empty exposed set and return null F1; they are not silently scored against every source sentence. `all_case_source_facts_legacy` identifies direct calls using the historical denominator.

### `matched_pair_count`

Number of one-to-one matched pairs at the default threshold.

### Sensitivity analysis

`sensitivity_sweep()` runs matching at thresholds `{0.25, 0.5, 0.75}` and returns all three results. Use only to characterise measurement stability — do NOT use to select the tuned threshold.

---

## Schema Completeness

### `schema_completeness`

**Type**: float [0, 1]  
**Formula**: `sum(status_weights) / (9 × 3)`

Status weights:
| Status | Weight |
|---|---|
| `unknown` | 0 |
| `candidate` | 1 |
| `conflicting` | 1 |
| `supported` | 2 |
| `verified` | 3 |

Reflects progress through the state machine. Does NOT measure ground-truth correctness.

---

## Information Yield

### `novel_fields_total`

Total count of schema fields that were first surfaced (status transitioned out of `unknown`) during participant turns. Summed across all participant turns.

### `novel_fields_per_response`

`novel_fields_total / n_participant_turns`. Zero if no participant turns.

---

## Process Diagnostics

### `question_redundancy_rate`

**Formula**: `redundant_follow_up_questions / eligible_follow_up_questions`  
**Null when**: No follow-up questions (turn_id > 0).

Uses the `redundant` flag set by `engine._is_redundant()` (lexical overlap ≥ 0.85 against prior questions). Detects lexical redundancy only, not semantic.

### `gap_targeting_rate`

**Formula**: `aligned_follow_up_questions / eligible_follow_up_questions`  
**Null when**: No follow-up questions, OR `ablation = "state_machine"`.

A follow-up question is time-consistently aligned if:
1. Its `target_field` matches `knowledge_gap_targeted`.
2. The field's status in `knowledge_state_before_question` was `unknown`, `candidate`, or `conflicting`.

Uses the `knowledge_state_before_question` snapshot stored **before** each question is generated.

### `conflicting_field_rate`

**Formula**: `conflicting_fields / non_unknown_fields`  
**Null when**: All fields are `unknown`.

Fraction of active fields in the `conflicting` state. Conflict detection is lexical (not logical contradiction).

### `evidence_presence_rate`

**Formula**: `populated_fields_with_evidence / populated_fields`  
**Null when**: No populated fields.

Fraction of extracted fields that have non-empty `evidence` metadata. Evidence presence ≠ evidence validity.

### `evidence_text_support_rate`

Fraction of extracted field updates whose recorded evidence has at least the configured lexical support threshold against the participant's disclosed source sentences. This automatic check helps identify unsupported extraction claims; it does not establish semantic entailment or replace human validation.

---

## LLM-as-a-Judge Score

### `follow_up_question_quality`

**Type**: Integer `{0, 1, 2, 3, 4}` (panel median), or `null`  
**Metric schema version**: v2  
**Null when**: No valid judge scores available.

Single rubric-based metric assessing observable transcript-level question quality:

| Score | Meaning |
|---|---|
| 0 | Unacceptable: most follow-ups irrelevant or incoherent |
| 1 | Poor: generic, poorly connected to prior responses |
| 2 | Mixed: inconsistent targeting or specificity |
| 3 | Good: relevant, specific, appropriate progression |
| 4 | Excellent: consistent, non-leading, advancing inquiry |

Judges evaluate the sanitized transcript only. Candidate model identity, timing metadata, and extraction data are stripped before evaluation.

### `judge_disagreement_mad`

Median absolute deviation of valid judge scores. `null` if fewer than 2 valid scores.  
`0.0` MAD means all valid judges agreed exactly. `null` MAD means disagreement is **not estimable**.

---

## Statistical Analysis

### Resampling strategy

Bootstrap CIs use **cluster-aware resampling** at the `case_group_id` level. Multiple question variants and iterations of the same source case are kept together. Do not pool resampling across `case_group_id` boundaries.

Bootstrap seed: `42` (fixed, recorded in output).

### Paired comparisons

For each pair of models sharing identical `(case_group_id, question_variant_id)` keys, pairwise differences in `reference_fact_f1` are computed. Bootstrap CIs on differences use the same cluster-aware approach.

> CIs reflect statistical uncertainty due to sampling. Stability of a CI does **not** establish construct validity or causal interpretations.
