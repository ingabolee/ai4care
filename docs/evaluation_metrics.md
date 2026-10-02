# AI4CARE Evaluation Metrics

This document describes all metrics computed by the AI4CARE evaluation framework.

---

## Reference-Based Metrics (Objective)

These are computed by comparing the session's extracted knowledge against the benchmark reference facts.

| Metric | Description | Range |
|--------|-------------|-------|
| `schema_completeness` | Fraction of schema fields at `supported` or higher, weighted by status level | 0–1 |
| `reference_fact_recall` | Fraction of gold reference facts matched by extracted values | 0–1 |
| `reference_fact_precision` | Fraction of extracted values that match a gold reference fact | 0–1 |
| `reference_fact_f1` | Harmonic mean of recall and precision | 0–1 |
| `unsupported_fact_rate` | Fraction of extracted values that do NOT match any reference fact (potential hallucination) | 0–1 |
| `evidence_support_rate` | Fraction of extracted facts backed by direct transcript evidence | 0–1 |
| `contradiction_rate` | Fraction of non-unknown fields in a `conflicting` state | 0–1 |
| `gap_targeting_rate` | Fraction of interviewer questions that targeted an unresolved gap | 0–1 |
| `redundancy_rate` | Fraction of interviewer questions flagged as redundant (cosine similarity > 0.85 to prior Q) | 0–1 |

---

## Turn-Level Novelty / Tacit Knowledge Metrics

These are tracked per participant turn in the trace.

| Field | Description |
|-------|-------------|
| `newly_surfaced_fields` | List of schema fields that transitioned out of `unknown` in this turn |
| `novel_facts_count` | Integer count of newly surfaced fields per turn |
| `completeness_after` | Schema completeness immediately after this turn's extraction |

These enable **turn-efficiency curves**: plotting `novel_facts_count` per turn shows when knowledge elicitation becomes saturated.

---

## Judge-Based Metrics (Subjective, MoE)

These are produced by the `LLMJudge` panel in PHASE 2 of the evaluation. Each metric is the **median** of scores across all judges. Disagreement is reported as `<metric>_stdev`.

| Metric | Description | Rubric (0=worst, 10=best) |
|--------|-------------|---------------------------|
| `turn_level_completeness_efficiency` | How efficiently each question elicited new information | 10 = every question produces new knowledge |
| `turn_level_contradiction_rate` | How often the interviewer probed apparent contradictions | 10 = contradictions always addressed |
| `overall_elicit_efficiency` | Overall ratio of information gained vs. turns spent | 10 = maximum knowledge in minimum turns |
| `participant_engagement` | How natural and engaging the conversation felt | 10 = highly engaging, natural flow |
| `redundancy_penalty` | Penalty for repeating already-answered topics | 0 = no redundancy (best), 10 = extreme redundancy (worst) |
| `hallucination_penalty` | Penalty for interviewer questions based on fabricated context | 0 = no hallucination (best), 10 = extreme (worst) |

> **Note**: `redundancy_penalty` and `hallucination_penalty` are inverted — lower scores are better. The framework uses raw values; normalise if needed for composite scoring.

---

## Latency Metrics

These are recorded directly from the generation timestamps.

| Metric | Description |
|--------|-------------|
| `generation_latency_ms` | Time to generate an interviewer question |
| `extraction_latency_ms` | Time to run structured extraction on a participant answer |
| `retrieval_latency_ms` | Time for ChromaDB similarity search |
| `total_runtime_ms` | Total wall-clock time for one complete episode |

---

## Experiment-Level Identifiers

| Field | Description |
|-------|-------------|
| `run_id` | Unique `{iteration_id}_{model_id}_{case_id[:8]}` string |
| `iteration_id` | Iteration number (zero-padded 4-digit string) |
| `model_id` | Candidate model being evaluated |
| `case_id` | Benchmark case identifier |
| `ablation` | Active ablation (`none`, `no_rag`, `state_machine`) |
| `successful_judges` | Number of judges that returned parseable scores |
