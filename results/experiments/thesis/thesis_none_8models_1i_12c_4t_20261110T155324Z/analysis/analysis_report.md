# Experiment analysis: thesis_none_8models_1i_12c_4t_20261110T155324Z

Profile: **thesis**

## Data coverage

- Successful episodes: 96; failed episodes: 0
- Failed episodes by model: none
- Planned episode coverage: 100.0%
- Benchmark variants evaluated: 12 of 57
- Candidate models: deepseek-r1-distill-qwen-1.5b, llama-3.2-1b-instruct, llama-3.2-3b-instruct, phi-3.5-mini-instruct, phi-4-mini-instruct, qwen2.5-1.5b-instruct, qwen2.5-3b-instruct, qwen3-1.7b
- Independent source-case groups: 3
- Judge score rows: 384 across 96 traces
- Scope: finite proof of concept; findings are limited to these source documents.

## Proof-of-concept extraction check

LLM extractions occurred in 87 of 96 successful episodes; total LLM extractions: 308.0.
Mean schema completeness: 0.166; mean reference-fact F1: 0.335.

## RQ1 — Schema completeness and reference-fact F1

15 of 28 pairwise schema-completeness source-cluster intervals exclude zero in this finite benchmark; the highest mean was phi-4-mini-instruct (0.275) and the lowest was qwen2.5-1.5b-instruct (0.077). For the reference-fact F1 proxy, 20 of 28 paired intervals exclude zero; its highest mean was llama-3.2-1b-instruct (0.746) and lowest was qwen2.5-3b-instruct (0.190). The measures rank models differently. Treat these as benchmark-specific evidence: only three independent source groups were sampled, and pairwise intervals are unadjusted for multiple comparisons.

Model estimates and paired cluster-bootstrap intervals are in `rq1_model_estimates.csv` and `rq1_paired_differences.csv`.
For the reference-fact F1 proxy, 20 of 28 paired source-group intervals exclude zero; the highest model mean was llama-3.2-1b-instruct (0.746) and the lowest was qwen2.5-3b-instruct (0.190). Schema completeness and reference-fact F1 rank models differently. Reference-fact F1 is a lexical proxy applied only to full-source facts disclosed by the synthetic respondent.
A paired interval excluding zero is benchmark-specific evidence of a difference; an interval including zero leaves that comparison unresolved, not proof of equivalence. These pairwise intervals are unadjusted and based on only three source groups.

## RQ2 — Elicitation efficiency

Descriptively, phi-4-mini-instruct had the highest novel-field gain (1.52 per response), qwen2.5-1.5b-instruct had the highest gap-targeting rate (0.53). Redundancy was highest for qwen3-1.7b (0.36) and lowest for llama-3.2-3b-instruct (0.00); several models tied at zero. Turn counts were 4 for every episode, so this run cannot distinguish models by turn count. No single efficiency measure has a prespecified pass threshold.
Model-level descriptive values are in `rq2_descriptive_statistics.csv`. All turn-count values are shown; invariant turn counts cannot rank models by speed. Read the other efficiency indicators with schema completeness and reference-fact F1.

## RQ3 — Amended question: judge-assessed follow-up quality

Under the amended descriptive quality question, 59.4% of valid ratings were good or excellent. By candidate model the observed range was 39.6% (qwen3-1.7b) to 72.9% (phi-3.5-mini-instruct). This does not answer the original inter-rater reliability question or establish a pass mark.
This post-completion amendment uses the rubric's good/excellent categories (scores 3–4) to describe judge-assessed quality. It does not measure judge reliability, and the original reliability question remains unanswered. The overall and per-model rating shares have no pass threshold; see `rq3_good_excellent_by_candidate.csv`, `rq3_score_distribution.csv`, and `rq3_judge_coverage.csv`.

## Interpretation limits

- The full profile evaluates benchmark variants from a small finite proof-of-concept corpus; it is not a representative enterprise sample.
- Question variants from one case_group_id are repeated measures, not independent source documents; report the source-group count with all results.
- Cluster-bootstrap intervals are based on 3 source groups; with three groups they are coarse and should be interpreted descriptively.
- Lexical reference-fact F1 is a proxy and judge-assessed quality does not establish human validity.
- RQ2 summaries are descriptive; the thesis does not specify a formal significance test for those outcomes.
- The completed-run RQ3 analysis was explicitly reframed from inter-rater reliability to descriptive judge-assessed quality; the original reliability question remains unanswered by this outcome.
