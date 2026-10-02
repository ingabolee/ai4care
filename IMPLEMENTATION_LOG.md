# IMPLEMENTATION_LOG.md

## Current Architecture (Post-Refactor)

### Core Components

| File | Role |
|------|------|
| `model_registry.py` | Single source of truth for model configs. Defines `CHROMA_COLLECTION = "ai4care_knowledge"`. |
| `verify_models.py` | Checks each enabled model: file exists, loads, generates, has chat template. |
| `setup.py` | Downloads models, creates single Chroma collection `ai4care_knowledge`. |
| `knowledge.py` | All Chroma ops against single collection. Atomic fact storage, episode cleanup. |
| `engine.py` | Full interview engine: LLM JSON extraction, evidence state machine, adaptive gap selection, rolling-state prompts. |
| `cli.py` | Interactive REPL using updated engine. |
| `curate_seed_corpus.py` | Reconstructs logical documents, deduplicates, classifies, chunks, batch-upserts as namespace=corpus. |
| `compile_benchmark_dataset.py` | Builds frozen benchmark Q/A pairs and reference facts. |
| `synthetic_respondent.py` | Deterministic participant using word-overlap scoring and seeded RNG. No LLM. |
| `evaluate.py` | Main runner: iteration randomization, all models same assignment, clean-slate, metrics, CSVs. |
| `analyze_benchmark_size.py` | Bootstrap CI analysis for benchmark adequacy. |
| `seed_dataset.py` | Updated for single collection with upsert idempotency. |
| `tests/test_suite.py` | 26 test classes covering all acceptance criteria. |

---

## Existing Problems (Pre-Refactor)

- Multiple Chroma collections (one per model): ai4care_general_small, ai4care_general_main, etc.
- Regex-based keyword extraction with no evidence, no conflict detection, no state machine.
- Full transcript concatenated and sent to LLM on every turn.
- Manual chat token injection bypassing the model's native chat template.
- No deterministic evaluation harness — required human input via terminal loop.
- No configurable model registry.
- Storage errors silently swallowed with bare except.
- Per-record Chroma existence checks instead of batch upsert.

---

## Modifications Made

### `setup.py`
- Removed per-model collection creation loop.
- Single ai4care_knowledge collection with seed bootstrap.
- Now imports all config from model_registry.py.

### `knowledge.py`
- Full rewrite. Removed per-model collection lookup.
- Added: `store_atomic_fact()` with evidence, confidence, status metadata.
- Added: `delete_episode_records(interview_id)` for clean-slate enforcement.
- Added: Unified `query(text, n, where)` with optional metadata filter.

### `engine.py`
- Full rewrite.
- Removed: `_extract()` regex/keyword extraction.
- Removed: manual chat token string construction.
- Added: LLM JSON extraction with two-attempt retry.
- Added: Evidence-based knowledge state machine (unknown, candidate, supported, verified, conflicting).
- Added: `select_next_gap()` — priority-ordered gap selection.
- Added: Rolling-state prompts with RECENT_TURNS_WINDOW (not full transcript).
- Added: Native `create_chat_completion()` for all model calls.
- Added: Question redundancy detection using word overlap.
- Added: Atomic facts accumulated and stored via knowledge.store_atomic_fact().

### `cli.py`
- Updated for new engine API, knowledge state display, storage status reporting.
- Added `--list-models` flag.

### `seed_dataset.py`
- Updated to point to single Chroma collection with upsert idempotency.
- Added namespace=corpus metadata tag.

---

## New Files

- `model_registry.py` — configurable model pool
- `verify_models.py` — model compatibility testing
- `curate_seed_corpus.py` — full curation pipeline
- `compile_benchmark_dataset.py` — frozen benchmark compilation
- `synthetic_respondent.py` — deterministic grounded participant
- `evaluate.py` — automated evaluation runner
- `analyze_benchmark_size.py` — bootstrap CI analysis
- `tests/test_suite.py` — 26 test classes

---

## Removed Logic

- Per-model Chroma collection creation and maintenance code.
- Regex keyword extraction (`_extract()` in old `engine.py`).
- Manual chat template token injection.
- `store_all()` (stored concatenated summary across all 4 collections).
- Hardcoded model-specific collection constants in `knowledge.py`.

---

## Tests

See `tests/test_suite.py`. Run with: `python -m pytest tests/test_suite.py -v`

Test classes include:
- TestChromaSingleCollection
- TestModelRegistry
- TestStateMachine (transitions, conflicts, verified status)
- TestGapSelection
- TestSyntheticRespondent (determinism, grounding)
- TestIterationRandomization (same seed = same order)
- TestSameCasesAcrossModels
- TestCleanSlateEpisode
- TestTargetCaseLeakage
- TestStorageFailureReporting
- TestBatchIngestion (deduplication)
- TestMetrics
- TestCaseGrouping
- TestCSVResults

---

## Benchmark Configuration

- Source dataset: ozguragrali/enterprise-knowledge-qa-dataset-gemini-flash-for-t5-large
- Benchmark formats: benchmark_cases.jsonl + benchmark_facts.jsonl
- Question variants: 9 types per case (overall, challenge, root_cause, decision, technology, people, process, impact, lessons_learned)
- Reference facts: deterministic sentence-level extraction (no LLM)
- Randomization: one manifest per iteration, shared across all models

---

## Evaluation Results

- To be populated after first evaluation run.

---

## Unresolved Limitations

- Fact matching uses word overlap rather than a separate embedding model (preserves the controlled constant status of Chroma's embedding function).
- Synthetic respondent does not use an LLM — deterministic but less natural than a real participant.
- Bootstrap CI analysis requires completed evaluation runs — run analyze_benchmark_size.py after evaluate.py.
