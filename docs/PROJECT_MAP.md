# AI4CARE project map

This guide lists the files needed for the thesis workflow and the order to use them. The repository keeps separate modules for separate experimental responsibilities; combining them would make it harder to audit the benchmark and reproduce a run.

## Main run order

1. **Environment:** install `requirements.txt`. Install `requirements-analysis.txt` for the notebook and analysis extras.
2. **Vector store:** run `python setup.py` once to initialise the shared Chroma collection.
3. **Corpus:** run `curate_seed_corpus.py` when the source corpus must be rebuilt. It writes the reproducible corpus under `data/corpus/` and fills the `corpus` namespace in ChromaDB.
4. **Benchmark:** run `compile_benchmark_dataset.py` once to create the frozen cases, full sentence-level reference inventory, and manifest in `data/benchmark/`. Rebuild only when the corpus or benchmark/scoring design changes, and keep its hash with each run.
5. **Interviews:** `python evaluate.py --profile thesis --models all` starts the compact main thesis run. The hardware estimator selects eight fitting compact models. The default sample is 12 variants balanced across the three source groups, with at most four participant replies per interview. A `pilot` profile is only a pipeline check. `full` evaluates every available variant.
6. **Judges:** `python judge_benchmark.py --experiment-dir <run-folder>` runs the estimator-selected panel of three small and one larger judge where hardware permits. The default deterministic sample uses up to one trace per interviewer-model/source-group stratum.
7. **Analysis:** run `python analyze_experiment.py --experiment-dir <run-folder>`, then use `notebooks/thesis_results_analysis.ipynb` for the detailed tables, plots, interpretation rules, and missing-data explanation.

The run ID records the profile, ablation, number of interviewer models, iterations, selected benchmark cases, reply cap, and UTC creation time. For example: `thesis_none_8models_1i_12c_4t_20261110T120000Z`. `pilot_...` and `thesis_...` runs are therefore distinct, and experiment outputs stay under profile-specific folders.

## Maintained code by responsibility

| File | Responsibility |
|---|---|
| `evaluate.py` | Reproducible assignment, interviewer episodes, checkpoints/progress, deterministic metric output, and run manifest. |
| `estimator.py` | Hardware and memory estimates used to choose the interviewer and judge panels. |
| `model_registry.py` | Model resolution, on-demand GGUF download/cache, runner setup, and shared Chroma collection name. |
| `inference.py` | Local llama.cpp and OpenAI-compatible inference interfaces. |
| `engine.py` | Interview flow, extraction, evidence state, RAG, and question selection. |
| `synthetic_respondent.py` | Deterministic source-grounded respondent; not an LLM and not a human participant. |
| `knowledge.py` | Shared Chroma access and per-episode knowledge storage/retrieval. |
| `compile_benchmark_dataset.py` | Frozen benchmark cases plus a complete sentence inventory with stable source indices. |
| `evaluation_metrics.py` | Interview metrics, including source-exposure-conditioned reference matching. |
| `judge_benchmark.py` | Blinded judge sampling, estimator-selected judge execution, resumable scores, and clean CSV/JSON output. |
| `judge_selector.py` | Hardware-ranked candidate judges and size/family balancing. |
| `llm_judge.py` | Judge rubric, output parsing/validation, and retries. |
| `panel_aggregator.py` | Per-trace panel medians and inter-judge disagreement. |
| `analyze_experiment.py` | Scripted RQ summaries, paired estimates, cluster-aware intervals, and judge reliability. |
| `analyze_benchmark_size.py` | Supplemental analysis of benchmark adequacy. |
| `optimizations.py` | Shared low-overhead snapshots, indexes, and buffered I/O helpers used by the runtime. |
| `curate_seed_corpus.py` | Source dataset reconstruction, curation, chunking, and Chroma ingestion. |
| `setup.py` | Optional setup/bootstrap utilities for Chroma and older manually registered model demos. |
| `tests/` | Maintained automated checks for the engine, metrics, provenance, judges, and statistical analysis. |

`cli.py` and `verify_models.py` support manual demonstrations and model troubleshooting. They are not required for the automated thesis run. One-off optimization patchers, duplicate root-level test runners, stale capacity reports, and the superseded corpus-ingestion script have been removed; the maintained runtime optimizations remain in the modules above.

## Performance and reproducibility controls

- Interview episodes are grouped by candidate model so each worker loads a model once and reuses it across that model's assigned cases.
- On CPU, automatic concurrency is limited by both available cores and the combined estimator memory estimates, reserving RAM for the operating system. A single local GPU defaults to one model worker to avoid duplicating model weights in accelerator memory.
- Episode traces are checkpointed as workers finish; progress and a rolling ETA are reported while work is running. Compatible checkpoints can be resumed using the same run folder and configuration.
- Judge work uses a seeded, balanced candidate-model by source-group sample by default. A local judge is loaded alone, and every completed score is flushed to disk so interrupted judging can resume.
- Each run records the selected estimator panel, hardware, configuration fingerprint, benchmark hash, sample settings, and stage status. Run folders isolate pilot, thesis, and exhaustive profiles.

## Data that must be kept

- `data/benchmark/`: benchmark cases, complete source sentence facts, and the benchmark manifest/hash.
- `data/corpus/`: curated source documents used to create the benchmark.
- `.chroma_db/`: indexed corpus used by RAG.
- `.models/`: downloaded model cache; may be recreated from the estimator-selected model IDs, but retaining it avoids repeat downloads.
- `results/experiments/<profile>/<run-id>/`: per-run traces, CSV metrics, logs/manifests, judge outputs, and analysis.
- `results_previous_runs_*/`: prior run archive retained for provenance; the analysis notebook never merges its records into a new experiment.
- `thesis_latex_source/`: thesis source and literature material.

Do not delete these folders as routine cleanup. The estimator panel and benchmark hash should remain identifiable in each new manifest.

## Analysis interpretation boundary

The benchmark currently has three independent source documents. Variants from the same document are paired repeated tasks, not new independent documents. The study can show proof-of-concept behavior on these sources; it cannot establish performance across a broad workforce. Reference-fact F1 uses only source sentences the respondent actually exposed in each interview. It remains a lexical proxy and does not replace semantic or human validation.
