# AI4CARE — Automated LLM Knowledge Elicitation Evaluation Framework

A reproducible research framework for benchmarking small **decoder-only language models** as adaptive interviewers for tacit organisational knowledge elicitation. Local runs use deterministic seeds; OpenAI-compatible services may not guarantee identical generations. Runs are resumable and use a synthetic respondent rather than human participants.

---

## How It Works

The pipeline runs in three stages:

```
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE 1 — GENERATION                                                │
│                                                                      │
│  evaluate.py                                                         │
│      └─► InterviewSession (engine.py)                                │
│               ├─► ChatRunner → GGUF candidate model (inference.py)   │
│               ├─► SyntheticRespondent  (deterministic, no LLM)       │
│               ├─► Knowledge state machine (9-field schema)           │
│               ├─► Gap selector → ChromaDB RAG (top-5 chunks)         │
│               └─► Trace appended to <experiment>/evaluation_traces.jsonl│
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  PHASE 2 — BLINDED MoE JUDGE EVALUATION                              │
│                                                                      │
│  judge_benchmark.py                                                  │
│      └─► sanitize_trace()  ← strips model ID, timestamps, latency   │
│               └─► LLMJudge × N  (llm_judge.py, temp=0.1)            │
│                       └─► PanelAggregator                            │
│                               ├─► median score per metric            │
│                               └─► MAD (inter-judge disagreement)     │
│  Output: <experiment>/judges/judge_scores_long.csv                   │
│          <experiment>/judges/panel_aggregated.csv                    │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 3 — THESIS ANALYSIS                                           │
│  analyze_experiment.py                                               │
│      └─► clustered bootstrap, efficiency summaries, judge agreement  │
│  Output: <experiment>/analysis/                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**Hard invariant**: judge models never see the candidate model ID — `sanitize_trace()` strips all metadata before evaluation.

---

## Project Structure

```
ai4care-master/
│
├── Core pipeline
│   ├── inference.py                # ChatRunner ABC: LlamaCppRunner, OpenAICompatibleRunner
│   ├── model_registry.py           # get_runner() factory, GGUF resolution and hardware offload
│   ├── engine.py                   # InterviewSession: state machine, extraction, RAG, generation
│   ├── knowledge.py                # ChromaDB helpers (single collection, namespaced)
│   ├── synthetic_respondent.py     # Deterministic word-overlap respondent (NO LLM)
│   ├── llm_judge.py                # Blinded LLM judge with JSON scrubbing and retry
│   ├── panel_aggregator.py         # Median + MAD aggregation across N judges
│   ├── evaluate.py                 # Trace generation + deterministic metrics
│   ├── analyze_experiment.py       # RQ1/RQ2/RQ3 analysis artifacts
│   └── cli.py                      # Interactive interview terminal (manual mode)
│
├── Data pipeline
│   ├── curate_seed_corpus.py       # Curate, chunk, classify, ingest corpus → ChromaDB
│   ├── compile_benchmark_dataset.py# Compile frozen benchmark Q/A JSONL files
│   └── setup.py                    # Initialise the shared Chroma collection
│
├── Hardware & model selection
│   ├── estimator.py                # Hardware detection + VRAM estimation
│   ├── judge_selector.py           # Hardware-aware judge model selection
│   ├── judge_benchmark.py          # Full automated judge benchmarking pipeline
│   └── verify_models.py            # Verify downloaded GGUF files are intact
│
├── Tests
│   └── tests/
│       ├── test_engine.py          # State machine, gap selection, extraction unit tests
│       ├── test_judge.py           # Judge parsing, panel aggregation unit tests
│       ├── test_suite.py           # Integration test suite
│       └── test_judge_benchmark.py # Benchmark pipeline tests (mocked hardware)
│
├── Notebooks
│   └── notebooks/judge_benchmark_analysis.ipynb
│
├── Data & results  (git-ignored)
│   ├── data/benchmark/             # benchmark_cases.jsonl, benchmark_facts.jsonl
│   ├── data/corpus/                # curated_corpus.jsonl
│   ├── .chroma_db/                 # ChromaDB persistent store
│   ├── .models/                    # Downloaded GGUF files
│   └── results/experiments/
│       ├── pilot/<run-id>/         # Small check run; never mixed with full output
│       └── full/<run-id>/          # Proof-of-concept run: all benchmark variants once
│           ├── evaluation_traces.jsonl
│           ├── evaluation_results.csv
│           ├── performance_profile.json
│           ├── judges/             # Raw, long, and panel judge results
│           └── analysis/            # Research-question report and tables
│
└── docs/
    ├── PROJECT_MAP.md              # Guide to maintained files and run order
    └── evaluation_metrics.md       # Full metrics reference
```

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If `llama-cpp-python` fails on Windows, install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first, then re-run the install.

### 3. Initialise ChromaDB *(run once)*

```bash
python setup.py
```

This creates the shared `ai4care_knowledge` collection. The thesis workflow selects eight compact interviewer models using the hardware estimator and downloads each GGUF on demand into `.models/`.

### 4. Curate the knowledge corpus *(run once)*

```bash
python curate_seed_corpus.py
```

Downloads [`ozguragrali/enterprise-knowledge-qa-dataset-gemini-flash-for-t5-large`](https://huggingface.co/datasets/ozguragrali/enterprise-knowledge-qa-dataset-gemini-flash-for-t5-large) from Hugging Face, deduplicates and classifies records into 18 canonical categories, chunks them into ~600-token segments with 75-token overlap, and upserts everything into ChromaDB under the `corpus` namespace.

### 5. Compile the frozen benchmark *(run once)*

```bash
python compile_benchmark_dataset.py
```

Generates `data/benchmark/benchmark_cases.jsonl` and `data/benchmark/benchmark_facts.jsonl`. **Do not regenerate these during or between evaluation runs** — they must stay identical so all candidate models are compared on the same cases.

---

## Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Key variables:

| Variable | Default | Description |
|---|---|---|
| `AI4CARE_BACKEND` | `local` | `local` (GGUF via llama-cpp) or `openai` (any OpenAI-compatible API) |
| `OPENAI_API_KEY` | — | Required when backend is `openai` |
| `OPENAI_API_BASE` | `http://localhost:8000/v1` | API base URL for remote backend |
| `AI4CARE_GPU_LAYERS` | auto | llama.cpp GPU layers (`-1` means all); auto falls back to CPU if accelerator loading fails |
| `AI4CARE_N_BATCH` | `512` | llama.cpp prompt batch size |

---

## Running Evaluations

### Thesis run, pipeline check, and exhaustive run

```bash
# Main thesis run: 12 balanced variants with 8 estimator-selected interviewer models
python evaluate.py --profile thesis --models all

# Optional pipeline check; do not report this as thesis evidence
python evaluate.py --profile pilot --models all

# Optional exhaustive run: every available benchmark variant, once per model
python evaluate.py --profile full --models all
```

Each invocation creates a timestamped folder under `results/experiments/<profile>/`. Run IDs encode profile, ablation, model count, iterations, cases, maximum replies, and UTC timestamp (for example, `thesis_none_8models_1i_12c_4t_<timestamp>`). The manifest records the estimator-selected model IDs and hardware estimates. `pilot` is a pipeline check; `thesis` is the compact balanced main study (12 variants, four from each of the three source groups); `full` evaluates every checked-in variant (currently 57) and takes longer. The benchmark still represents only three source documents, so variants do not provide additional independent source cases and conclusions must remain limited to this proof of concept. Turn progress, completed episodes, elapsed time, rolling ETA, and checkpoints are printed during the run.

By default, `--models all` calls the hardware estimator at Q4_K_M and 2,048-token context, filters to models that fit, prefers distinct model families, and records the exact panel in `run_manifest.json`. Keep this estimator choice for the thesis comparison; explicit model lists are for separately documented exploratory runs.

On a single local GPU, automatic worker selection uses one model at a time; on CPU or an OpenAI-compatible endpoint it uses concurrent model batches. Local runs reuse each loaded candidate model across its assigned episodes. Set `--workers` explicitly to tune CPU concurrency; several local workers share accelerator memory if you override the single-GPU default.

To resume an interrupted run, pass its folder back to the command with the same options:

```bash
python evaluate.py --profile thesis --models all --run-dir "results/experiments/thesis/<run-id>"
```

The printed folder path is the `<run-id>` to use. Completed traces are reused. `--skip-generation` only rebuilds deterministic metrics from traces already in that folder.

### Ablation conditions

```bash
# No RAG — disable ChromaDB retrieval entirely
python evaluate.py --profile custom --models general_main --ablation no_rag --cases 20

# State-machine ablation — suppress gap selection signal
python evaluate.py --profile custom --models general_main --ablation state_machine --cases 20
```

### Automated judge benchmarking

```bash
# Step 1: run the thesis profile and note its printed folder
python evaluate.py --profile thesis --models all

# Step 2: run three small and one large estimator-selected judge on a balanced sample
python judge_benchmark.py --experiment-dir "results/experiments/thesis/<run-id>"

# Step 3: generate the thesis analysis and RQ report
python analyze_experiment.py --experiment-dir "results/experiments/thesis/<run-id>"
```

The default judge panel is selected with `estimator.py`: three fitting models up to 4B and one fitting model at least 7B, if available on the detected hardware. Exact choices are saved in the summary. Every judge sees the same deterministic sample of one trace per interviewer-model/source-group stratum. For eight interviewers and three source groups, this is up to 24 traces and 96 judge scores. Set `--sample-per-model-source 0` to score every successful trace. The runner loads one local judge at a time, resumes completed judge/trace pairs, saves each completed score, and prints progress and ETA.

Judge artifacts are written under `<experiment>/judges/`: `judge_scores_long.csv` has one row per trace and judge, `panel_aggregated.csv` has panel medians and disagreement MAD, and `judge_benchmark_summary.json` records the estimator panel, sample, load time, latency, and parse success. Downloaded GGUF files are reused from `.models/`.

The analysis stage writes `analysis_report.md`, `analysis_summary.json`, and CSV tables for RQ1 bootstrap estimates and paired differences, RQ2 efficiency metrics, and RQ3 judge reliability. The thesis's reliability criterion is **Spearman ρ > 0.7 for at least two judge pairs**; ρ is not a p-value. The notebook at `notebooks/thesis_results_analysis.ipynb` explains metrics, decision rules, limitations, and visualizations. Its default is the newest `thesis` run and it never merges unrelated run IDs. Reference-fact F1 is scored against only full-source sentences actually disclosed by the synthetic respondent, so undisclosed sentences are not counted as extraction misses.

---

## Available Models

| ID | Model | Approx. size |
|---|---|---|
| `general_small` | TinyLlama 1.1B Chat Q4_K_M | ~670 MB |
| `general_main` | Qwen2.5 1.5B Instruct Q4_K_M | ~1.1 GB |
| `coder_small` | Qwen2.5-Coder 0.5B Q4_K_M | ~490 MB |
| `coder_main` | Qwen2.5-Coder 1.5B Q4_K_M | ~1.1 GB |

These registered IDs support older/manual demonstrations. The thesis panel comes from the estimator and may vary with hardware; its exact model IDs and estimates are recorded per run.

---

## Experimental Variables

| Type | Definition |
|---|---|
| **Independent** | The GGUF candidate model driving the interview |
| **Controlled** | ChromaDB collection, embedding function, corpus, benchmark JSONL, synthetic respondent, schema fields, all hyperparameters |
| **Ablations** | `no_rag` (no retrieval), `state_machine` (free-form questions, no gap signal) |
| **Dependent** | Schema completeness, reference-fact F1, turns per episode, novel facts per response, redundancy rate, gap-targeting accuracy, extraction method counts, follow-up question quality, and inter-judge MAD |

---

## Evaluation Metrics

### Automated (computed from traces)

| Metric | Description |
|---|---|
| `schema_completeness` | Weighted fraction of 9 schema fields reaching verified status (0–1) |
| `novel_facts_count` | Fields newly surfaced per participant turn |
| `redundancy_rate` | Fraction of questions with Jaccard overlap ≥ 0.85 to a prior question |
| `gap_targeting_rate` | Fraction of follow-up questions where the next participant turn yielded an extraction update for the target field |
| `extraction_method` | Per-turn counts of `llm`, `fallback` (keyword sentence-level), and empty extraction results |

### Judge-assessed (scored 0–4 per judge, aggregated as panel median)

| Metric key | What it measures |
|---|---|
| `follow_up_question_quality` | Overall follow-up quality: relevance, specificity, answerability, and coherence |

See [`docs/evaluation_metrics.md`](docs/evaluation_metrics.md) for the full reference.

---

## ChromaDB Architecture

- **One collection**: `ai4care_knowledge`
- **Embedding**: ChromaDB built-in default (a controlled constant — never changes between model runs)
- **Namespaces**:
  - `corpus` — ingested knowledge chunks (immutable after `curate_seed_corpus.py`)
  - `interview_fact` — atomic facts extracted per episode (written only when `persist_runtime=True`)
  - `interview_artifact` — episode-level summaries

> **Troubleshooting**: If ChromaDB throws a Rust panic after an upgrade, delete `.chroma_db/` and re-run `python setup.py` and `python curate_seed_corpus.py`.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Expected: all tests pass. The Chroma integration test requires a fresh `setup.py` run if `.chroma_db/` is missing.

---

## Interactive CLI (Manual Mode)

Run a live interview from the terminal against a real person (not the synthetic respondent):

```bash
python cli.py --model general_main

# Using a remote OpenAI-compatible API:
AI4CARE_BACKEND=openai OPENAI_API_KEY=your-key python cli.py --model general_main
```

In-session commands:

| Command | Action |
|---|---|
| `/schema` | Show the current knowledge state for all 9 fields |
| `/chroma` | Show document counts in ChromaDB |
| `/model <id>` | Switch to a different model mid-session |
| `/exit` | End the interview |

---

## Knowledge Schema

The interviewer targets nine structured fields per episode:

| Field | What it captures |
|---|---|
| `domain_area` | The subject domain of the knowledge scenario |
| `primary_challenge` | The main problem or difficulty described |
| `root_cause` | Why the problem occurred |
| `key_decisions` | Decisions made to address the challenge |
| `technologies_used` | Tools, platforms, and systems involved |
| `people_involved` | Teams, roles, and individuals mentioned |
| `processes_affected` | Workflows and procedures that were impacted |
| `operational_impact` | Outcomes and consequences of the actions taken |
| `lessons_learned` | Takeaways and recommendations for the future |

Each field transitions through states: `unknown → candidate → supported → verified` (or `conflicting`). An episode terminates when 75% of the maximum weighted score is reached or after 15 turns.
