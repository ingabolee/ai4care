# AI4CARE — Automated LLM Knowledge Elicitation Evaluation Framework

A fully automated, reproducible research framework for benchmarking how well different small, locally-run **decoder-only language models** perform as adaptive interviewers for tacit organisational knowledge elicitation. Every run is deterministic, resumable, and requires no cloud APIs or human participants.

---

## How It Works

The pipeline runs in two phases:

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
│               └─► Trace appended to results/evaluation_traces.jsonl  │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  PHASE 2 — BLINDED MoE JUDGE EVALUATION                              │
│                                                                      │
│  evaluate.py / judge_benchmark.py                                    │
│      └─► sanitize_trace()  ← strips model ID, timestamps, latency   │
│               └─► LLMJudge × N  (llm_judge.py, temp=0.1)            │
│                       └─► PanelAggregator                            │
│                               ├─► median score per metric            │
│                               └─► stdev (inter-judge disagreement)   │
│  Output: results/evaluation_results.csv                              │
│          results/judge_benchmark/*.csv                               │
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
│   ├── model_registry.py           # Model pool, get_runner() factory, GGUF resolution
│   ├── engine.py                   # InterviewSession: state machine, extraction, RAG, generation
│   ├── knowledge.py                # ChromaDB helpers (single collection, namespaced)
│   ├── synthetic_respondent.py     # Deterministic word-overlap respondent (NO LLM)
│   ├── llm_judge.py                # Blinded LLM judge with JSON scrubbing and retry
│   ├── panel_aggregator.py         # Median + stdev aggregation across N judges
│   ├── evaluate.py                 # Main 2-phase evaluation runner (CLI entry point)
│   └── cli.py                      # Interactive interview terminal (manual mode)
│
├── Data pipeline
│   ├── curate_seed_corpus.py       # Curate, chunk, classify, ingest corpus → ChromaDB
│   ├── compile_benchmark_dataset.py# Compile frozen benchmark Q/A JSONL files
│   └── seed_dataset.py             # Legacy seed script (superseded by curate_seed_corpus.py)
│
├── Hardware & model selection
│   ├── estimator.py                # Hardware detection + VRAM estimation
│   ├── judge_selector.py           # Hardware-aware judge model selection
│   ├── judge_benchmark.py          # Full automated judge benchmarking pipeline
│   └── verify_models.py            # Verify downloaded GGUF files are intact
│
├── Setup
│   └── setup.py                    # Download models + initialise ChromaDB (run once)
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
│   └── results/
│       ├── iterations/             # Per-iteration randomisation manifests
│       ├── evaluation_traces.jsonl # Raw interview traces (resumable)
│       ├── evaluation_results.csv  # Per-episode aggregate metrics
│       └── judge_benchmark/        # Judge scores (long CSV + panel CSV + summary JSON)
│
└── docs/
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

### 3. Download models and initialise ChromaDB *(run once)*

```bash
python setup.py
```

This downloads the four GGUF models to `.models/` and creates the `ai4care_knowledge` ChromaDB collection.

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

---

## Running Evaluations

### Full evaluation run

```bash
# Generate traces + judge all candidate models
python evaluate.py --models all --judges general_main --iterations 3 --cases 50

# Skip generation — re-judge existing traces only
python evaluate.py --skip-generation --judges general_main,coder_main
```

### Ablation conditions

```bash
# No RAG — disable ChromaDB retrieval entirely
python evaluate.py --models general_main --ablation no_rag --cases 20

# State-machine ablation — suppress gap selection signal
python evaluate.py --models general_main --ablation state_machine --cases 20
```

### Automated judge benchmarking

```bash
# Step 1: generate traces first
python evaluate.py --models all --cases 50

# Step 2: auto-detect hardware, select judges, score all traces
python judge_benchmark.py --top-n 3

# Step 3: open the analysis notebook
# notebooks/judge_benchmark_analysis.ipynb
```

The benchmark pipeline saves:
- `results/judge_benchmark/judge_scores_long.csv` — tidy format: one row per (episode, judge, metric)
- `results/judge_benchmark/panel_aggregated.csv` — one row per (episode, metric) with median + stdev
- `results/judge_benchmark/judge_benchmark_summary.json` — hardware profile + per-judge parsing success rates

---

## Available Models

| ID | Model | Approx. size |
|---|---|---|
| `general_small` | TinyLlama 1.1B Chat Q4_K_M | ~670 MB |
| `general_main` | Qwen2.5 1.5B Instruct Q4_K_M | ~1.1 GB |
| `coder_small` | Qwen2.5-Coder 0.5B Q4_K_M | ~490 MB |
| `coder_main` | Qwen2.5-Coder 1.5B Q4_K_M | ~1.1 GB |

To add a new candidate model, add an entry to `MODEL_CONFIGS` in `model_registry.py`.

---

## Experimental Variables

| Type | Definition |
|---|---|
| **Independent** | The GGUF candidate model driving the interview |
| **Controlled** | ChromaDB collection, embedding function, corpus, benchmark JSONL, synthetic respondent, schema fields, all hyperparameters |
| **Ablations** | `no_rag` (no retrieval), `state_machine` (free-form questions, no gap signal) |
| **Dependent** | Schema completeness, turns per episode, novel facts per turn, redundancy rate, gap-targeting accuracy, extraction method distribution, 6 judge-assessed metric scores, inter-judge stdev |

---

## Evaluation Metrics

### Automated (computed from traces)

| Metric | Description |
|---|---|
| `schema_completeness` | Weighted fraction of 9 schema fields reaching verified status (0–1) |
| `novel_facts_count` | Fields newly surfaced per participant turn |
| `redundancy_rate` | Fraction of questions with Jaccard overlap ≥ 0.85 to a prior question |
| `gap_targeting_accuracy` | Fraction of turns where the targeted field was actually progressed |
| `extraction_method` | `llm` (primary JSON path) vs `fallback` (keyword sentence-level) |

### Judge-assessed (scored 0–10 per judge, aggregated as panel median)

| Metric key | What it measures |
|---|---|
| `turn_level_completeness_efficiency` | Schema-relevant information extracted per turn |
| `turn_level_contradiction_rate` | Degree of logical contradictions in the interviewer's questions (lower = better) |
| `overall_elicit_efficiency` | Holistic judgment of how effectively the interview covered the knowledge scenario |
| `participant_engagement` | Whether the questions feel natural and would sustain a real participant |
| `redundancy_penalty` | Extent of topic repetition without adding new angles (higher = worse) |
| `hallucination_penalty` | Interviewer questions presupposing facts not stated by the participant (higher = worse) |

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
