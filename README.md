# AI4CARE — Research-Grade Evaluation Framework for LLM-Based Knowledge Elicitation

A reproducible, automated research framework for evaluating the effectiveness of different **decoder-only language models** as adaptive interviewers for tacit organisational knowledge elicitation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: GENERATION                                         │
│                                                              │
│  evaluate.py ──► InterviewSession (engine.py)               │
│                        │                                     │
│                   ChatRunner (inference.py)                  │
│                   [LlamaCppRunner | OpenAICompatibleRunner]  │
│                        │                                     │
│               SyntheticRespondent answers                    │
│                        │                                     │
│               Trace saved → evaluation_traces.jsonl          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  PHASE 2: BLINDED MoE EVALUATION                             │
│                                                              │
│  evaluate.py ──► LLMJudge × N (llm_judge.py)               │
│                     (candidate model ID stripped)            │
│                        │                                     │
│              PanelAggregator (panel_aggregator.py)           │
│               Median scores + stdev disagreement             │
│                        │                                     │
│              Results → evaluation_results.csv                │
└─────────────────────────────────────────────────────────────┘
```

**Key invariant**: Judge models never see the candidate model ID. All timing and metadata is stripped before evaluation.

---

## Project Structure

```
ai4care-master/
├── inference.py                  # ChatRunner ABC (LlamaCppRunner, OpenAICompatibleRunner)
├── model_registry.py             # Model pool + get_runner() factory
├── engine.py                     # InterviewSession: extraction, state machine, generation
├── knowledge.py                  # ChromaDB helpers (single collection)
├── llm_judge.py                  # Blinded LLM judge with robust JSON parsing
├── panel_aggregator.py           # Median aggregation + disagreement stats from N judges
├── evaluate.py                   # 2-phase evaluation runner (generation + judging)
├── cli.py                        # Interactive interview CLI
├── synthetic_respondent.py       # Deterministic synthetic participant
├── setup.py                      # Download models + init Chroma
├── curate_seed_corpus.py         # Corpus curation + Chroma ingestion
├── compile_benchmark_dataset.py  # Compile frozen benchmark Q/A
├── .env.example                  # Environment variable documentation
├── docs/
│   └── evaluation_metrics.md     # Full metrics reference
├── tests/
│   ├── test_engine.py            # Unit tests: state machine, gap selection, extraction
│   ├── test_judge.py             # Unit tests: judge parsing, panel aggregation
│   └── test_suite.py             # Integration test suite
└── results/
    ├── iterations/               # Per-iteration randomization manifests
    ├── evaluation_traces.jsonl   # Raw interview traces (resumable)
    └── evaluation_results.csv    # Final per-episode metrics
```

---

## Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Mac/Linux
```

### 2. Install Dependencies

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
pip install chromadb huggingface_hub datasets pytest requests
```

> **Windows note:** If the wheel fails, install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first.

### 3. Download Models + Initialise Chroma

```bash
python setup.py
```

### 4. Curate Corpus + Compile Benchmark

```bash
python curate_seed_corpus.py
python compile_benchmark_dataset.py
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```env
AI4CARE_BACKEND=local           # 'local' (GGUF) or 'openai' (any OpenAI-compatible API)
OPENAI_API_KEY=your-key         # Used when AI4CARE_BACKEND=openai
OPENAI_API_BASE=http://localhost:8000/v1
```

---

## Running Evaluation

```bash
# Full run: generate traces + judge them
python evaluate.py --models all --judges general_main --iterations 3 --cases 50

# Ablation: no RAG
python evaluate.py --models general_small --ablation no_rag --cases 20

# Ablation: state-machine bypassed
python evaluate.py --models general_main --ablation state_machine --cases 20

# Skip generation (re-judge existing traces only)
python evaluate.py --skip-generation --judges general_main,coder_main

# Multiprocessing
python evaluate.py --models all --workers 4 --cases 100
```

### Experimental Protocol

1. One **randomized benchmark assignment** is created per iteration (shared across all models).
2. Every model runs the exact same benchmark cases in that iteration.
3. Every episode starts with a **clean runtime state** — no carryover between episodes.
4. Traces are saved to `evaluation_traces.jsonl` — **runs can be safely resumed**.
5. Judges are loaded **sequentially** to avoid OOM on local hardware.

---

## Interactive CLI (Manual Mode)

```bash
python cli.py --model general_main
# Remote API:
AI4CARE_BACKEND=openai OPENAI_API_KEY=xxx python cli.py --model general_main
```

In-session commands: `/schema`, `/chroma`, `/model <id>`, `/exit`

---

## Models

| ID | Model | Size |
|---|---|---|
| `general_small` | TinyLlama 1.1B Chat | ~670 MB |
| `general_main` | Qwen2.5 1.5B Instruct | ~1.1 GB |
| `coder_small` | Qwen2.5-Coder 0.5B | ~490 MB |
| `coder_main` | Qwen2.5-Coder 1.5B | ~1.1 GB |

Add any decoder-only GGUF model to `model_registry.py`.

---

## Metrics

See [`docs/evaluation_metrics.md`](docs/evaluation_metrics.md) for the full reference.

**Summary of key metrics per episode:**

| Category | Metric |
|----------|--------|
| Reference-based | `schema_completeness`, `reference_fact_recall`, `reference_fact_f1`, `contradiction_rate` |
| Tacit-knowledge novelty | `novel_facts_count` per turn, `newly_surfaced_fields` |
| Judge-based (MoE) | `turn_level_completeness_efficiency`, `overall_elicit_efficiency`, `participant_engagement` |
| Efficiency | `generation_latency_ms`, `redundancy_rate`, `gap_targeting_rate` |

---

## Chroma Architecture

- **One collection**: `ai4care_knowledge`
- **Embedding**: Chroma's built-in default (controlled constant across all models)
- **Namespaces**: `corpus` (immutable), `interview_fact` (per-episode), `interview_artifact` (summaries)

> If you upgrade ChromaDB and get a Rust panic, delete `data/.chroma_db` and re-run `python setup.py`.

---

## Experimental Variables

| Type | Variable |
|---|---|
| **Independent** | Decoder-only GGUF generator model |
| **Controlled** | Chroma collection, embedding function, seed corpus, benchmark, synthetic respondent, schema |
| **Ablations** | `no_rag` (no retrieval), `state_machine` (free-form questioning) |
| **Dependent** | All metrics above |

---

## Running Tests

```bash
python -m pytest tests/ -v
# Expected: 75 passed, 1 skipped (Chroma env check — requires fresh setup.py)
```

