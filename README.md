# AI4CARE — Knowledge Elicitation CLI

A command-line tool that conducts AI-assisted interviews to capture tacit knowledge from domain experts. It uses small local GGUF language models and ChromaDB for retrieval-augmented generation. No cloud APIs required.

---

## Project Structure

```
cli/
  setup.py          # Downloads models and initialises ChromaDB  [run once]
  seed_dataset.py   # Loads an external enterprise dataset into ChromaDB  [run once]
  engine.py         # Interview session logic and LLM question generation
  knowledge.py      # ChromaDB query, store, and inspect helpers
  cli.py            # Interactive terminal entry point
```

---

## Step 1 — Create a Virtual Environment

```bash
# Create the environment
python -m venv .venv

# Activate it — Windows
.venv\Scripts\activate

# Activate it — Mac / Linux
source .venv/bin/activate
```

---

## Step 2 — Install Dependencies

Install `llama-cpp-python` using the CPU-only pre-built wheel (no compiler needed):

```bash
python -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

Then install the remaining dependencies:

```bash
pip install chromadb huggingface_hub datasets
```

> **Windows note:** If the CPU wheel above fails, install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and re-run `pip install llama-cpp-python`.

---

## Step 3 — Run Setup *(once)*

This downloads all four GGUF models from Hugging Face and creates the ChromaDB collections.

```bash
python setup.py
```

Models are saved to `.models/` and the database to `.chroma_db/`. This can take a few minutes depending on your connection.

---

## Step 4 — Seed the Knowledge Base *(once)*

This loads an external enterprise knowledge dataset into ChromaDB to give the interviewer richer context.

```bash
python seed_dataset.py
```

Dataset source: [`ozguragrali/enterprise-knowledge-qa-dataset-gemini-flash-for-t5-large`](https://huggingface.co/datasets/ozguragrali/enterprise-knowledge-qa-dataset-gemini-flash-for-t5-large)

---

## Step 5 — Start an Interview

```bash
python cli.py
```

To choose a specific model:

```bash
python cli.py --model general_main
```

### Available Models

| ID | Model | Size |
|---|---|---|
| `general_small` | TinyLlama 1.1B Chat | ~670 MB |
| `general_main` | Qwen2.5 1.5B Instruct | ~1.1 GB |
| `coder_small` | Qwen2.5-Coder 0.5B | ~490 MB |
| `coder_main` | Qwen2.5-Coder 1.5B | ~1.1 GB |

---

## In-Session Commands

Type any of these at the `You:` prompt:

| Command | Action |
|---|---|
| `/schema` | Show the current knowledge map |
| `/model <id>` | Switch models mid-interview |
| `/chroma` | Show document counts per collection |
| `/exit` | End the session |

---

## How It Works

1. The interviewer opens with a question about a recent project or challenge.
2. You respond naturally at the `You:` prompt.
3. After each response, the engine extracts entities (technologies, roles, decisions, processes) and updates a knowledge schema.
4. The active local model generates the next question, informed by the conversation history and relevant ChromaDB documents.
5. The interview ends after 15 turns or when 7 of 9 schema fields are filled. The final knowledge artifact is stored back into ChromaDB.

All data stays on your machine.
