#!/bin/bash
#SBATCH --job-name=ai4care-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=ai4care-full-%j.out
#SBATCH --error=ai4care-full-%j.err

set -euo pipefail

# Submit this file from the project root so Slurm can find the code and data.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"

if [[ ! -f evaluate.py || ! -f judge_benchmark.py || ! -f analyze_experiment.py ]]; then
  echo "ERROR: Submit this job from the AI4CARE project root (evaluate.py not found)." >&2
  exit 2
fi

# Accept either virtual-environment name used by this project and older setups.
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
elif [[ -f venv/bin/activate ]]; then
  source venv/bin/activate
else
  echo "ERROR: No .venv/bin/activate or venv/bin/activate found in $PROJECT_DIR." >&2
  exit 2
fi

# The benchmark is frozen input. Do not rebuild it as part of a run.
for required_file in \
  data/benchmark/benchmark_cases.jsonl \
  data/benchmark/benchmark_facts.jsonl; do
  if [[ ! -s "$required_file" ]]; then
    echo "ERROR: Required benchmark file is missing or empty: $required_file" >&2
    echo "Copy the checked-in benchmark data to the server; do not regenerate it during this run." >&2
    exit 2
  fi
done

# Initialize the shared Chroma collection. If the corpus has not been ingested
# on this server, build it once from the project's pinned corpus source.
NEEDS_CORPUS_SETUP=0
[[ -d .chroma_db ]] || NEEDS_CORPUS_SETUP=1
[[ -s data/manifests/seed_corpus_manifest.json ]] || NEEDS_CORPUS_SETUP=1
[[ -s data/corpus/curated_corpus.jsonl ]] || NEEDS_CORPUS_SETUP=1

echo "[SETUP] Checking the shared knowledge collection..."
python -u setup.py
if [[ "$NEEDS_CORPUS_SETUP" -eq 1 ]]; then
  echo "[SETUP] Ingesting the curated knowledge corpus (first run on this server)..."
  python -u curate_seed_corpus.py
fi

# A unique job-specific folder separates this full run from pilot/test runs.
# Slurm requeues keep the same ID, allowing evaluate.py and judge_benchmark.py
# to resume from their saved checkpoints.
JOB_TAG="${SLURM_JOB_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$PROJECT_DIR/results/experiments/full/full_none_estimator_8models_1i_allcases_4t_${JOB_TAG}"
mkdir -p "$PROJECT_DIR/results/experiments/full"

echo "=== AI4CARE FULL RUN: $JOB_TAG ==="
echo "Experiment output: $RUN_DIR"
echo "Models: eight compact interviewer models selected by estimator.py"
echo "Cases: every eligible frozen benchmark variant, once per model"
echo "Judges: estimator-selected panel (target: 3 small + 1 large), balanced sample"

echo "[1/3] Generating the full interview evaluation..."
python -u evaluate.py \
  --profile full \
  --models all \
  --iterations 1 \
  --max-turns 4 \
  --seed 1234 \
  --ablation none \
  --run-dir "$RUN_DIR"

echo "[2/3] Scoring a balanced sample with the estimator-selected judge panel..."
python -u judge_benchmark.py \
  --experiment-dir "$RUN_DIR" \
  --top-n 4 \
  --small-judges 3 \
  --large-judges 1 \
  --context-length 4096 \
  --sample-per-model-source 1 \
  --sample-seed 42

echo "[3/3] Producing the research-question analysis..."
python -u analyze_experiment.py --experiment-dir "$RUN_DIR"

echo "AI4CARE full evaluation finished at $(date)"
echo "Results and analysis: $RUN_DIR"
