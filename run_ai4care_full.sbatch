#!/bin/bash
#SBATCH --job-name=ai4care-full
# Compute-node check found 16 logical CPUs, 128 GB node RAM, and one RTX A5000
# with 24 GB VRAM. Request the node's CPU capacity and a safe 56 GB RAM allocation.
#SBATCH --partition=dl-gpu-a5000
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=56G
#SBATCH --time=12:00:00
#SBATCH --signal=USR1@300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=ai4care-full-%j.out
#SBATCH --error=ai4care-full-%j.err

set -euo pipefail

# Submit this file from the project root so Slurm can find the code and data.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"
JOB_TAG="${1:-${SLURM_JOB_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)}}"
ACTIVE_CHILD=""

run_stage() {
  "$@" &
  ACTIVE_CHILD=$!
  local exit_code=0
  wait "$ACTIVE_CHILD" || exit_code=$?
  ACTIVE_CHILD=""
  return "$exit_code"
}

continue_before_walltime() {
  trap - USR1
  echo "Walltime signal received; checkpoints are saved under run tag $JOB_TAG."
  if [[ -n "$ACTIVE_CHILD" ]]; then
    wait "$ACTIVE_CHILD" 2>/dev/null || true
  fi
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: Cannot queue a continuation outside Slurm." >&2
    exit 1
  fi
  echo "Submitting a continuation after job $SLURM_JOB_ID..."
  if ! sbatch --dependency="afterany:${SLURM_JOB_ID}" \
      "$PROJECT_DIR/run_ai4care_full.sbatch" "$JOB_TAG"; then
    echo "ERROR: Slurm continuation submission failed." >&2
    exit 1
  fi
  exit 0
}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  trap continue_before_walltime USR1
fi

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

# Keep OpenMP/BLAS-based embedding and numeric libraries from each spawning a
# full host-sized pool inside every model worker. ONNX Runtime also respects
# these settings when its build uses OpenMP.
export OMP_NUM_THREADS=1
export OMP_WAIT_POLICY=PASSIVE
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK:-unknown}; requested memory: 56G"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
else
  echo "GPU diagnostic: nvidia-smi is unavailable in this job environment."
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
run_stage python -u setup.py
if [[ "$NEEDS_CORPUS_SETUP" -eq 1 ]]; then
  echo "[SETUP] Ingesting the curated knowledge corpus (first run on this server)..."
  run_stage python -u curate_seed_corpus.py
fi

# A stable run tag is carried into automatic walltime continuation jobs, so
# evaluate.py and judge_benchmark.py resume from the same saved checkpoints.
RUN_DIR="$PROJECT_DIR/results/experiments/full/full_none_estimator_8models_1i_allcases_10t_${JOB_TAG}"
mkdir -p "$PROJECT_DIR/results/experiments/full"

echo "=== AI4CARE FULL RUN: $JOB_TAG ==="
echo "Experiment output: $RUN_DIR"
echo "Models: eight compact interviewer models selected by estimator.py"
echo "Cases: every eligible frozen benchmark variant, once per model"
echo "Interview limit: 10 participant replies; stop early at 75% schema completeness"
echo "Judges: estimator-selected panel (target: 3 small + 1 large), every successful trace"

echo "[1/3] Generating the full interview evaluation..."
run_stage python -u evaluate.py \
  --profile full \
  --models all \
  --iterations 1 \
  --max-turns 10 \
  --seed 1234 \
  --ablation none \
  --run-dir "$RUN_DIR"

# evaluate.py records failed episodes in its results CSV but may still exit 0.
# Print the common causes and stop before judging an incomplete/failed run.
python - "$RUN_DIR/evaluation_results.csv" <<'PY'
import csv
import json
import sys
from collections import Counter
from pathlib import Path

path = sys.argv[1]
run_dir = Path(path).parent
manifest_path = run_dir / "run_manifest.json"
if not manifest_path.is_file():
    sys.exit(f"ERROR: Run manifest is missing: {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
generation_status = manifest.get("stage_status", {}).get("generation")
expected = manifest.get("expected_episode_count")
with open(path, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
failed_rows = [row for row in rows if row.get("status") == "failed"]
successes = sum(row.get("status") == "success" for row in rows)
print(f"Interview results: {successes} succeeded, {len(failed_rows)} failed, {len(rows)} total; "
      f"expected {expected}; generation={generation_status}.")
if failed_rows:
    reasons = Counter((row.get("failure_reason") or "No failure reason recorded") for row in failed_rows)
    for reason, count in reasons.most_common(5):
        print(f"  Failure ({count}): {reason[:500]}")
if not rows or failed_rows or expected is None or successes != int(expected) or generation_status != "complete":
    sys.exit("ERROR: Full interview generation is incomplete; refusing to start judges.")
PY

echo "[2/3] Scoring every successful interview trace with the estimator-selected judge panel..."
run_stage python -u judge_benchmark.py \
  --experiment-dir "$RUN_DIR" \
  --top-n 4 \
  --small-judges 3 \
  --large-judges 1 \
  --context-length 4096 \
  --sample-per-model-source 0 \
  --sample-seed 42

python - "$RUN_DIR/judges/judge_benchmark_summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"ERROR: Judge summary was not created: {path}")
summary = json.loads(path.read_text(encoding="utf-8"))
judges = summary.get("judges_used", [])
sampling = summary.get("trace_sampling", {})
available = int(summary.get("available_traces", 0))
scored = int(summary.get("total_traces", 0))
status = summary.get("judge_panel_status")
per_judge = summary.get("per_judge_summary", {})
failed_scores = sum(int(item.get("parse_failures", 0)) for item in per_judge.values())
expected_evaluations = available * len(judges)
evaluations = int(summary.get("total_judge_evaluations", 0))
print(f"Judge coverage check: {len(judges)} judges; {scored}/{available} traces; panel={status}.")
if (status != "complete" or len(judges) != 4 or scored != available
        or evaluations != expected_evaluations or failed_scores != 0
        or sampling.get("sample_per_model_source") != 0
        or sampling.get("strategy") != "all_successful_traces"):
    raise SystemExit(
        f"ERROR: Judge panel, scoring coverage, or parse success is incomplete "
        f"({evaluations}/{expected_evaluations} evaluations; {failed_scores} failed scores); "
        "analysis was not run."
    )
PY

echo "[3/3] Producing the research-question analysis..."
run_stage python -u analyze_experiment.py --experiment-dir "$RUN_DIR"

echo "AI4CARE full evaluation finished at $(date)"
echo "Results and analysis: $RUN_DIR"
