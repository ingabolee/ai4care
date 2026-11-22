#!/bin/bash
#SBATCH --job-name=ai4care-judges-full
#SBATCH --partition=dl-gpu-a5000
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=56G
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=ai4care-judges-full-%j.out
#SBATCH --error=ai4care-judges-full-%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"

if [[ ! -f judge_benchmark.py || ! -f analyze_experiment.py ]]; then
  echo "ERROR: Submit this job from the AI4CARE project root." >&2
  exit 2
fi

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
elif [[ -f venv/bin/activate ]]; then
  source venv/bin/activate
else
  echo "ERROR: No .venv/bin/activate or venv/bin/activate found in $PROJECT_DIR." >&2
  exit 2
fi

if [[ $# -lt 1 || -z "$1" ]]; then
  echo "Usage: sbatch run_ai4care_full_judges.sbatch results/experiments/full/<run-id>" >&2
  exit 2
fi

EXPERIMENT_DIR="$1"
if [[ "$EXPERIMENT_DIR" != /* ]]; then
  EXPERIMENT_DIR="$PROJECT_DIR/$EXPERIMENT_DIR"
fi

if [[ ! -s "$EXPERIMENT_DIR/run_manifest.json" || \
      ! -s "$EXPERIMENT_DIR/evaluation_results.csv" || \
      ! -s "$EXPERIMENT_DIR/evaluation_traces.jsonl" ]]; then
  echo "ERROR: Experiment folder is missing its manifest, results CSV, or interview traces:" >&2
  echo "  $EXPERIMENT_DIR" >&2
  exit 2
fi

# Only proceed when this is a complete full interview run with no failed episodes.
python - "$EXPERIMENT_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
if manifest.get("profile") != "full":
    raise SystemExit(f"ERROR: Expected profile='full'; found {manifest.get('profile')!r}.")
generation_status = manifest.get("stage_status", {}).get("generation")
if generation_status != "complete":
    raise SystemExit(
        f"ERROR: Interview generation status is {generation_status!r}, not 'complete'. "
        "Finish or repair the full interview run before full judging."
    )

with (run_dir / "evaluation_results.csv").open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
successful = sum(row.get("status") == "success" for row in rows)
failed = sum(row.get("status") == "failed" for row in rows)
expected = manifest.get("expected_episode_count")
print(f"Interview input check: {successful} successful, {failed} failed, "
      f"{len(rows)} total; expected {expected}.")
if not rows or failed or expected is None or successful != int(expected):
    raise SystemExit("ERROR: The interview data is incomplete; refusing to call this a full judge run.")
PY

export OMP_NUM_THREADS=1
export OMP_WAIT_POLICY=PASSIVE
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "=== AI4CARE FULL JUDGE RUN ==="
echo "Experiment : $EXPERIMENT_DIR"
echo "Node       : $(hostname)"
echo "CPUs       : ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Judges     : estimator-selected target of 3 small + 1 large"
echo "Coverage   : every successful interview trace (no subsampling)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
fi

python -u judge_benchmark.py \
  --experiment-dir "$EXPERIMENT_DIR" \
  --top-n 4 \
  --small-judges 3 \
  --large-judges 1 \
  --context-length 4096 \
  --sample-per-model-source 0 \
  --sample-seed 42

# Refuse to produce a final analysis if the judge runner used a subset or could
# not load the requested four-model panel. Completed judge/trace scores can be
# resumed by submitting this same script with the same experiment folder.
python - "$EXPERIMENT_DIR/judges/judge_benchmark_summary.json" <<'PY'
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
print(f"Judge coverage check: {len(judges)} judges; {scored}/{available} traces; panel={status}.")
if (status != "complete" or len(judges) != 4 or scored != available
        or sampling.get("sample_per_model_source") != 0
        or sampling.get("strategy") != "all_successful_traces"):
    raise SystemExit("ERROR: The judge panel or trace coverage is partial; analysis was not run.")
PY

python -u analyze_experiment.py --experiment-dir "$EXPERIMENT_DIR"
echo "Full judge run and analysis finished at $(date)"
echo "Results: $EXPERIMENT_DIR/judges and $EXPERIMENT_DIR/analysis"
