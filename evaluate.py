"""
evaluate.py -- Main automated evaluation runner for the AI4CARE experiment.

Evaluation schema version: v2

Key improvements over v1:
  - Collision-resistant run_id includes full case_id, variant_id, ablation, config fingerprint.
  - Config fingerprint (SHA-256) verifies compatibility before resuming cached runs.
  - Benchmark hash recorded for provenance.
  - Production evaluation metrics module wired in (reference-claim matching etc.).
  - Single follow_up_question_quality judge metric with strict validation.
  - Failed runs included in output with null metric values.
  - Null (not zero) for unavailable or not-applicable metrics.

Generation seed note:
  LlamaCppRunner accepts a seed parameter. The effective per-run seed is derived
  deterministically from (iteration_seed, model_id, case_id, variant_id).
  OpenAI-compatible runners forward the seed parameter but cannot guarantee
  reproducibility — this limitation is recorded in trace metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
import uuid
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from model_registry import MODEL_CONFIGS, CHROMA_COLLECTION, get_enabled_models, get_runner
from knowledge import delete_episode_records
from engine import InterviewSession, schema_completeness
from synthetic_respondent import SyntheticRespondent
from llm_judge import LLMJudge, METRIC as JUDGE_METRIC, JUDGE_SCHEMA_VERSION
from panel_aggregator import aggregate_judge_results
from evaluation_metrics import compute_all_metrics, sensitivity_sweep, EVALUATION_SCHEMA_VERSION

# ─── PATHS ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")
ITERATIONS_DIR = RESULTS_DIR / "iterations"
BENCHMARK_CASES_PATH = Path("data/benchmark/benchmark_cases.jsonl")
BENCHMARK_FACTS_PATH = Path("data/benchmark/benchmark_facts.jsonl")
BENCHMARK_MANIFEST_PATH = Path("data/benchmark/benchmark_manifest.json")

TRACES_FILE = RESULTS_DIR / "evaluation_traces.jsonl"
CSV_RESULTS = RESULTS_DIR / "evaluation_results_v2.csv"
FAILURES_FILE = RESULTS_DIR / "evaluation_failures_v2.jsonl"

# ─── EXPERIMENT CONFIG ────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    dataset_name: str = "ozguragrali/enterprise-knowledge-qa"
    dataset_version: str = "v1"
    model_ids: list = field(default_factory=list)
    judge_ids: list = field(default_factory=list)
    chroma_collection: str = CHROMA_COLLECTION
    # Generation settings actually propagated to engine.py / runner.generate()
    # temperature for extraction: 0.0 (hard-coded in engine._extract)
    # temperature for question gen: 0.7 (hard-coded in engine._generate_question)
    # These are documented here but engine.py does not currently accept overrides
    # from ExperimentConfig. Recorded as effective values for provenance.
    effective_extraction_temperature: float = 0.0
    effective_question_temperature: float = 0.7
    max_tokens: int = 200              # Passed to extraction (400) and question gen (200)
    n_ctx: int = 2048                  # Used by LlamaCppRunner at init
    top_k: int = 5                     # RAG_TOP_K used by engine.py
    max_turns: int = 15                # MAX_TURNS used by engine.py
    completion_threshold: float = 0.75 # COMPLETION_THRESHOLD used by engine.py
    iterations: int = 3
    cases_per_iteration: int = 50
    sampling_mode: str = "proportional"
    random_seed: int = 1234
    ablation: str = "none"

    def fingerprint(self) -> str:
        """SHA-256 fingerprint of the effective generation configuration.

        Used to determine whether a cached trace is compatible with the current run.
        Fields not affecting generation output (e.g. judge_ids, output paths) are
        excluded from the fingerprint.
        """
        canonical = {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "model_ids": sorted(self.model_ids),
            "effective_extraction_temperature": self.effective_extraction_temperature,
            "effective_question_temperature": self.effective_question_temperature,
            "max_turns": self.max_turns,
            "completion_threshold": self.completion_threshold,
            "top_k": self.top_k,
            "ablation": self.ablation,
            "sampling_mode": self.sampling_mode,
            "random_seed": self.random_seed,
        }
        raw = json.dumps(canonical, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def to_dict(self) -> dict:
        return asdict(self)


# ─── RUN IDENTITY ─────────────────────────────────────────────────────────────

def make_run_id(
    iteration_id: str,
    model_id: str,
    case_id: str,
    question_variant_id: str,
    ablation: str,
    config_fingerprint: str,
) -> str:
    """Create a collision-resistant, stable run identifier.

    Components:
      - iteration_id: which iteration of case sampling
      - model_id: the candidate model being evaluated
      - case_id: the full benchmark case_id (NOT truncated)
      - question_variant_id: the specific question variant
      - ablation: experimental condition affecting generation
      - config_fingerprint: SHA-256 of effective generation config

    Returns a 16-character SHA-256 prefix (lower hex) that is deterministic for
    these inputs. The full components are stored separately in trace metadata
    for reproducibility — the ID alone is not sufficient for full provenance.
    """
    components = "|".join([
        iteration_id,
        model_id,
        case_id,
        question_variant_id,
        ablation,
        config_fingerprint[:16],   # use prefix of fingerprint to limit length
    ])
    return hashlib.sha256(components.encode()).hexdigest()[:16]


def derive_run_seed(iteration_seed: int, model_id: str, case_id: str, variant_id: str) -> int:
    """Derive a deterministic per-run seed for candidate generation.

    This seed is passed to runner.generate() to improve reproducibility for
    backends that support seed-based generation (e.g. LlamaCppRunner).

    OpenAI-compatible runners forward the seed parameter but reproducibility
    is NOT guaranteed by the specification.
    """
    raw = f"{iteration_seed}|{model_id}|{case_id}|{variant_id}"
    return int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16)


# ─── BENCHMARK LOADING ────────────────────────────────────────────────────────

def compute_file_hash(path: Path) -> Optional[str]:
    """Compute SHA-256 hash of a file. Returns None if file doesn't exist."""
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_benchmark() -> tuple[list[dict], dict[str, list[dict]], str]:
    """Load benchmark cases and reference facts.

    Returns:
        cases: list of benchmark case dicts
        facts_by_case: mapping of case_id → list of fact dicts
        benchmark_hash: SHA-256 fingerprint of benchmark files for provenance

    Raises FileNotFoundError if benchmark_cases.jsonl does not exist.
    """
    if not BENCHMARK_CASES_PATH.exists():
        raise FileNotFoundError(
            f"Benchmark not found at {BENCHMARK_CASES_PATH}. "
            "Run compile_benchmark_dataset.py first."
        )

    cases: list[dict] = []
    with open(BENCHMARK_CASES_PATH, encoding="utf-8") as f:
        for line in f:
            cases.append(json.loads(line))

    facts_by_case: dict[str, list[dict]] = {}
    if BENCHMARK_FACTS_PATH.exists():
        with open(BENCHMARK_FACTS_PATH, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                facts_by_case.setdefault(rec["case_id"], []).append(rec)

    cases_hash = compute_file_hash(BENCHMARK_CASES_PATH) or "unknown"
    facts_hash = compute_file_hash(BENCHMARK_FACTS_PATH) or "unknown"
    benchmark_hash = hashlib.sha256(
        f"{cases_hash}|{facts_hash}".encode()
    ).hexdigest()[:16]

    return cases, facts_by_case, benchmark_hash


def create_iteration_manifest(
    iteration_id: str,
    cases: list[dict],
    n_cases: int,
    seed: int,
    config: ExperimentConfig,
    benchmark_hash: str,
) -> dict:
    rng = random.Random(seed)
    eligible = [c for c in cases if c.get("answer", "").strip()]

    if config.sampling_mode == "balanced":
        by_cat: dict[str, list[dict]] = {}
        for c in eligible:
            by_cat.setdefault(c["category"], []).append(c)
        cats = sorted(by_cat.keys())
        per_cat = max(1, n_cases // len(cats))
        selected: list[dict] = []
        for cat in cats:
            pool = by_cat[cat]
            rng.shuffle(pool)
            selected.extend(pool[:per_cat])
        selected = selected[:n_cases]
    else:
        selected = rng.sample(eligible, min(n_cases, len(eligible)))
    rng.shuffle(selected)

    manifest = {
        "iteration_id": iteration_id,
        "seed": seed,
        "dataset_version": config.dataset_version,
        "benchmark_hash": benchmark_hash,
        "config_fingerprint": config.fingerprint(),
        "sampling_mode": config.sampling_mode,
        "n_cases": len(selected),
        "cases": [
            {
                "case_id": c["case_id"],
                "case_group_id": c.get("case_group_id", ""),
                "question_variant_id": c["question_variant_id"],
                "variant_key": c.get("variant_key", ""),
                "category": c["category"],
            }
            for c in selected
        ],
        "category_distribution": {},
    }
    cat_dist: dict[str, int] = {}
    for c in selected:
        cat_dist[c["category"]] = cat_dist.get(c["category"], 0) + 1
    manifest["category_distribution"] = cat_dist

    ITERATIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ITERATIONS_DIR / f"iteration_{iteration_id}_cfg{config.fingerprint()[:8]}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# ─── EPISODE RUNNER ──────────────────────────────────────────────────────────

def run_episode(
    model_id: str,
    case: dict,
    iteration_id: str,
    iteration_seed: int,
    config: ExperimentConfig,
    benchmark_hash: str,
) -> dict:
    config_fp = config.fingerprint()
    case_id = case["case_id"]
    variant_id = case["question_variant_id"]

    run_id = make_run_id(
        iteration_id=iteration_id,
        model_id=model_id,
        case_id=case_id,
        question_variant_id=variant_id,
        ablation=config.ablation,
        config_fingerprint=config_fp,
    )
    run_seed = derive_run_seed(iteration_seed, model_id, case_id, variant_id)

    try:
        delete_episode_records(run_id)

        runner = get_runner(model_id, verbose=False)
        rag_enabled = config.ablation != "no_rag"

        session = InterviewSession(
            episode_id=run_id,
            model_id=model_id,
            runner=runner,
            rag_enabled=rag_enabled,
            persist_runtime=False,
            ablation=config.ablation,
        )

        respondent = SyntheticRespondent(
            case_id=case_id,
            reference_answer=case["answer"],
            iteration_seed=iteration_seed,
        )

        initial_q = case.get("question", session.opening_question)

        t_start = time.time()

        participant_answer = respondent.answer(initial_q, turn_id=0)["text"]
        state = session.respond(participant_answer)

        for turn_num in range(1, config.max_turns):
            if state.get("is_done"):
                break
            last_q = next(
                (t["text"] for t in reversed(state["turns"]) if t["role"] == "interviewer"),
                initial_q,
            )
            response = respondent.answer(last_q, turn_id=turn_num)
            state = session.respond(response["text"])

        total_runtime_ms = int((time.time() - t_start) * 1000)

        return {
            "run_id": run_id,
            "iteration_id": iteration_id,
            "model_id": model_id,
            "case_id": case_id,
            "case_group_id": case.get("case_group_id", ""),
            "question_variant_id": variant_id,
            "variant_key": case.get("variant_key", ""),
            "case_category": case.get("category", ""),
            "ablation": config.ablation,
            "config_fingerprint": config_fp,
            "benchmark_hash": benchmark_hash,
            "effective_extraction_temperature": config.effective_extraction_temperature,
            "effective_question_temperature": config.effective_question_temperature,
            "effective_seed": run_seed,
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
            "status": "success",
            "total_runtime_ms": total_runtime_ms,
            "trace": state,
        }

    except Exception as exc:
        return {
            "run_id": run_id,
            "iteration_id": iteration_id,
            "model_id": model_id,
            "case_id": case_id,
            "case_group_id": case.get("case_group_id", ""),
            "question_variant_id": variant_id,
            "variant_key": case.get("variant_key", ""),
            "case_category": case.get("category", ""),
            "ablation": config.ablation,
            "config_fingerprint": config_fp,
            "benchmark_hash": benchmark_hash,
            "status": "failed",
            "failure_reason": str(exc),
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        }


# ─── TRACE LOADING ────────────────────────────────────────────────────────────

def load_traces(config_fingerprint: str, benchmark_hash: str) -> dict:
    """Load existing traces, filtering to those compatible with the current config.

    A trace is considered compatible if its config_fingerprint and benchmark_hash
    match the current run. Incompatible traces are preserved in the file but not
    reused.

    Duplicate run_ids with conflicting content are detected and reported.
    """
    if not TRACES_FILE.exists():
        return {}

    traces: dict[str, dict] = {}
    conflicts: list[tuple[str, dict, dict]] = []

    with open(TRACES_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            run_id = rec.get("run_id")
            if not run_id:
                continue

            # Only reuse traces matching current config + benchmark
            if (rec.get("config_fingerprint") != config_fingerprint or
                    rec.get("benchmark_hash") != benchmark_hash):
                continue

            if run_id in traces:
                existing = traces[run_id]
                # Detect conflict: same run_id but different status or model
                if (existing.get("status") != rec.get("status") or
                        existing.get("model_id") != rec.get("model_id")):
                    conflicts.append((run_id, existing, rec))
                # Keep the most recent version (later in file) if statuses match
                traces[run_id] = rec
            else:
                traces[run_id] = rec

    if conflicts:
        print(f"  [WARNING] {len(conflicts)} duplicate run_id(s) with conflicting content detected:")
        for run_id, a, b in conflicts:
            print(f"    run_id={run_id}: "
                  f"status {a.get('status')} vs {b.get('status')}, "
                  f"model {a.get('model_id')} vs {b.get('model_id')}")

    return traces


def append_trace(record: dict) -> None:
    with open(TRACES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ─── CSV COLUMNS ─────────────────────────────────────────────────────────────

CSV_COLS = [
    # Identity and provenance
    "run_id", "iteration_id", "model_id", "case_id", "case_group_id",
    "question_variant_id", "variant_key", "case_category", "ablation",
    "config_fingerprint", "benchmark_hash", "evaluation_schema_version",
    "status", "failure_reason",
    # Generation settings
    "effective_extraction_temperature", "effective_question_temperature",
    "effective_seed",
    # Runtime
    "turn_count", "total_runtime_ms",
    # Primary outcome
    "reference_fact_f1", "reference_fact_precision", "reference_fact_recall",
    "matching_algorithm", "matching_threshold",
    "extracted_claim_count", "reference_fact_count", "matched_pair_count",
    "reference_metrics_status",
    # Secondary structural metrics
    "schema_completeness", "novel_fields_total", "novel_fields_per_response",
    # Process diagnostics
    "question_redundancy_rate", "redundant_question_count", "eligible_question_count",
    "gap_targeting_rate", "gap_aligned_count", "gap_eligible_count",
    # Knowledge state diagnostics
    "conflicting_field_rate", "conflicting_field_count",
    "evidence_presence_rate", "fields_with_evidence_count",
    # Judge score
    "follow_up_question_quality",
    "judge_disagreement_mad",
    "successful_judges", "total_judges",
    "judge_schema_version",
]


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI4CARE Evaluation Runner v2")
    parser.add_argument("--models", default="all", help="Comma-separated model IDs")
    parser.add_argument("--judges", default="general_main", help="Comma-separated judge model IDs")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--cases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ablation", choices=["none", "no_rag", "state_machine"], default="none")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()

    model_ids = get_enabled_models() if args.models == "all" else [m.strip() for m in args.models.split(",")]
    judge_ids = [m.strip() for m in args.judges.split(",")]

    print(f"\n=== AI4CARE Pipeline (schema v2) ===")
    print(f"  Models     : {model_ids}")
    print(f"  Judges     : {judge_ids}")
    print(f"  Iterations : {args.iterations} | Cases: {args.cases}")
    print(f"  Ablation   : {args.ablation}")

    config = ExperimentConfig(
        model_ids=model_ids,
        judge_ids=judge_ids,
        iterations=args.iterations,
        cases_per_iteration=args.cases,
        random_seed=args.seed,
        ablation=args.ablation,
    )
    config_fp = config.fingerprint()
    print(f"  Config fingerprint: {config_fp[:16]}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cases, facts_by_case, benchmark_hash = load_benchmark()
    print(f"  Benchmark hash: {benchmark_hash}")

    # ── PHASE 1: GENERATION ──────────────────────────────────────────────
    if not args.skip_generation:
        print(f"\n[PHASE 1] Trace Generation...")
        existing_traces = load_traces(config_fp, benchmark_hash)
        rng = random.Random(args.seed)

        episodes_to_run = []

        for iter_num in range(1, args.iterations + 1):
            iteration_id = f"{iter_num:04d}"
            iter_seed = rng.randint(0, 2**31)
            manifest = create_iteration_manifest(
                iteration_id, cases, args.cases, iter_seed, config, benchmark_hash
            )

            selected_vars = {c["question_variant_id"] for c in manifest["cases"]}
            ordered_cases = [c for c in cases if c["question_variant_id"] in selected_vars][:args.cases]

            for model_id in model_ids:
                for case in ordered_cases:
                    run_id = make_run_id(
                        iteration_id=iteration_id,
                        model_id=model_id,
                        case_id=case["case_id"],
                        question_variant_id=case["question_variant_id"],
                        ablation=config.ablation,
                        config_fingerprint=config_fp,
                    )
                    existing = existing_traces.get(run_id)
                    if existing and existing.get("status") == "success":
                        # Verify config compatibility (already filtered by load_traces)
                        continue
                    episodes_to_run.append((model_id, case, iteration_id, iter_seed))

        skipped = len(existing_traces)
        print(f"  Scheduled {len(episodes_to_run)} episodes (Skipped {skipped} already completed).")

        completed = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_ep = {
                executor.submit(run_episode, m_id, c, i_id, seed, config, benchmark_hash):
                    (m_id, c["case_id"])
                for (m_id, c, i_id, seed) in episodes_to_run
            }
            for future in as_completed(future_to_ep):
                rec = future.result()
                append_trace(rec)
                completed += 1
                if rec["status"] == "success":
                    print(f"  [{completed}/{len(episodes_to_run)}] Generated {rec['run_id']}")
                else:
                    print(f"  [{completed}/{len(episodes_to_run)}] FAILED {rec['run_id']}: {rec.get('failure_reason')}")

    # ── PHASE 2: EVALUATION ────────────────────────────────────────────────
    print(f"\n[PHASE 2] Independent Judge Evaluation...")
    all_traces = load_traces(config_fp, benchmark_hash)
    successful_traces = [v for v in all_traces.values() if v["status"] == "success"]
    failed_traces = [v for v in all_traces.values() if v["status"] == "failed"]

    if not successful_traces:
        print("  No successful traces to evaluate.")
    else:
        print(f"  Evaluating {len(successful_traces)} traces against {len(judge_ids)} judges.")

    judge_scores_by_trace: dict[str, list[dict]] = {
        t["run_id"]: [] for t in successful_traces
    }

    for j_id in judge_ids:
        print(f"  -> Loading Judge: {j_id}")
        try:
            j_runner = get_runner(j_id, verbose=False)
            judge = LLMJudge(judge_id=j_id, runner=j_runner)

            for i, trace_rec in enumerate(successful_traces, 1):
                run_id = trace_rec["run_id"]
                if i % 10 == 0:
                    print(f"     [Judge: {j_id}] Processed {i}/{len(successful_traces)}")
                j_result = judge.evaluate_interview(trace_rec["trace"])
                judge_scores_by_trace[run_id].append(j_result)

            del j_runner
            del judge
        except Exception as exc:
            print(f"  [ERROR] Judge {j_id} failed to load: {exc}")

    # ── PHASE 3: AGGREGATION & REPORTING ──────────────────────────────────
    print(f"\n[PHASE 3] Metric Computation and CSV Compilation...")

    csv_rows = []

    for trace_rec in successful_traces:
        run_id = trace_rec["run_id"]
        case_id = trace_rec["case_id"]
        ablation = trace_rec.get("ablation", "none")
        trace = trace_rec.get("trace", {})

        # Reference facts for this case
        ref_facts = facts_by_case.get(case_id, [])

        # Deterministic metrics
        det_metrics = compute_all_metrics(
            trace=trace,
            reference_facts=ref_facts if ref_facts else None,
            case_id=case_id,
            ablation=ablation,
        )

        # Judge panel aggregation
        raw_judge_results = judge_scores_by_trace.get(run_id, [])
        agg = aggregate_judge_results(raw_judge_results)

        row = {
            # Identity and provenance
            "run_id": run_id,
            "iteration_id": trace_rec["iteration_id"],
            "model_id": trace_rec["model_id"],
            "case_id": case_id,
            "case_group_id": trace_rec.get("case_group_id", ""),
            "question_variant_id": trace_rec.get("question_variant_id", ""),
            "variant_key": trace_rec.get("variant_key", ""),
            "case_category": trace_rec.get("case_category", ""),
            "ablation": ablation,
            "config_fingerprint": trace_rec.get("config_fingerprint", ""),
            "benchmark_hash": trace_rec.get("benchmark_hash", ""),
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
            "status": "success",
            "failure_reason": "",
            # Generation settings
            "effective_extraction_temperature": trace_rec.get("effective_extraction_temperature", ""),
            "effective_question_temperature": trace_rec.get("effective_question_temperature", ""),
            "effective_seed": trace_rec.get("effective_seed", ""),
            # Runtime
            "turn_count": trace.get("turn_count", 0),
            "total_runtime_ms": trace_rec.get("total_runtime_ms", 0),
            # Deterministic metrics (null for unavailable)
            **{k: v for k, v in det_metrics.items()},
            # Judge score
            "follow_up_question_quality": agg["panel_median"],
            "judge_disagreement_mad": agg["disagreement_mad"],
            "successful_judges": agg["successful_judges"],
            "total_judges": agg["total_judges"],
            "judge_schema_version": JUDGE_SCHEMA_VERSION,
        }
        csv_rows.append(row)

    # Include failed runs in CSV with null metric values
    for trace_rec in failed_traces:
        row = {
            "run_id": trace_rec["run_id"],
            "iteration_id": trace_rec["iteration_id"],
            "model_id": trace_rec["model_id"],
            "case_id": trace_rec["case_id"],
            "case_group_id": trace_rec.get("case_group_id", ""),
            "question_variant_id": trace_rec.get("question_variant_id", ""),
            "variant_key": trace_rec.get("variant_key", ""),
            "case_category": trace_rec.get("case_category", ""),
            "ablation": trace_rec.get("ablation", ""),
            "config_fingerprint": trace_rec.get("config_fingerprint", ""),
            "benchmark_hash": trace_rec.get("benchmark_hash", ""),
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
            "status": "failed",
            "failure_reason": trace_rec.get("failure_reason", ""),
        }
        # All metric columns are null for failed runs
        for col in CSV_COLS:
            if col not in row:
                row[col] = None
        csv_rows.append(row)

    with open(CSV_RESULTS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)

    n_success = sum(1 for r in csv_rows if r.get("status") == "success")
    n_failed = sum(1 for r in csv_rows if r.get("status") == "failed")
    print(f"  Results saved to {CSV_RESULTS}")
    print(f"  Successful: {n_success} | Failed: {n_failed} | Total: {len(csv_rows)}")
    print(f"  DONE.")


if __name__ == "__main__":
    main()
