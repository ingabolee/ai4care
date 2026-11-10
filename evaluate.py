"""
evaluate.py -- Main automated evaluation runner for the AI4CARE experiment.

Evaluation schema version: v2

Key improvements over v1:
  - Collision-resistant run_id includes full case_id, variant_id, ablation, config fingerprint.
  - Pilot/thesis/full profiles isolate traces, metrics, manifests, and analysis inputs.
  - Config fingerprint (SHA-256) verifies compatibility before resuming cached runs.
  - Benchmark hash recorded for provenance.
  - One local runner is reused across each model's assigned episodes.
  - Worker checkpoints preserve completed episodes if a batch stops early.
  - Deterministic metrics are produced here; the separate judge stage runs later.

Generation seed note:
  The effective per-run seed is derived from (iteration_seed, model_id, case_id,
  variant_id) and passed through to extraction and question-generation calls.
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
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from datetime import datetime, timezone

from model_registry import (
    MODEL_CONFIGS, CHROMA_COLLECTION, select_interview_models, get_runner,
)
from engine import InterviewSession, schema_completeness
from synthetic_respondent import SyntheticRespondent
from llm_judge import JUDGE_SCHEMA_VERSION
from evaluation_metrics import compute_all_metrics, EVALUATION_SCHEMA_VERSION
from estimator import HardwareDetector, LLMCapacityEngine

# ─── PATHS ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")
ITERATIONS_DIR = RESULTS_DIR / "iterations"
TRACE_SHARDS_DIR = RESULTS_DIR / "trace_shards"
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
    max_turns: int = 4                 # participant responses per episode in the thesis profile
    completion_threshold: float = 0.75 # COMPLETION_THRESHOLD used by engine.py
    iterations: int = 1
    cases_per_iteration: int = 50
    sampling_mode: str = "proportional"
    random_seed: int = 1234
    ablation: str = "none"
    run_profile: str = "pilot"
    n_threads_per_worker: int = 1

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
            "n_ctx": self.n_ctx,
            "completion_threshold": self.completion_threshold,
            "top_k": self.top_k,
            "ablation": self.ablation,
            "sampling_mode": self.sampling_mode,
            "random_seed": self.random_seed,
            "iterations": self.iterations,
            "cases_per_iteration": self.cases_per_iteration,
            "run_profile": self.run_profile,
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

    if config.run_profile == "pilot":
        # A pilot covers distinct source groups before spending time on
        # additional variants from the same document.
        by_group: dict[str, list[dict]] = {}
        for case in eligible:
            group_id = str(case.get("case_group_id") or case.get("case_id"))
            by_group.setdefault(group_id, []).append(case)
        sample_size = min(n_cases, len(by_group))
        groups_by_category: dict[str, list[tuple[str, list[dict]]]] = {}
        for group_id, group_cases in by_group.items():
            category = group_cases[0].get("category", "other")
            groups_by_category.setdefault(category, []).append((group_id, group_cases))

        # Allocate the pilot proportionally by source-case category.
        # Largest-remainder allocation guarantees exactly sample_size selections.
        total_groups = sum(len(groups) for groups in groups_by_category.values())
        quotas = {
            category: sample_size * len(groups) / total_groups
            for category, groups in groups_by_category.items()
        }
        counts = {category: math.floor(quota) for category, quota in quotas.items()}
        remaining = sample_size - sum(counts.values())
        for category in sorted(quotas, key=lambda cat: (-(quotas[cat] - counts[cat]), cat))[:remaining]:
            counts[category] += 1

        selected = []
        for category, groups in groups_by_category.items():
            rng.shuffle(groups)
            for _, group_cases in groups[:counts[category]]:
                selected.append(rng.choice(group_cases))
    elif config.run_profile == "thesis":
        # Four question variants per source group (12 total by default). Pick
        # distinct variant keys first so the compact run covers question types
        # rather than repeatedly sampling near-duplicates from one source.
        by_group: dict[str, list[dict]] = {}
        for case in eligible:
            group_id = str(case.get("case_group_id") or case.get("case_id"))
            by_group.setdefault(group_id, []).append(case)
        group_ids = sorted(by_group)
        sample_size = min(n_cases, len(eligible))
        base, remainder = divmod(sample_size, max(len(group_ids), 1))
        allocation = {group_id: base for group_id in group_ids}
        remainder_order = list(group_ids)
        rng.shuffle(remainder_order)
        for group_id in remainder_order[:remainder]:
            allocation[group_id] += 1

        selected = []
        for group_id in group_ids:
            group_cases = by_group[group_id]
            by_variant: dict[str, list[dict]] = {}
            for case in group_cases:
                by_variant.setdefault(str(case.get("variant_key", "")), []).append(case)
            variant_keys = list(by_variant)
            rng.shuffle(variant_keys)
            quota = allocation[group_id]
            chosen = [rng.choice(by_variant[key]) for key in variant_keys[:quota]]
            if len(chosen) < quota:
                chosen_ids = {c["question_variant_id"] for c in chosen}
                remaining_cases = [c for c in group_cases if c["question_variant_id"] not in chosen_ids]
                rng.shuffle(remaining_cases)
                chosen.extend(remaining_cases[:quota - len(chosen)])
            selected.extend(chosen)
    elif config.run_profile == "full":
        # The proof-of-concept full run samples benchmark variants. The
        # analysis retains source groups as the independent resampling unit.
        by_category: dict[str, list[dict]] = {}
        for case in eligible:
            by_category.setdefault(case.get("category", "other"), []).append(case)
        sample_size = min(n_cases, len(eligible))
        quotas = {
            category: sample_size * len(pool) / len(eligible)
            for category, pool in by_category.items()
        }
        counts = {category: math.floor(quota) for category, quota in quotas.items()}
        remaining = sample_size - sum(counts.values())
        for category in sorted(quotas, key=lambda cat: (-(quotas[cat] - counts[cat]), cat))[:remaining]:
            counts[category] += 1
        selected = []
        for category, pool in by_category.items():
            rng.shuffle(pool)
            selected.extend(pool[:counts[category]])
    elif config.sampling_mode == "balanced":
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


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


# ─── EPISODE RUNNER ──────────────────────────────────────────────────────────

def run_episode(
    model_id: str,
    case: dict,
    iteration_id: str,
    iteration_seed: int,
    config: ExperimentConfig,
    benchmark_hash: str,
    runner=None,
    experiment_id: str = "",
    progress_callback=None,
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
        # The caller supplies one process-local runner for a whole model batch.
        # persist_runtime=False means this episode creates no Chroma records, so
        # deleting episode records here only added a database query per episode.
        if runner is None:
            runner = get_runner(model_id, verbose=False, n_ctx=config.n_ctx)
        rag_enabled = config.ablation != "no_rag"

        session = InterviewSession(
            episode_id=run_id,
            model_id=model_id,
            runner=runner,
            rag_enabled=rag_enabled,
            persist_runtime=False,
            ablation=config.ablation,
            seed=run_seed,
            max_turns=config.max_turns,
        )

        respondent = SyntheticRespondent(
            case_id=case_id,
            reference_answer=case["answer"],
            iteration_seed=iteration_seed,
        )

        initial_q = case.get("question", session.opening_question)

        t_start = time.time()
        if progress_callback:
            progress_callback(run_id, 0, config.max_turns)

        response = respondent.answer(initial_q, turn_id=0)
        state = session.respond(
            response["text"],
            respondent_evidence_indices=response.get("evidence_indices", []),
            respondent_relevance_scores=response.get("relevance_scores", []),
            respondent_not_specified=response.get("not_specified", False),
        )
        if progress_callback:
            progress_callback(run_id, state.get("turn_count", 0), config.max_turns)

        for turn_num in range(1, config.max_turns):
            if state.get("is_done"):
                break
            last_q = next(
                (t["text"] for t in reversed(state["turns"]) if t["role"] == "interviewer"),
                initial_q,
            )
            response = respondent.answer(last_q, turn_id=turn_num)
            state = session.respond(
                response["text"],
                respondent_evidence_indices=response.get("evidence_indices", []),
                respondent_relevance_scores=response.get("relevance_scores", []),
                respondent_not_specified=response.get("not_specified", False),
            )
            if progress_callback:
                progress_callback(run_id, state.get("turn_count", 0), config.max_turns)

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
            "run_profile": config.run_profile,
            "experiment_id": experiment_id,
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
            "run_profile": config.run_profile,
            "experiment_id": experiment_id,
            "config_fingerprint": config_fp,
            "benchmark_hash": benchmark_hash,
            "status": "failed",
            "failure_reason": str(exc),
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        }


def _run_model_batch(model_id: str, episodes: list[tuple], config: ExperimentConfig,
                     benchmark_hash: str, experiment_id: str, n_threads: int,
                     episode_concurrency: int = 1, shard_path: Optional[Path] = None,
                     n_gpu_layers: int = 0) -> dict:
    """Load one generator once, then reuse it across all assigned episodes."""
    shard_path = shard_path or (TRACE_SHARDS_DIR / f"{model_id}.jsonl")
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = shard_path.with_name(f"{model_id}.progress.json")
    write_lock = threading.Lock()
    records_written = 0
    successes = 0
    worker_started = time.perf_counter()
    progress = {
        "model_id": model_id,
        "status": "loading",
        "completed": 0,
        "total": len(episodes),
        "succeeded": 0,
        "failed": 0,
        "active_episodes": {},
        "model_load_ms": None,
        "updated_at": time.time(),
    }

    def write_progress():
        progress["elapsed_seconds"] = round(time.perf_counter() - worker_started, 1)
        progress["updated_at"] = time.time()
        temp_path = progress_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as progress_file:
            json.dump(progress, progress_file)
        os.replace(temp_path, progress_path)

    write_progress()

    def persist_record(shard_file, rec):
        nonlocal records_written, successes
        with write_lock:
            shard_file.write(json.dumps(rec) + "\n")
            shard_file.flush()
            progress["active_episodes"].pop(rec.get("run_id"), None)
            records_written += 1
            is_success = rec.get("status") == "success"
            successes += is_success
            progress["completed"] = records_written
            progress["succeeded"] = successes
            progress["failed"] = records_written - successes
            progress["status"] = "running"
            write_progress()

    load_started = time.perf_counter()
    try:
        runner = get_runner(
            model_id, verbose=False, n_ctx=config.n_ctx, n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
        )
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - load_started) * 1000)
        progress["model_load_ms"] = elapsed_ms
        progress["load_error"] = str(exc)
        with shard_path.open("a", encoding="utf-8") as shard_file:
            for case, iteration_id, iteration_seed in episodes:
                rec = run_episode(
                    model_id, case, iteration_id, iteration_seed, config,
                    benchmark_hash, runner=_UnavailableRunner(str(exc)),
                    experiment_id=experiment_id,
                )
                persist_record(shard_file, rec)
        progress["status"] = "complete"
        write_progress()
        return {"model_id": model_id, "records_written": records_written,
                "successes": successes, "model_load_ms": elapsed_ms}

    load_ms = int((time.perf_counter() - load_started) * 1000)
    progress["model_load_ms"] = load_ms
    progress["status"] = "running"
    write_progress()
    try:
        with shard_path.open("a", encoding="utf-8") as shard_file:
            def evaluate_episode(ep):
                case, iteration_id, iteration_seed = ep
                def report_turn(run_id, completed_turns, total_turns):
                    with write_lock:
                        progress["active_episodes"][run_id] = {
                            "turns_completed": completed_turns,
                            "turns_total": total_turns,
                        }
                        progress["status"] = "running"
                        write_progress()

                rec = run_episode(
                    model_id, case, iteration_id, iteration_seed, config,
                    benchmark_hash, runner=runner, experiment_id=experiment_id,
                    progress_callback=report_turn,
                )
                persist_record(shard_file, rec)
                return rec.get("status")

            if episode_concurrency > 1:
                with ThreadPoolExecutor(max_workers=episode_concurrency) as pool:
                    futures = [pool.submit(evaluate_episode, ep) for ep in episodes]
                    for future in as_completed(futures):
                        future.result()
            else:
                for ep in episodes:
                    evaluate_episode(ep)
    finally:
        del runner
    progress["status"] = "complete"
    write_progress()
    return {"model_id": model_id, "records_written": records_written,
            "successes": successes, "model_load_ms": load_ms}


class _UnavailableRunner:
    """Sentinel allowing a model-load error to be recorded for every episode."""
    def __init__(self, reason: str):
        self.reason = reason

    def generate(self, **kwargs):
        raise RuntimeError(self.reason)


# ─── TRACE LOADING ────────────────────────────────────────────────────────────

def load_traces(config_fingerprint: str, benchmark_hash: str) -> dict:
    """Load existing traces, filtering to those compatible with the current config.

    A trace is considered compatible if its config_fingerprint and benchmark_hash
    match the current run. Incompatible traces are preserved in the file but not
    reused.

    Duplicate run_ids with conflicting content are detected and reported.
    """
    trace_paths = [TRACES_FILE] if TRACES_FILE.exists() else []
    if TRACE_SHARDS_DIR.exists():
        trace_paths.extend(sorted(TRACE_SHARDS_DIR.glob("*.jsonl")))
    if not trace_paths:
        return {}

    traces: dict[str, dict] = {}
    conflicts: list[tuple[str, dict, dict]] = []

    for trace_path in trace_paths:
        with open(trace_path, encoding="utf-8") as f:
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
                    if (existing.get("status") != rec.get("status") or
                            existing.get("model_id") != rec.get("model_id")):
                        conflicts.append((run_id, existing, rec))
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


def consolidate_trace_shards() -> None:
    """Append worker checkpoints to the public JSONL and clear merged shards."""
    if not TRACE_SHARDS_DIR.exists():
        return
    shard_paths = sorted(TRACE_SHARDS_DIR.glob("*.jsonl"))
    if not shard_paths:
        return
    with open(TRACES_FILE, "a", encoding="utf-8") as out:
        for shard_path in shard_paths:
            with open(shard_path, encoding="utf-8") as shard:
                for line in shard:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
        out.flush()
    for shard_path in shard_paths:
        shard_path.unlink()


def append_trace(record: dict) -> None:
    with open(TRACES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ─── CSV COLUMNS ─────────────────────────────────────────────────────────────

CSV_COLS = [
    # Identity and provenance
    "run_id", "experiment_id", "run_profile", "iteration_id", "model_id", "case_id", "case_group_id",
    "question_variant_id", "variant_key", "case_category", "ablation",
    "config_fingerprint", "benchmark_hash", "evaluation_schema_version",
    "status", "failure_reason",
    # Generation settings
    "effective_extraction_temperature", "effective_question_temperature",
    "effective_seed",
    # Runtime
    "turn_count", "total_runtime_ms", "llm_calls", "prompt_tokens", "completion_tokens",
    # Primary outcome
    "reference_fact_f1", "reference_fact_precision", "reference_fact_recall",
    "matching_algorithm", "matching_threshold",
    "extracted_claim_count", "reference_fact_count", "source_reference_fact_count",
    "exposed_reference_fact_count", "reference_fact_exposure_rate", "matched_pair_count",
    "reference_metrics_status", "reference_metrics_scope",
    # Secondary structural metrics
    "schema_completeness", "novel_fields_total", "novel_fields_per_response",
    # Process diagnostics
    "question_redundancy_rate", "redundant_question_count", "eligible_question_count",
    "gap_targeting_rate", "gap_aligned_count", "gap_eligible_count",
    # Knowledge state diagnostics
    "conflicting_field_rate", "conflicting_field_count",
    "evidence_presence_rate", "fields_with_evidence_count",
    "evidence_supported_update_count", "evidence_checked_update_count",
    "evidence_text_support_rate", "evidence_support_threshold",
    "llm_extraction_count", "fallback_extraction_count", "empty_extraction_count",
    "extraction_llm_calls", "extraction_diag_status",
    # Judge score
    "follow_up_question_quality",
    "judge_disagreement_mad",
    "successful_judges", "total_judges",
    "judge_schema_version",
]


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    global RESULTS_DIR, ITERATIONS_DIR, TRACE_SHARDS_DIR, TRACES_FILE, CSV_RESULTS, FAILURES_FILE
    parser = argparse.ArgumentParser(description="AI4CARE benchmark generation and deterministic evaluation")
    parser.add_argument("--profile", choices=["thesis", "pilot", "full", "custom"], default="thesis",
                        help="Run label. Thesis defaults to 12 balanced variants; full evaluates all available variants.")
    parser.add_argument("--models", default="all", help="Comma-separated generator model IDs")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--cases", type=int, default=None,
                        help="Cases per iteration; thesis defaults to 12 balanced cases (4 per source group)")
    parser.add_argument("--max-turns", type=int, default=4,
                        help="Maximum participant replies per interview (default: 4; hard maximum: 15).")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ablation", choices=["none", "no_rag", "state_machine"], default="none")
    parser.add_argument("--workers", type=int, default=0,
                        help="Concurrent model batches; 0 chooses a hardware-aware default.")
    parser.add_argument("--episode-concurrency", type=int, default=0,
                        help="Concurrent episodes per model for OpenAI-compatible backends; 0 auto-selects.")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Existing experiment folder to resume; otherwise a new named folder is created.")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Reuse traces in --run-dir and only rebuild deterministic metrics.")
    args = parser.parse_args()
    if args.skip_generation and not args.run_dir:
        parser.error("--skip-generation requires --run-dir so it can find the existing experiment traces.")

    hardware = HardwareDetector.get_system_hardware()
    if args.models == "all":
        model_ids, model_selection = select_interview_models(
            hardware=hardware, count=8, quant="Q4_K_M", context_length=2048,
        )
    else:
        model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
        model_selection = [
            {"model_id": model_id, "selection_source": "manual_override"}
            for model_id in model_ids
        ]
    cases, facts_by_case, benchmark_hash = load_benchmark()
    eligible_case_count = sum(bool(case.get("answer", "").strip()) for case in cases)
    source_group_count = len({
        str(case.get("case_group_id") or case.get("case_id")) for case in cases
    })
    iterations = args.iterations if args.iterations is not None else 1
    cases_per_iteration = args.cases if args.cases is not None else (
        eligible_case_count if args.profile == "full" else 12 if args.profile == "thesis"
        else 3 if args.profile == "pilot" else 10
    )
    if not model_ids:
        parser.error("At least one generator model must be selected.")
    if iterations < 1 or cases_per_iteration < 1:
        parser.error("--iterations and --cases must be positive integers.")
    if not 1 <= args.max_turns <= 15:
        parser.error("--max-turns must be between 1 and 15.")
    if args.profile == "thesis" and cases_per_iteration < source_group_count:
        parser.error(
            f"The thesis profile needs at least one case per independent source group ({source_group_count})."
        )
    if args.profile == "thesis" and iterations != 1:
        parser.error("The balanced thesis profile uses one pass through its selected benchmark variants.")
    if args.profile == "full" and iterations != 1:
        parser.error(
            "The full profile evaluates the finite benchmark once; repeated draws do not add "
            "new benchmark variants. Use --profile custom for exploratory repetitions."
        )
    if args.profile == "full" and cases_per_iteration != eligible_case_count:
        parser.error(
            f"The full profile evaluates all {eligible_case_count} available benchmark variants. "
            "Use --profile pilot or custom for a smaller exploratory run."
        )

    backend = os.environ.get("AI4CARE_BACKEND", "local").lower()
    if args.workers < 0:
        parser.error("--workers must be 0 (automatic) or a positive integer.")
    if args.episode_concurrency < 0:
        parser.error("--episode-concurrency must be 0 (automatic) or a positive integer.")
    cpu_cores = max(1, int(hardware.get("cpu_cores") or os.cpu_count() or 1))
    if args.workers:
        workers = min(args.workers, len(model_ids))
    elif backend == "openai":
        workers = min(8, len(model_ids), cpu_cores)
    elif hardware.get("gpu_count", 0):
        workers = 1  # avoid placing several full models on the same accelerator
    else:
        # On CPU hosts, fit the largest concurrently loaded selected models
        # inside the estimator's RAM budget, then use available cores across
        # those workers. This scales with cloud hardware instead of capping all
        # machines at four processes while avoiding an unsafe RAM-only guess.
        memory_estimates = sorted(
            (float(item["estimated_memory_gb"]) for item in model_selection
             if item.get("estimated_memory_gb") is not None),
            reverse=True,
        )
        memory_budget = float(hardware.get("ram_gb") or 0) * (
            1 - LLMCapacityEngine.MEMORY_RESERVE
        )
        memory_workers = 0
        memory_used = 0.0
        for estimate in memory_estimates:
            if memory_used + estimate > memory_budget:
                break
            memory_used += estimate
            memory_workers += 1
        if not memory_estimates:
            memory_workers = min(4, len(model_ids))
        workers = min(
            len(model_ids), max(1, cpu_cores // 4), max(1, memory_workers)
        )
    if backend != "openai" and hardware.get("gpu_count", 0) and args.workers > 1:
        print("  Note: multiple local model workers will share the detected accelerator memory.")
    if args.episode_concurrency:
        episode_concurrency = args.episode_concurrency if backend == "openai" else 1
        if backend != "openai":
            print("  Note: episode-level concurrency is limited to 1 for local llama.cpp runners.")
    else:
        episode_concurrency = 4 if backend == "openai" else 1
    gpu_layers_override = os.environ.get("AI4CARE_GPU_LAYERS")
    n_gpu_layers = int(gpu_layers_override) if gpu_layers_override is not None else (
        -1 if hardware.get("gpu_count", 0) and backend != "openai" else 0
    )
    threads_per_worker = max(1, cpu_cores // max(workers, 1))

    config = ExperimentConfig(
        model_ids=model_ids,
        iterations=iterations,
        cases_per_iteration=cases_per_iteration,
        max_turns=args.max_turns,
        sampling_mode="balanced_source_and_variant" if args.profile == "thesis" else "proportional",
        random_seed=args.seed,
        ablation=args.ablation,
        run_profile=args.profile,
        n_threads_per_worker=threads_per_worker,
    )
    config_fp = config.fingerprint()
    if args.run_dir:
        run_dir = Path(args.run_dir)
        experiment_id = run_dir.name
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        experiment_id = f"{args.profile}_{args.ablation}_{len(model_ids)}models_{iterations}i_{cases_per_iteration}c_{args.max_turns}t_{stamp}"
        run_dir = RESULTS_DIR / "experiments" / args.profile / experiment_id

    RESULTS_DIR = run_dir
    ITERATIONS_DIR = run_dir / "iterations"
    TRACE_SHARDS_DIR = run_dir / "trace_shards"
    TRACES_FILE = run_dir / "evaluation_traces.jsonl"
    CSV_RESULTS = run_dir / "evaluation_results.csv"
    FAILURES_FILE = run_dir / "evaluation_failures.jsonl"
    if args.profile == "thesis":
        print(
            f"  THESIS COMPACT RUN: {cases_per_iteration} balanced benchmark variants across "
            f"{source_group_count} source groups; up to {args.max_turns} participant replies per episode.",
            flush=True,
        )
    elif args.profile == "pilot":
        print(
            f"  PILOT: {source_group_count} independent source cases; this is a "
            "pipeline check, not thesis evidence.",
            flush=True,
        )
    elif args.profile == "custom":
        print(
            f"  EXPLORATORY: custom draw of {cases_per_iteration} benchmark variants from "
            f"{source_group_count} independent source cases; use the full profile to evaluate "
            "all benchmark variants.",
            flush=True,
        )
    else:
        print(
            f"  FULL PROOF OF CONCEPT: evaluating all {eligible_case_count} benchmark variants "
            f"from {source_group_count} source cases. "
            "Interpret findings within this finite corpus.",
            flush=True,
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ITERATIONS_DIR.mkdir(parents=True, exist_ok=True)
    config_fp = config.fingerprint()
    run_manifest_path = run_dir / "run_manifest.json"
    if run_manifest_path.exists():
        existing_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if (existing_manifest.get("config_fingerprint") != config_fp or
                existing_manifest.get("benchmark_hash") != benchmark_hash):
            parser.error("--run-dir contains a different configuration. Use its original --profile, --models, --iterations, --cases, --seed, and --ablation values.")
    else:
        run_manifest = {
            "experiment_id": experiment_id,
            "profile": args.profile,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_fingerprint": config_fp,
            "config": config.to_dict(),
            "benchmark_hash": benchmark_hash,
            "benchmark_cases_available": len(cases),
            "independent_source_cases": source_group_count,
            "sampling_unit": "balanced_benchmark_variants_clustered_by_source_group" if args.profile == "thesis" else "independent_source_case_group" if args.profile == "pilot" else "benchmark_variant",
            "hardware": hardware,
            "model_selection": {
                "method": "estimator hardware fit at Q4_K_M / 2048 context" if args.models == "all" else "manual override",
                "requested_count": len(model_ids),
                "selected_models": model_selection,
            },
            "backend": backend,
            "workers": workers,
            "threads_per_worker": threads_per_worker,
            "gpu_layers": n_gpu_layers,
            "paths": {
                "traces": str(TRACES_FILE),
                "results_csv": str(CSV_RESULTS),
                "judge_output": str(run_dir / "judges"),
            },
            "stage_status": {"generation": "pending", "judging": "pending", "analysis": "pending"},
        }
        run_manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    print(f"\n=== AI4CARE {args.profile.upper()} RUN ===")
    print(f"  Experiment : {experiment_id}")
    print(f"  Models     : {model_ids}")
    if args.models == "all":
        print("  Estimator  : " + "; ".join(
            f"{m['model_id']} ({m['params_b']:.2f}B, {m['estimated_memory_gb']:.2f} GB est.)"
            for m in model_selection
        ))
    print(f"  Iterations : {iterations} | Cases per iteration: {cases_per_iteration} | Max replies: {args.max_turns}")
    print(f"  Ablation   : {args.ablation} | model workers: {workers} | episode concurrency: {episode_concurrency} | threads/worker: {threads_per_worker} | GPU layers: {n_gpu_layers}")
    print(f"  Hardware   : {hardware.get('cpu_cores')} CPU cores | {hardware.get('gpu_name')} ({hardware.get('vram_gb')} GB)")
    print(f"  Benchmark  : {benchmark_hash}")
    print(f"  Data       : {len(cases)} benchmark variants | {source_group_count} independent source cases")
    print(f"  Output     : {run_dir.resolve()}")

    performance_path = run_dir / "performance_profile.json"
    try:
        previous_performance = json.loads(performance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        previous_performance = {}
    previous_load_times = previous_performance.get("model_load_ms", {})
    generation_started = time.perf_counter()
    model_load_times: dict[str, int] = {}
    generated_count = 0
    if not args.skip_generation:
        print("\n[PHASE 1] Generating interview traces...")
        existing_traces = load_traces(config_fp, benchmark_hash)
        rng = random.Random(args.seed)
        episodes_by_model: dict[str, list[tuple]] = {m: [] for m in model_ids}
        expected_episode_count = 0

        case_lookup = {(c["case_id"], c["question_variant_id"]): c for c in cases}
        for iter_num in range(1, iterations + 1):
            iteration_id = f"{iter_num:04d}"
            iter_seed = rng.randint(0, 2**31)
            iteration_manifest = create_iteration_manifest(
                iteration_id, cases, cases_per_iteration, iter_seed, config, benchmark_hash
            )
            selected = [case_lookup[(c["case_id"], c["question_variant_id"])]
                        for c in iteration_manifest["cases"]]
            expected_episode_count += len(selected) * len(model_ids)
            for model_id in model_ids:
                for case in selected:
                    run_id = make_run_id(
                        iteration_id, model_id, case["case_id"],
                        case["question_variant_id"], config.ablation, config_fp,
                    )
                    if existing_traces.get(run_id, {}).get("status") == "success":
                        continue
                    episodes_by_model[model_id].append((case, iteration_id, iter_seed))

        scheduled = sum(len(eps) for eps in episodes_by_model.values())
        print(
            f"  Scheduled {scheduled} episodes; reused {len(existing_traces)} compatible traces.",
            flush=True,
        )
        active_batches = [(m, eps) for m, eps in episodes_by_model.items() if eps]
        if active_batches:
            TRACE_SHARDS_DIR.mkdir(parents=True, exist_ok=True)
            progress_paths = {
                model_id: TRACE_SHARDS_DIR / f"{model_id}.progress.json"
                for model_id, _ in active_batches
            }
            progress_started = time.perf_counter()
            last_seen: dict[str, tuple] = {}
            last_heartbeat = progress_started
            with ProcessPoolExecutor(max_workers=min(workers, len(active_batches))) as executor:
                futures = {
                    executor.submit(_run_model_batch, m, eps, config, benchmark_hash,
                                    experiment_id, threads_per_worker, episode_concurrency,
                                    TRACE_SHARDS_DIR / f"{m}.jsonl", n_gpu_layers): m
                    for m, eps in active_batches
                }
                print(
                    f"  Live progress: interview turns plus a checkpoint per completed episode; "
                    f"{len(active_batches)} model batch(es) started.",
                    flush=True,
                )
                pending_futures = set(futures)
                while pending_futures:
                    done_futures, pending_futures = wait(
                        pending_futures, timeout=2, return_when=FIRST_COMPLETED
                    )

                    now = time.perf_counter()
                    completed_live = 0
                    states: dict[str, dict] = {}
                    for model_id, progress_path in progress_paths.items():
                        try:
                            state = json.loads(progress_path.read_text(encoding="utf-8"))
                        except (OSError, json.JSONDecodeError):
                            state = {
                                "model_id": model_id,
                                "status": "starting",
                                "completed": 0,
                                "total": len(episodes_by_model[model_id]),
                                "succeeded": 0,
                                "failed": 0,
                            }
                        states[model_id] = state
                        completed_live += int(state.get("completed", 0))

                    elapsed = max(now - progress_started, 0.1)
                    rate = completed_live / elapsed
                    eta_text = (
                        _format_duration((scheduled - completed_live) / rate)
                        if rate > 0 else "estimating"
                    )
                    progress_changed = False
                    for model_id, state in states.items():
                        active = state.get("active_episodes", {})
                        current = next(iter(active.values()), {})
                        turn_marker = (
                            current.get("turns_completed"), current.get("turns_total")
                        )
                        marker = (
                            state.get("status"), state.get("completed"),
                            state.get("total"), turn_marker,
                        )
                        if marker != last_seen.get(model_id):
                            last_seen[model_id] = marker
                            progress_changed = True
                            turn_progress = (
                                f"; current turn {current['turns_completed']}/"
                                f"{current['turns_total']}"
                                if current else ""
                            )
                            print(
                                f"  [{completed_live}/{scheduled}] {model_id}: "
                                f"{state.get('status')} "
                                f"{state.get('completed', 0)}/{state.get('total', 0)} episodes "
                                f"(ok={state.get('succeeded', 0)}, failed={state.get('failed', 0)})"
                                f"{turn_progress}; "
                                f"elapsed {_format_duration(elapsed)}, ETA {eta_text}",
                                flush=True,
                            )
                    if progress_changed:
                        last_heartbeat = now
                    elif now - last_heartbeat >= 60:
                        loading = [
                            f"{model_id}={state.get('status')}"
                            for model_id, state in states.items()
                            if state.get("completed", 0) < state.get("total", 0)
                        ]
                        print(
                            f"  [{completed_live}/{scheduled}] Still running: "
                            f"{', '.join(loading) or 'finalizing'}; "
                            f"elapsed {_format_duration(elapsed)}, ETA {eta_text}",
                            flush=True,
                        )
                        last_heartbeat = now

                    for future in done_futures:
                        try:
                            batch = future.result()
                        except Exception as exc:
                            print(f"  Model batch {futures[future]} stopped early: {exc}. "
                                  "Completed episode checkpoints will be kept for resuming.",
                                  flush=True)
                            continue
                        model_load_times[batch["model_id"]] = batch["model_load_ms"]
                        generated_count += batch["records_written"]
                        print(f"  Finished batch for {batch['model_id']} "
                              f"({batch['records_written']} episodes; {batch['successes']} succeeded; "
                              f"model load {batch['model_load_ms'] / 1000:.1f}s).", flush=True)

    generation_wall_seconds = time.perf_counter() - generation_started
    consolidate_trace_shards()
    cumulative_generation_seconds = previous_performance.get("generation_wall_seconds", 0.0)
    if generated_count:
        cumulative_generation_seconds += generation_wall_seconds
    cumulative_load_times = dict(previous_load_times)
    for model_id, load_ms in model_load_times.items():
        cumulative_load_times[model_id] = cumulative_load_times.get(model_id, 0) + load_ms

    # ── PHASE 2: DETERMINISTIC METRICS ───────────────────────────────────
    print("\n[PHASE 2] Computing deterministic metrics...")
    all_traces = load_traces(config_fp, benchmark_hash)
    successful_traces = [v for v in all_traces.values() if v.get("status") == "success"]
    failed_traces = [v for v in all_traces.values() if v.get("status") == "failed"]
    print(f"  {len(successful_traces)} successful traces; {len(failed_traces)} failed traces.")

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

        row = {
            # Identity and provenance
            "run_id": run_id,
            "experiment_id": trace_rec.get("experiment_id", experiment_id),
            "run_profile": trace_rec.get("run_profile", args.profile),
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
            "llm_calls": sum(
                (turn.get("extraction_llm_calls", 0) if turn.get("role") == "participant" else 0) +
                (1 if turn.get("generation_latency_ms") else 0)
                for turn in trace.get("turns", [])
            ),
            "prompt_tokens": sum(
                (turn.get("extraction_prompt_tokens", 0) if turn.get("role") == "participant" else 0) +
                turn.get("generation_prompt_tokens", 0)
                for turn in trace.get("turns", [])
            ),
            "completion_tokens": sum(
                (turn.get("extraction_completion_tokens", 0) if turn.get("role") == "participant" else 0) +
                turn.get("generation_completion_tokens", 0)
                for turn in trace.get("turns", [])
            ),
            # Deterministic metrics (null for unavailable)
            **{k: v for k, v in det_metrics.items()},
            # Judge score
            "follow_up_question_quality": None,
            "judge_disagreement_mad": None,
            "successful_judges": None,
            "total_judges": None,
            "judge_schema_version": JUDGE_SCHEMA_VERSION,
        }
        csv_rows.append(row)

    # Include failed runs in CSV with null metric values
    for trace_rec in failed_traces:
        row = {
            "run_id": trace_rec["run_id"],
            "experiment_id": trace_rec.get("experiment_id", experiment_id),
            "run_profile": trace_rec.get("run_profile", args.profile),
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
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if not args.skip_generation:
        if len(all_traces) < expected_episode_count:
            generation_status = "incomplete"
        elif failed_traces:
            generation_status = "complete_with_failures"
        else:
            generation_status = "complete"
        run_manifest["expected_episode_count"] = expected_episode_count
    else:
        generation_status = run_manifest.get("stage_status", {}).get("generation", "reused")
    run_manifest["stage_status"]["generation"] = generation_status
    judge_scores_path = run_dir / "judges" / "judge_scores_long.csv"
    run_manifest["stage_status"]["judging"] = (
        "pending" if generated_count or not judge_scores_path.exists() else "complete"
    )
    run_manifest["stage_status"]["analysis"] = "pending"
    run_manifest["paths"]["results_csv"] = str(CSV_RESULTS)
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    trace_model_latency = {}
    for rec in successful_traces:
        model_stats = trace_model_latency.setdefault(rec.get("model_id", ""), {
            "episodes": 0, "episode_runtime_ms": 0, "llm_calls": 0,
            "llm_latency_ms": 0, "extraction_latency_ms": 0,
            "question_latency_ms": 0, "retrieval_latency_ms": 0,
            "prompt_tokens": 0, "completion_tokens": 0,
        })
        model_stats["episodes"] += 1
        model_stats["episode_runtime_ms"] += rec.get("total_runtime_ms", 0)
        for turn in rec.get("trace", {}).get("turns", []):
            if turn.get("role") == "participant":
                extraction_latency = turn.get("extraction_latency_ms", 0)
                model_stats["extraction_latency_ms"] += extraction_latency
                model_stats["llm_calls"] += turn.get("extraction_llm_calls", 0)
                model_stats["llm_latency_ms"] += extraction_latency
                model_stats["retrieval_latency_ms"] += turn.get("retrieval_latency_ms", 0)
                model_stats["prompt_tokens"] += turn.get("extraction_prompt_tokens", 0)
                model_stats["completion_tokens"] += turn.get("extraction_completion_tokens", 0)
            elif turn.get("role") == "interviewer":
                latency = turn.get("generation_latency_ms", 0)
                if latency:
                    model_stats["question_latency_ms"] += latency
                    model_stats["llm_latency_ms"] += latency
                    model_stats["llm_calls"] += 1
                model_stats["prompt_tokens"] += turn.get("generation_prompt_tokens", 0)
                model_stats["completion_tokens"] += turn.get("generation_completion_tokens", 0)

    performance = {
        "experiment_id": experiment_id,
        "profile": args.profile,
        "generation_wall_seconds": round(cumulative_generation_seconds, 3),
        "last_invocation_wall_seconds": round(generation_wall_seconds, 3),
        "episodes_generated_this_invocation": generated_count,
        "successful_episode_traces": len(successful_traces),
        "model_load_ms": cumulative_load_times,
        "model_batch_workers": workers,
        "episode_concurrency_per_model": episode_concurrency,
        "per_model": trace_model_latency,
        "notes": [
            "Episode latency excludes model initialization; initialization time is reported separately.",
            "Judge inference runs as a separate resumable stage using judge_benchmark.py.",
        ],
    }
    (run_dir / "performance_profile.json").write_text(json.dumps(performance, indent=2), encoding="utf-8")

    print(f"  Results saved to {CSV_RESULTS}")
    print(f"  Successful: {n_success} | Failed: {n_failed} | Total: {len(csv_rows)}")
    print(f"  Performance profile: {run_dir / 'performance_profile.json'}")
    print(
        f"  Next: python judge_benchmark.py --experiment-dir \"{run_dir}\" "
        "--top-n 4 --small-judges 3 --large-judges 1 --sample-per-model-source 1"
    )
    print("  DONE.")


if __name__ == "__main__":
    main()
