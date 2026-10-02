"""
evaluate.py -- Main automated evaluation runner for the AI4CARE experiment.

Incorporates candidate episode generation and MoE multi-model judge pipeline.
Supports resuming from saved traces to avoid losing work.
"""

from __future__ import annotations

import argparse
import csv
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
from llm_judge import LLMJudge, METRICS as JUDGE_METRICS
from panel_aggregator import PanelAggregator

# ─── PATHS ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")
ITERATIONS_DIR = RESULTS_DIR / "iterations"
BENCHMARK_CASES_PATH = Path("data/benchmark/benchmark_cases.jsonl")
BENCHMARK_FACTS_PATH = Path("data/benchmark/benchmark_facts.jsonl")

TRACES_FILE = RESULTS_DIR / "evaluation_traces.jsonl"
CSV_RESULTS = RESULTS_DIR / "evaluation_results.csv"


# ─── EXPERIMENT CONFIG ────────────────────────────────────────────────────────
@dataclass
class ExperimentConfig:
    dataset_name: str = "ozguragrali/enterprise-knowledge-qa"
    dataset_version: str = "v1"
    model_ids: list[str] = field(default_factory=list)
    judge_ids: list[str] = field(default_factory=list)
    chroma_collection: str = CHROMA_COLLECTION
    temperature: float = 0.7
    max_tokens: int = 200
    n_ctx: int = 2048
    top_k: int = 5
    max_turns: int = 15
    completion_threshold: float = 0.75
    iterations: int = 3
    cases_per_iteration: int = 50
    sampling_mode: str = "proportional"
    random_seed: int = 1234
    ablation: str = "none"

    def to_dict(self) -> dict:
        return asdict(self)


# ─── BENCHMARK LOADING ────────────────────────────────────────────────────────

def load_benchmark() -> tuple[list[dict], dict[str, dict]]:
    if not BENCHMARK_CASES_PATH.exists():
        raise FileNotFoundError(f"Benchmark not found at {BENCHMARK_CASES_PATH}.")
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

    return cases, facts_by_case


def create_iteration_manifest(
    iteration_id: str,
    cases: list[dict],
    n_cases: int,
    seed: int,
    config: ExperimentConfig,
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
        "benchmark_version": "v1",
        "sampling_mode": config.sampling_mode,
        "n_cases": len(selected),
        "cases": [
            {
                "case_id": c["case_id"],
                "question_variant_id": c["question_variant_id"],
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
    out_path = ITERATIONS_DIR / f"iteration_{iteration_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# ─── EPISODE RUNNER (GENERATION PHASE) ──────────────────────────────────────────

def run_episode(
    model_id: str,
    case: dict,
    iteration_id: str,
    iteration_seed: int,
    config: ExperimentConfig,
) -> dict:
    run_id = f"{iteration_id}_{model_id}_{case['case_id'][:8]}"
    
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
            ablation=config.ablation
        )

        respondent = SyntheticRespondent(
            case_id=case["case_id"],
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
            participant_text = response["text"]
            
            state = session.respond(participant_text)

        total_runtime_ms = int((time.time() - t_start) * 1000)
        
        return {
            "run_id": run_id,
            "iteration_id": iteration_id,
            "model_id": model_id,
            "case_id": case["case_id"],
            "case_category": case.get("category", ""),
            "status": "success",
            "total_runtime_ms": total_runtime_ms,
            "trace": state,
        }

    except Exception as exc:
        return {
            "run_id": run_id,
            "iteration_id": iteration_id,
            "model_id": model_id,
            "case_id": case["case_id"],
            "status": "failed",
            "error": str(exc)
        }


# ─── JUDGING PHASE ───────────────────────────────────────────────────────────

def load_traces() -> dict:
    if not TRACES_FILE.exists():
        return {}
    traces = {}
    with open(TRACES_FILE, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            traces[rec["run_id"]] = rec
    return traces


def append_trace(record: dict) -> None:
    with open(TRACES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI4CARE Evaluation Runner")
    parser.add_argument("--models", default="all", help="Comma-separated model IDs")
    parser.add_argument("--judges", default="general_main", help="Comma-separated judge model IDs")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--cases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ablation", choices=["none", "no_rag", "state_machine"], default="none")
    parser.add_argument("--workers", type=int, default=1, help="Max multiprocessing workers")
    parser.add_argument("--skip-generation", action="store_true", help="Skip running models and only run judges on existing traces")
    args = parser.parse_args()

    model_ids = get_enabled_models() if args.models == "all" else [m.strip() for m in args.models.split(",")]
    judge_ids = [m.strip() for m in args.judges.split(",")]

    print(f"\n=== AI4CARE Pipeline ===")
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cases, _ = load_benchmark()
    
    # ── PHASE 1: GENERATION ──────────────────────────────────────────────
    if not args.skip_generation:
        print(f"\n[PHASE 1] Trace Generation...")
        existing_traces = load_traces()
        rng = random.Random(args.seed)
        
        episodes_to_run = []
        
        for iter_num in range(1, args.iterations + 1):
            iteration_id = f"{iter_num:04d}"
            iter_seed = rng.randint(0, 2**31)
            manifest = create_iteration_manifest(iteration_id, cases, args.cases, iter_seed, config)
            
            selected_vars = {c["question_variant_id"] for c in manifest["cases"]}
            ordered_cases = [c for c in cases if c["question_variant_id"] in selected_vars][:args.cases]
            
            for model_id in model_ids:
                for case in ordered_cases:
                    run_id = f"{iteration_id}_{model_id}_{case['case_id'][:8]}"
                    if run_id not in existing_traces or existing_traces[run_id]["status"] != "success":
                        episodes_to_run.append((model_id, case, iteration_id, iter_seed))
        
        print(f"  Scheduled {len(episodes_to_run)} episodes (Skipped {len(existing_traces)} already completed).")
        
        completed = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_ep = {
                executor.submit(run_episode, m_id, c, i_id, seed, config): (m_id, c["case_id"])
                for (m_id, c, i_id, seed) in episodes_to_run
            }
            for future in as_completed(future_to_ep):
                rec = future.result()
                append_trace(rec)
                completed += 1
                if rec["status"] == "success":
                    print(f"  [{completed}/{len(episodes_to_run)}] Generated {rec['run_id']}")
                else:
                    print(f"  [{completed}/{len(episodes_to_run)}] FAILED {rec['run_id']}: {rec.get('error')}")

    # ── PHASE 2: EVALUATION MoE ──────────────────────────────────────────
    print(f"\n[PHASE 2] Independent Judge Evaluation...")
    all_traces = load_traces()
    successful_traces = [v for k, v in all_traces.items() if v["status"] == "success"]
    
    if not successful_traces:
        print("  No successful traces to evaluate.")
        return

    # Sequentially load each judge to avoid OOM, evaluate ALL traces, then swap
    print(f"  Evaluating {len(successful_traces)} traces against {len(judge_ids)} judges.")
    
    judge_scores_by_trace = {t["run_id"]: [] for t in successful_traces}
    
    for j_id in judge_ids:
        print(f"  -> Loading Judge: {j_id}")
        j_runner = get_runner(j_id, verbose=False)
        judge = LLMJudge(judge_id=j_id, runner=j_runner)
        
        for i, trace_rec in enumerate(successful_traces, 1):
            run_id = trace_rec["run_id"]
            if i % 10 == 0:
                print(f"     [Judge: {j_id}] Processed {i}/{len(successful_traces)}")
            # Evaluate trace
            j_result = judge.evaluate_interview(trace_rec["trace"])
            judge_scores_by_trace[run_id].append(j_result)
        
        # Cleanup runner (LlamaCpp will unmap on destruction)
        del j_runner 
        del judge

    # ── AGGREGATION & REPORTING ──────────────────────────────────────────
    print(f"\n[PHASE 3] Panel Aggregation and CSV Compilation...")
    
    CSV_COLS = [
        "run_id", "iteration_id", "model_id", "case_id", "case_category",
        "turn_count", "schema_completeness", "generation_runtime_ms"
    ] + JUDGE_METRICS + [f"{m}_stdev" for m in JUDGE_METRICS] + ["successful_judges"]
    
    csv_rows = []
    
    for trace_rec in successful_traces:
        run_id = trace_rec["run_id"]
        # Fake a PanelAggregator for just the pre-computed results
        raw_results = judge_scores_by_trace[run_id]
        
        import statistics
        final_metrics = {}
        disagreements = {}
        successful_judges = 0
        
        for m in JUDGE_METRICS:
            valid_scores = [r["metrics"][m] for r in raw_results if r.get("success") and m in r.get("metrics", {})]
            if valid_scores:
                final_metrics[m] = statistics.median(valid_scores)
                disagreements[m] = statistics.stdev(valid_scores) if len(valid_scores) > 1 else 0.0
            else:
                final_metrics[m] = None
                disagreements[m] = 0.0
        
        successful_judges = sum(1 for r in raw_results if r.get("success"))
        
        row = {
            "run_id": run_id,
            "iteration_id": trace_rec["iteration_id"],
            "model_id": trace_rec["model_id"],
            "case_id": trace_rec["case_id"],
            "case_category": trace_rec.get("case_category", ""),
            "turn_count": trace_rec["trace"].get("turn_count", 0),
            "schema_completeness": trace_rec["trace"].get("completeness", 0),
            "generation_runtime_ms": trace_rec.get("total_runtime_ms", 0),
            "successful_judges": successful_judges
        }
        
        for m in JUDGE_METRICS:
            row[m] = final_metrics[m]
            row[f"{m}_stdev"] = disagreements[m]
            
        csv_rows.append(row)
        
    with open(CSV_RESULTS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLS)
        writer.writeheader()
        writer.writerows(csv_rows)
        
    print(f"  Results saved to {CSV_RESULTS}\n  DONE.")


if __name__ == "__main__":
    main()
