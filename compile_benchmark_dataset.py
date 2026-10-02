"""
compile_benchmark_dataset.py -- Build the frozen benchmark Q/A dataset.

Pipeline:
    1. Load the curated JSONL corpus (grouped by source case).
    2. Reconstruct case-level units (NOT individual chunks).
    3. Generate multiple question variants per case.
    4. Extract reference facts for each case.
    5. Save to:
         data/benchmark/benchmark_cases.jsonl
         data/benchmark/benchmark_facts.jsonl

The benchmark is IMMUTABLE after compilation. All models use the exact same
reference answers and facts. Do not regenerate during evaluation.

Usage:
    python compile_benchmark_dataset.py [--limit N] [--seed 42]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Optional

BENCHMARK_CASES_PATH = Path("data/benchmark/benchmark_cases.jsonl")
BENCHMARK_FACTS_PATH = Path("data/benchmark/benchmark_facts.jsonl")
CURATED_JSONL = Path("data/corpus/curated_corpus.jsonl")

# Question variant templates for each field type
QUESTION_TEMPLATES: dict[str, list[str]] = {
    "overall": [
        "Can you summarise what this case is about and the main challenge described?",
        "What is the primary topic and key problem described in this case?",
    ],
    "challenge": [
        "What was the main challenge or problem described in this case?",
        "What difficulties did the organisation face according to this case?",
    ],
    "root_cause": [
        "What caused the main problem described in this case?",
        "What was the root cause of the issue described here?",
        "Why did this problem occur according to the case?",
    ],
    "decision": [
        "What decisions were made to address the problem described?",
        "What actions or decisions were taken in response to the challenge?",
    ],
    "technology": [
        "What technologies or systems are mentioned in this case?",
        "Which tools, platforms, or technologies played a role in this case?",
    ],
    "people": [
        "Who were the key people or roles involved in this case?",
        "What teams or individuals are mentioned in this case?",
    ],
    "process": [
        "What processes or workflows are described in this case?",
        "Which operational procedures are discussed in this case?",
    ],
    "impact": [
        "What was the impact or outcome of the events described in this case?",
        "What were the consequences or results of the actions taken?",
    ],
    "lessons_learned": [
        "What lessons or takeaways does this case provide?",
        "What could be learned or improved based on this case?",
    ],
}


def stable_case_id(source_unique_id: str) -> str:
    raw = f"case_{source_unique_id}"
    return "case_" + hashlib.md5(raw.encode()).hexdigest()[:10]


def stable_variant_id(case_id: str, variant_key: str, variant_index: int) -> str:
    raw = f"{case_id}_{variant_key}_{variant_index}"
    return hashlib.md5(raw.encode()).hexdigest()[:8]


def extract_reference_facts(text: str, case_id: str) -> list[dict]:
    """Extract simple reference facts from a case text.

    Facts are extracted as atomic sentences covering different schema fields.
    This is deterministic and does NOT use an LLM (to remain stable).
    """
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 20]
    facts = []
    for i, sent in enumerate(sentences[:12]):  # up to 12 reference facts per case
        fact_id = f"{case_id}_fact{i:02d}"
        facts.append({
            "fact_id": fact_id,
            "case_id": case_id,
            "text": sent + ".",
            "source_index": i,
        })
    return facts


def load_curated_corpus() -> dict[str, list[dict]]:
    """Load and group curated records by source_unique_id."""
    if not CURATED_JSONL.exists():
        raise FileNotFoundError(
            f"Curated corpus not found at {CURATED_JSONL}.\n"
            "Run: python curate_seed_corpus.py"
        )
    groups: dict[str, list[dict]] = {}
    with open(CURATED_JSONL, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            uid = rec["source_unique_id"]
            groups.setdefault(uid, []).append(rec)
    return groups


def compile_benchmark(
    limit: Optional[int] = None,
    seed: int = 42,
) -> dict:
    rng = random.Random(seed)

    groups = load_curated_corpus()
    uids = sorted(groups.keys())
    if limit:
        uids = uids[:limit]

    print(f"Compiling benchmark from {len(uids)} logical cases...")

    all_cases: list[dict] = []
    all_facts: list[dict] = []

    for uid in uids:
        chunks = sorted(groups[uid], key=lambda r: r.get("chunk_index", 0))
        full_text = " ".join(r["text"] for r in chunks)
        case_id = stable_case_id(uid)
        category = chunks[0].get("category", "other")
        subcategory = chunks[0].get("classification_method", "keyword")

        # Reference answer = full reconstructed document text
        reference_answer = full_text

        # Reference facts
        ref_facts = extract_reference_facts(full_text, case_id)
        all_facts.extend(ref_facts)

        # Generate question variants
        for variant_key, templates in QUESTION_TEMPLATES.items():
            for vi, template in enumerate(templates):
                variant_id = stable_variant_id(case_id, variant_key, vi)
                all_cases.append({
                    "case_id": case_id,
                    "case_group_id": uid,
                    "question_variant_id": variant_id,
                    "question": template,
                    "answer": reference_answer,
                    "reference_fact_ids": [f["fact_id"] for f in ref_facts],
                    "category": category,
                    "subcategory": subcategory,
                    "source_id": uid,
                    "variant_key": variant_key,
                })

    print(f"  Generated {len(all_cases)} benchmark cases ({len(uids)} source cases).")
    print(f"  Generated {len(all_facts)} reference facts.")

    # ── Save ─────────────────────────────────────────────────────────────────
    BENCHMARK_CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_CASES_PATH, "w", encoding="utf-8") as f:
        for rec in all_cases:
            f.write(json.dumps(rec) + "\n")
    print(f"  Benchmark cases → {BENCHMARK_CASES_PATH}")

    with open(BENCHMARK_FACTS_PATH, "w", encoding="utf-8") as f:
        for rec in all_facts:
            f.write(json.dumps(rec) + "\n")
    print(f"  Benchmark facts → {BENCHMARK_FACTS_PATH}")

    # Category distribution
    cat_dist: dict[str, int] = {}
    seen_cases: set[str] = set()
    for rec in all_cases:
        if rec["case_id"] not in seen_cases:
            cat_dist[rec["category"]] = cat_dist.get(rec["category"], 0) + 1
            seen_cases.add(rec["case_id"])

    return {
        "source_cases": len(uids),
        "benchmark_cases": len(all_cases),
        "reference_facts": len(all_facts),
        "category_distribution": cat_dist,
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description="Compile benchmark Q/A dataset")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stats = compile_benchmark(limit=args.limit, seed=args.seed)
    print("\n── Benchmark Summary ──────────────────────────────────────────")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
