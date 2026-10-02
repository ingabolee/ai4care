"""
curate_seed_corpus.py -- Curate, chunk, classify and ingest the background
                          knowledge corpus into the single Chroma collection.

Pipeline:
    1. Load source dataset from HuggingFace.
    2. Group chunks by source/case to reconstruct logical documents.
    3. Remove duplicates (deterministic ID-based).
    4. Remove metadata-only fragments (too short).
    5. Classify each document into canonical categories.
    6. Chunk long documents into ~600 token segments with ~75 token overlap.
    7. Attach metadata (category, source, classification_method, etc.).
    8. Batch-upsert into ai4care_knowledge with namespace="corpus".
    9. Save JSONL manifest.

Usage:
    python curate_seed_corpus.py [--limit N] [--chunk-size 600] [--overlap 75]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Optional

import chromadb

from model_registry import CHROMA_DIR, CHROMA_COLLECTION

# ─── CONSTANTS ───────────────────────────────────────────────────────────────
MANIFEST_PATH = Path("data/manifests/seed_corpus_manifest.json")
CURATED_JSONL = Path("data/corpus/curated_corpus.jsonl")

# Approx tokens (words / 0.75) – simple word-based estimator
DEFAULT_CHUNK_SIZE = 600   # tokens
DEFAULT_OVERLAP = 75       # tokens
MIN_TEXT_TOKENS = 30       # minimum length to keep a fragment

CANONICAL_CATEGORIES = [
    "technology", "finance", "healthcare", "nutrition", "retail",
    "manufacturing", "energy", "pharmaceuticals", "transportation",
    "hospitality", "education", "legal", "hr", "marketing",
    "operations", "security", "data_science", "other",
]

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "technology": ["software", "api", "cloud", "server", "network", "database", "code", "devops", "kubernetes", "docker"],
    "finance": ["cost", "budget", "revenue", "financial", "roi", "profit", "expense", "investment", "billing"],
    "healthcare": ["patient", "clinical", "hospital", "medical", "diagnosis", "treatment", "care", "physician", "ehr"],
    "nutrition": ["food", "diet", "nutrition", "calorie", "meal", "supplement", "vitamin", "nutrient"],
    "retail": ["customer", "inventory", "store", "sales", "product", "warehouse", "fulfillment", "e-commerce"],
    "manufacturing": ["factory", "production", "assembly", "manufacturing", "supply chain", "quality control", "plant"],
    "energy": ["power", "energy", "electricity", "grid", "renewable", "solar", "utility", "generation"],
    "pharmaceuticals": ["drug", "pharmaceutical", "clinical trial", "fda", "molecule", "compound", "therapeutic"],
    "transportation": ["logistics", "transportation", "fleet", "delivery", "shipping", "route", "vehicle"],
    "hospitality": ["hotel", "hospitality", "restaurant", "guest", "reservation", "tourism", "catering"],
    "education": ["student", "school", "university", "learning", "curriculum", "training", "course", "classroom"],
    "legal": ["legal", "compliance", "regulation", "contract", "law", "audit", "policy", "governance"],
    "hr": ["employee", "hr", "hiring", "workforce", "onboarding", "performance", "recruitment", "salary"],
    "marketing": ["marketing", "campaign", "brand", "advertising", "seo", "social media", "conversion"],
    "operations": ["operations", "process", "workflow", "efficiency", "automation", "procedure", "incident"],
    "security": ["security", "vulnerability", "breach", "authentication", "encryption", "zero-trust", "firewall"],
    "data_science": ["ml", "model", "training", "inference", "data", "algorithm", "feature", "prediction", "llm"],
}

SYNONYM_MAP = {
    "it": "technology", "tech": "technology", "software engineering": "technology",
    "medicine": "healthcare", "hospital administration": "healthcare",
    "human resources": "hr", "people ops": "hr",
    "ecommerce": "retail", "e-commerce": "retail",
    "supply chain": "manufacturing",
    "pharma": "pharmaceuticals",
}


# ─── UTILITY ─────────────────────────────────────────────────────────────────

def approx_tokens(text: str) -> int:
    return max(1, len(text.split()))


def stable_doc_id(source_id: str, chunk_index: int) -> str:
    raw = f"corpus_{source_id}_{chunk_index}"
    return "corpus_" + hashlib.md5(raw.encode()).hexdigest()[:12]


def classify_text(text: str, metadata_category: Optional[str] = None) -> tuple[str, float, str]:
    """Return (category, confidence, method)."""
    # 1. Use existing metadata first
    if metadata_category:
        normalized = SYNONYM_MAP.get(metadata_category.lower(), metadata_category.lower())
        if normalized in CANONICAL_CATEGORIES:
            return normalized, 1.0, "metadata"
        # Partial match
        for cat in CANONICAL_CATEGORIES:
            if cat in normalized or normalized in cat:
                return cat, 0.9, "metadata_partial"

    # 2. Keyword scoring
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        scores[cat] = sum(1 for kw in keywords if kw in text_lower)

    best_cat = max(scores, key=scores.get)
    best_score = scores[best_cat]
    if best_score == 0:
        return "other", 0.5, "keyword_fallback"
    total = sum(scores.values()) or 1
    confidence = round(best_score / total, 3)
    return best_cat, confidence, "keyword"


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def curate(
    limit: Optional[int] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    dry_run: bool = False,
) -> dict:
    from datasets import load_dataset

    print(f"Loading dataset: ozguragrali/enterprise-knowledge-qa-dataset-gemini-flash-for-t5-large")
    ds = load_dataset("ozguragrali/enterprise-knowledge-qa-dataset-gemini-flash-for-t5-large", split="train")
    raw_count = len(ds)
    print(f"  Loaded {raw_count} raw records.")

    if limit:
        ds = ds.select(range(min(limit, raw_count)))
        print(f"  Limiting to {len(ds)} records.")

    # ── Group by unique_id (reconstruct logical documents) ──────────────────
    groups: dict[str, list] = {}
    for rec in ds:
        uid = str(rec.get("unique_id", rec.get("id", "unknown")))
        groups.setdefault(uid, []).append(rec)

    print(f"  Reconstructed {len(groups)} logical documents from {len(ds)} chunks.")

    curated_records: list[dict] = []
    seen_hashes: set[str] = set()

    for uid, chunks in groups.items():
        # Sort by chunk_id if available
        chunks.sort(key=lambda r: r.get("chunk_id", 0))

        # Reconstruct full document text
        full_text = " ".join(
            str(r.get("text", "")).strip() for r in chunks if str(r.get("text", "")).strip()
        )
        if not full_text or approx_tokens(full_text) < MIN_TEXT_TOKENS:
            continue  # Skip metadata-only fragments

        # Deduplication by content hash
        content_hash = hashlib.md5(full_text.encode()).hexdigest()
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)

        # Category classification
        meta_cat = chunks[0].get("category", "") if chunks else ""
        category, cat_conf, cat_method = classify_text(full_text, meta_cat)

        # Sub-chunk if needed
        sub_chunks = chunk_text(full_text, chunk_size, overlap)
        for ci, chunk in enumerate(sub_chunks):
            if approx_tokens(chunk) < MIN_TEXT_TOKENS:
                continue
            doc_id = stable_doc_id(uid, ci)
            curated_records.append({
                "doc_id": doc_id,
                "source_unique_id": uid,
                "chunk_index": ci,
                "text": chunk,
                "category": category,
                "category_confidence": cat_conf,
                "classification_method": cat_method,
                "source": "enterprise-qa",
                "namespace": "corpus",
            })

    print(f"  Curated {len(curated_records)} chunks from {len(groups)} documents.")

    if not dry_run:
        # Save JSONL
        CURATED_JSONL.parent.mkdir(parents=True, exist_ok=True)
        with open(CURATED_JSONL, "w", encoding="utf-8") as f:
            for rec in curated_records:
                f.write(json.dumps(rec) + "\n")
        print(f"  Saved curated JSONL → {CURATED_JSONL}")

        # Batch upsert into Chroma
        ingest_into_chroma(curated_records)

    # ── Manifest ─────────────────────────────────────────────────────────────
    cat_dist: dict[str, int] = {}
    for r in curated_records:
        cat_dist[r["category"]] = cat_dist.get(r["category"], 0) + 1

    manifest = {
        "source_dataset": "ozguragrali/enterprise-knowledge-qa-dataset-gemini-flash-for-t5-large",
        "source_version": "train",
        "raw_record_count": raw_count,
        "curated_case_count": len(groups),
        "curated_chunk_count": len(curated_records),
        "category_distribution": cat_dist,
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
        "chroma_collection": CHROMA_COLLECTION,
        "embedding_function": "chromadb_default",
        "corpus_version": "v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if not dry_run:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"  Manifest saved → {MANIFEST_PATH}")

    return manifest


def ingest_into_chroma(records: list[dict], batch_size: int = 200) -> None:
    """Batch upsert curated records into Chroma (idempotent)."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    coll = client.get_or_create_collection(CHROMA_COLLECTION)

    ids_before = coll.count()
    total_upserted = 0

    for i in range(0, len(records), batch_size):
        batch = records[i: i + batch_size]
        ids = [r["doc_id"] for r in batch]
        docs = [r["text"] for r in batch]
        metas = [
            {
                "namespace": r["namespace"],
                "source_unique_id": r["source_unique_id"],
                "category": r["category"],
                "category_confidence": r["category_confidence"],
                "classification_method": r["classification_method"],
                "source": r["source"],
            }
            for r in batch
        ]
        coll.upsert(ids=ids, documents=docs, metadatas=metas)
        total_upserted += len(ids)

    ids_after = coll.count()
    print(f"  Chroma: {ids_before} → {ids_after} documents (upserted {total_upserted} records).")


def main():
    parser = argparse.ArgumentParser(description="Curate and ingest knowledge corpus")
    parser.add_argument("--limit", type=int, default=None, help="Limit raw records for testing")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--dry-run", action="store_true", help="Parse only, do not write to Chroma")
    args = parser.parse_args()

    manifest = curate(
        limit=args.limit,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        dry_run=args.dry_run,
    )
    print("\n── Corpus Manifest ──────────────────────────────────────────────")
    print(f"  Curated cases  : {manifest['curated_case_count']}")
    print(f"  Curated chunks : {manifest['curated_chunk_count']}")
    print(f"  Categories     : {manifest['category_distribution']}")
    print(f"  Collection     : {manifest['chroma_collection']}")
    print(f"  Embedding fn   : {manifest['embedding_function']}")


if __name__ == "__main__":
    main()
