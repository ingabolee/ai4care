"""
knowledge.py -- ChromaDB helpers: query, store, and inspect knowledge collections.
Each model has its own collection. Completed interviews are stored across all four.
"""

import uuid
from pathlib import Path
import chromadb

from setup import CHROMA_DIR, MODELS


def _get_client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def query(model_id: str, text: str, n: int = 2) -> list[dict]:
    """Retrieve the top-n relevant documents from the model's collection."""
    client = _get_client()
    cfg = MODELS.get(model_id, MODELS["general_small"])
    coll = client.get_or_create_collection(cfg["collection"])
    if coll.count() == 0:
        return []
    results = coll.query(query_texts=[text], n_results=min(n, coll.count()))
    docs = results.get("documents", [[]])[0]
    dists = results.get("distances", [[0.0] * len(docs)])[0]
    return [{"text": d, "score": round(1.0 / (1.0 + float(dist)), 3)} for d, dist in zip(docs, dists)]


def store_all(text: str, metadata: dict) -> str:
    """Store a completed interview artifact across all four model collections."""
    client = _get_client()
    stem = f"artifact_{str(uuid.uuid4())[:8]}"
    for model_id, cfg in MODELS.items():
        coll = client.get_or_create_collection(cfg["collection"])
        meta = {**metadata, "indexed_for": model_id}
        coll.add(ids=[f"{stem}_{model_id}"], documents=[text], metadatas=[meta])
    return stem


def inspect() -> dict:
    """Return document counts and a sample from each collection."""
    client = _get_client()
    summary = {}
    for model_id, cfg in MODELS.items():
        coll = client.get_or_create_collection(cfg["collection"])
        count = coll.count()
        sample = coll.peek(limit=3)["documents"] if count > 0 else []
        summary[model_id] = {"collection": cfg["collection"], "count": count, "sample": sample}
    return summary
