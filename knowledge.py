"""
<<<<<<< HEAD
knowledge.py -- ChromaDB helpers: query, store, and inspect knowledge.

ARCHITECTURE:
    ONE collection: ai4care_knowledge
    ALL generator models query the same collection.
    The embedding function is Chroma's built-in default (fixed constant).
    Atomic facts are stored with deterministic IDs and rich metadata.

This module is intentionally kept thin. The engine and evaluation harness
are responsible for the higher-level logic.
=======
knowledge.py -- ChromaDB helpers: query, store, and inspect knowledge collections.
Each model has its own collection. Completed interviews are stored across all four.
>>>>>>> 663c32fa4d020ebf24239f0135157ae04f1562a9
"""

import uuid
from pathlib import Path
<<<<<<< HEAD
from typing import Optional
import chromadb

from model_registry import CHROMA_DIR, CHROMA_COLLECTION


_CLIENT: Optional[chromadb.PersistentClient] = None

def _get_client() -> chromadb.PersistentClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _CLIENT


def _get_collection(client: Optional[chromadb.PersistentClient] = None):
    """Return the single shared Chroma collection."""
    c = client or _get_client()
    return c.get_or_create_collection(CHROMA_COLLECTION)


# ─── QUERY ──────────────────────────────────────────────────────────────────

def query(text: str, n: int = 5, where: Optional[dict] = None) -> list[dict]:
    """Retrieve the top-n relevant documents from the shared collection.

    Args:
        text: Query string (embedded by Chroma's built-in function).
        n:    Number of results to return.
        where: Optional Chroma metadata filter dict.

    Returns:
        List of dicts with keys: doc_id, text, score, metadata.
    """
    client = _get_client()
    coll = _get_collection(client)
    count = coll.count()
    if count == 0:
        return []
    limit = min(n, count)
    kwargs = {"query_texts": [text], "n_results": limit}
    if where:
        kwargs["where"] = where
    results = coll.query(**kwargs)
    docs = results.get("documents", [[]])[0]
    dists = results.get("distances", [[0.0] * len(docs)])[0]
    ids = results.get("ids", [[]])[0]
    metas = results.get("metadatas", [[{}] * len(docs)])[0]
    return [
        {
            "doc_id": ids[i],
            "text": docs[i],
            "score": round(1.0 / (1.0 + float(dists[i])), 4),
            "distance": round(float(dists[i]), 4),
            "metadata": metas[i],
        }
        for i in range(len(docs))
    ]


# ─── STORE ATOMIC FACT ───────────────────────────────────────────────────────

def store_atomic_fact(
    *,
    fact_id: str,
    interview_id: str,
    turn_id: int,
    field: str,
    value: str,
    evidence: str,
    confidence: float,
    status: str,
    model_id: str,
    source_turn_ids: list[int],
    namespace: str = "interview_fact",
) -> str:
    """Store one atomic extracted fact into the shared collection.

    Atomic facts are stored with a deterministic ID and rich metadata so they
    can later be retrieved, filtered by namespace, and excluded from seed-corpus
    queries.

    Returns:
        The stored document ID.
    """
    client = _get_client()
    coll = _get_collection(client)
    metadata = {
        "namespace": namespace,
        "interview_id": interview_id,
        "turn_id": turn_id,
        "field": field,
        "confidence": round(float(confidence), 4),
        "status": status,
        "model_id": model_id,
        "source_turn_ids": ",".join(str(t) for t in source_turn_ids),
    }
    coll.upsert(
        ids=[fact_id],
        documents=[f"{field}: {value}\nEvidence: {evidence}"],
        metadatas=[metadata],
    )
    return fact_id


def store_interview_artifact(
    *,
    run_id: str,
    model_id: str,
    turns: int,
    summary_text: str,
    extra_metadata: Optional[dict] = None,
) -> str:
    """Store a completed interview summary artifact.

    Returns:
        A deterministic artifact ID.
    """
    artifact_id = f"artifact_{run_id}"
    client = _get_client()
    coll = _get_collection(client)
    metadata = {
        "namespace": "interview_artifact",
        "run_id": run_id,
        "model_id": model_id,
        "turns": turns,
        **(extra_metadata or {}),
    }
    coll.upsert(ids=[artifact_id], documents=[summary_text], metadatas=[metadata])
    return artifact_id


# ─── DELETE EPISODE RECORDS ──────────────────────────────────────────────────

def delete_episode_records(interview_id: str) -> int:
    """Delete all runtime facts and artifacts for a given interview_id.

    Used to enforce the clean-slate requirement between episodes.
    NEVER deletes seed-corpus documents (namespace="corpus").

    Returns:
        Number of records deleted.
    """
    client = _get_client()
    coll = _get_collection(client)
    try:
        existing = coll.get(where={"interview_id": interview_id})
        ids_to_delete = existing.get("ids", [])
        if ids_to_delete:
            coll.delete(ids=ids_to_delete)
        return len(ids_to_delete)
    except Exception as exc:
        raise RuntimeError(f"Failed to delete episode records for {interview_id}: {exc}") from exc


# ─── INSPECT ─────────────────────────────────────────────────────────────────

def inspect() -> dict:
    """Return document counts and namespace breakdown for the shared collection."""
    client = _get_client()
    coll = _get_collection(client)
    count = coll.count()
    sample = coll.peek(limit=3)["documents"] if count > 0 else []
    return {
        "collection": CHROMA_COLLECTION,
        "total_documents": count,
        "sample": sample,
    }
=======
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
>>>>>>> 663c32fa4d020ebf24239f0135157ae04f1562a9
