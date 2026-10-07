"""
<<<<<<< HEAD
seed_dataset.py -- Load the enterprise knowledge dataset into the single
                   Chroma collection with deduplication and batch upsert.

This is the LEGACY ingestion script, now updated to use the single
ai4care_knowledge collection. For full curation, use curate_seed_corpus.py.

Run once after setup.py (for basic seeding):
    python seed_dataset.py

For full curation pipeline:
    python curate_seed_corpus.py
=======
seed_dataset.py -- Load an external enterprise knowledge dataset into ChromaDB.
Run once after setup.py: python seed_dataset.py
>>>>>>> 663c32fa4d020ebf24239f0135157ae04f1562a9
"""

from pathlib import Path
from datasets import load_dataset
import chromadb
<<<<<<< HEAD

from model_registry import CHROMA_DIR, CHROMA_COLLECTION


def ingest_dataset(batch_size: int = 200) -> int:
    """Ingest dataset into the single shared Chroma collection.

    Uses upsert for idempotency — running twice does not create duplicates.
    Returns total documents ingested.
    """
    print("Loading enterprise knowledge dataset from HuggingFace...")
    ds = load_dataset("ozguragrali/enterprise-knowledge-qa-dataset-gemini-flash-for-t5-large", split="train")
    print(f"Loaded {len(ds)} records. Connecting to ChromaDB...")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    coll = client.get_or_create_collection(CHROMA_COLLECTION)

    print(f"Ingesting into collection: {CHROMA_COLLECTION}")
    total_inserted = 0

    for i in range(0, len(ds), batch_size):
        batch = ds[i: i + batch_size]
        ids, docs, metas = [], [], []
        for j in range(len(batch["text"])):
            uid = str(batch.get("unique_id", [f"unk_{i+j}"])[j] if "unique_id" in batch else f"unk_{i+j}")
            cid = str(batch.get("chunk_id", [j])[j] if "chunk_id" in batch else j)
            doc_id = f"seed_{uid}_{cid}"
            text = str(batch["text"][j]).strip()
            if not text or len(text.split()) < 5:
                continue
            ids.append(doc_id)
            docs.append(text)
            metas.append({
                "namespace": "corpus",
                "source": "enterprise-qa",
                "source_unique_id": uid,
            })
        if ids:
            coll.upsert(ids=ids, documents=docs, metadatas=metas)
            total_inserted += len(ids)

    print(f"Ingestion complete. Total upserted: {total_inserted}")
    print(f"Collection size: {coll.count()} documents")
    return total_inserted
=======
from setup import CHROMA_DIR, MODELS


def ingest_dataset():
    print("Loading enterprise knowledge dataset from HuggingFace...")
    ds = load_dataset('ozguragrali/enterprise-knowledge-qa-dataset-gemini-flash-for-t5-large', split='train')

    print(f"Loaded {len(ds)} records. Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    for model_id, cfg in MODELS.items():
        print(f"Populating collection for {model_id}...")
        coll = client.get_or_create_collection(cfg["collection"])

        batch_size = 200
        for i in range(0, len(ds), batch_size):
            batch = ds[i: i + batch_size]
            ids, docs, metas = [], [], []

            for j in range(len(batch['text'])):
                uniq_id = f"ext_{batch['unique_id'][j]}_{batch['chunk_id'][j]}_{model_id}"
                # Skip documents already present to avoid duplicates on reruns
                if coll.get(ids=[uniq_id])['ids']:
                    continue
                ids.append(uniq_id)
                docs.append(batch['text'][j])
                metas.append({"source": "enterprise-qa", "dataset": "ozguragrali", "indexed_for": model_id})

            if ids:
                coll.add(ids=ids, documents=docs, metadatas=metas)
                print(f"  Inserted {len(ids)} new documents into {cfg['collection']}")

    print("Ingestion complete.")
>>>>>>> 663c32fa4d020ebf24239f0135157ae04f1562a9


if __name__ == "__main__":
    ingest_dataset()
