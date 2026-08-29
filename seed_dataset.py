"""
seed_dataset.py -- Load an external enterprise knowledge dataset into ChromaDB.
Run once after setup.py: python seed_dataset.py
"""

from pathlib import Path
from datasets import load_dataset
import chromadb
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


if __name__ == "__main__":
    ingest_dataset()
