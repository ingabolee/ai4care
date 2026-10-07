"""
setup.py -- Download GGUF models from HuggingFace and initialise the single
            ChromaDB collection used for all experiments.

Run once before starting interviews or evaluations:
    python setup.py

ARCHITECTURE NOTE:
    There is exactly ONE Chroma collection: ai4care_knowledge
    All generator models query the same collection.
    The embedding function is Chroma's built-in default (fixed constant).
    The decoder-only GGUF model is the experimental variable.
"""

from pathlib import Path
import chromadb

from model_registry import (
    MODEL_CONFIGS,
    MODELS_DIR,
    CHROMA_DIR,
    CHROMA_COLLECTION,
)

# Backwards-compatible alias so that existing imports from setup continue working
MODELS = MODEL_CONFIGS

# Minimal seed documents to bootstrap the collection on first run
SEED_DOCS = [
    ("seed_0", "Team migrated care deployment pipeline after a major outage from unverified config changes.", {"category": "incident_response", "source": "seed"}),
    ("seed_1", "Zero-trust model review workflow: every prompt template passes static analysis before staging.", {"category": "security_policy", "source": "seed"}),
    ("seed_2", "Automated regression testing reduced reviewer overhead by 40% on knowledge artifact transitions.", {"category": "quality_assurance", "source": "seed"}),
    ("seed_3", "Small quantized GGUF models on edge hardware eliminated latency spikes and API key dependencies.", {"category": "edge_inference", "source": "seed"}),
]


def download_models(model_ids: list | None = None):
    """Download GGUF files for all enabled models (or a specific subset)."""
    from huggingface_hub import hf_hub_download
    import shutil

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    targets = model_ids or [k for k, v in MODEL_CONFIGS.items() if v.get("enabled", False)]

    for model_id in targets:
        cfg = MODEL_CONFIGS.get(model_id)
        if not cfg:
            print(f"[WARN] Unknown model_id '{model_id}' — skipping.")
            continue
        dest = MODELS_DIR / cfg["local"]
        if dest.exists() and dest.stat().st_size > 10_000:
            print(f"[OK] {model_id} already downloaded ({dest.stat().st_size // 1_048_576} MB).")
            continue
        print(f"Downloading {model_id} from {cfg['repo']} ...")
        downloaded = Path(hf_hub_download(repo_id=cfg["repo"], filename=cfg["file"], local_dir=str(MODELS_DIR)))
        if downloaded != dest:
            shutil.copyfile(downloaded, dest)
            try:
                downloaded.unlink()
            except Exception:
                pass
        print(f"[OK] {model_id} saved to {dest}")


def init_chroma():
    """Create (or verify) the single ai4care_knowledge collection and seed it."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Single collection — no model-specific namespaces
    coll = client.get_or_create_collection(CHROMA_COLLECTION)
    print(f"[OK] Collection '{CHROMA_COLLECTION}' ready ({coll.count()} documents already present).")

    if coll.count() == 0:
        ids, docs, metas = zip(*SEED_DOCS)
        coll.add(ids=list(ids), documents=list(docs), metadatas=list(metas))
        print(f"[OK] Seeded '{CHROMA_COLLECTION}' with {len(SEED_DOCS)} bootstrap documents.")
    else:
        print(f"[INFO] Collection already contains documents — skipping seed insertion.")

    # Record embedding function identity where available
    try:
        ef = getattr(coll, "_embedding_function", None) or getattr(coll, "embedding_function", None)
        ef_name = type(ef).__name__ if ef is not None else "chromadb_default"
        print(f"[INFO] Embedding function: {ef_name}")
    except Exception:
        print("[INFO] Embedding function: chromadb_default (could not introspect)")


if __name__ == "__main__":
    print("=== AI4CARE Setup ===")
    print("\n-- Downloading models --")
    download_models()
    print("\n-- Initialising ChromaDB --")
    init_chroma()
    print("\nSetup complete.")
    print("Next steps:")
    print("  python curate_seed_corpus.py          # curate and ingest the knowledge corpus")
    print("  python compile_benchmark_dataset.py   # compile frozen benchmark Q/A pairs")
    print("  python verify_models.py               # verify all models pass generation tests")
    print("  python evaluate.py --models all --iterations 3 --cases 50")
