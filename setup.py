"""
setup.py -- Download GGUF models from HuggingFace and initialise ChromaDB collections.
Run once before starting interviews: python setup.py
"""

import os
from pathlib import Path
import chromadb

BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / ".models"
CHROMA_DIR = BASE_DIR / ".chroma_db"

MODELS = {
    "general_small": {
        "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "file": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "local": "general_small.gguf",
        "collection": "ai4care_general_small",
    },
    "general_main": {
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "file": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "local": "general_main.gguf",
        "collection": "ai4care_general_main",
    },
    "coder_small": {
        "repo": "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF",
        "file": "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
        "local": "coder_small.gguf",
        "collection": "ai4care_coder_small",
    },
    "coder_main": {
        "repo": "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        "file": "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        "local": "coder_main.gguf",
        "collection": "ai4care_coder_main",
    },
}

# Minimal seed documents to bootstrap empty ChromaDB collections
SEED_DOCS = [
    ("seed_0", "Team migrated care deployment pipeline after a major outage from unverified config changes.", {"category": "incident_response"}),
    ("seed_1", "Zero-trust model review workflow: every prompt template passes static analysis before staging.", {"category": "security_policy"}),
    ("seed_2", "Automated regression testing reduced reviewer overhead by 40% on knowledge artifact transitions.", {"category": "quality_assurance"}),
    ("seed_3", "Small quantized GGUF models on edge hardware eliminated latency spikes and API key dependencies.", {"category": "edge_inference"}),
]


def download_models():
    from huggingface_hub import hf_hub_download
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for model_id, cfg in MODELS.items():
        dest = MODELS_DIR / cfg["local"]
        if dest.exists() and dest.stat().st_size > 10_000:
            print(f"[OK] {model_id} already downloaded.")
            continue
        print(f"Downloading {model_id} ...")
        downloaded = Path(hf_hub_download(repo_id=cfg["repo"], filename=cfg["file"], local_dir=str(MODELS_DIR)))
        if downloaded != dest:
            import shutil
            shutil.copyfile(downloaded, dest)
            try:
                downloaded.unlink()
            except Exception:
                pass
        print(f"[OK] {model_id} saved to {dest}")


def init_chroma():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    for model_id, cfg in MODELS.items():
        coll = client.get_or_create_collection(cfg["collection"])
        if coll.count() == 0:
            ids, docs, metas = zip(*[(s[0] + f"_{model_id}", s[1], s[2]) for s in SEED_DOCS])
            coll.add(ids=list(ids), documents=list(docs), metadatas=list(metas))
            print(f"[OK] Seeded {cfg['collection']} with {len(SEED_DOCS)} documents.")
        else:
            print(f"[OK] {cfg['collection']} already initialised ({coll.count()} docs).")


if __name__ == "__main__":
    print("=== AI4CARE Setup ===")
    print("\n-- Downloading models --")
    download_models()
    print("\n-- Initialising ChromaDB --")
    init_chroma()
    print("\nSetup complete. Run: python cli.py")
