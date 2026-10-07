"""
model_registry.py -- Configurable model registry and loader.

All decoder-only GGUF models used in the AI4CARE evaluation must be registered
here. To add a new model, add an entry to MODEL_CONFIGS. The model does NOT
need to be a specific architecture; any GGUF compatible with llama-cpp-python
may be added, provided it has a known chat template.

IMPORTANT: The model registry is an experimental variable mechanism.
           The embedding model/function (Chroma built-in) is a FIXED CONSTANT.
           Models here are decoder-only GENERATORS, NOT embedding models.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / ".models"
CHROMA_DIR = BASE_DIR / ".chroma_db"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ─── SINGLE CHROMA COLLECTION ────────────────────────────────────────────────
# ALL models query the same collection. The embedding function is Chroma's
# built-in default, which remains constant across all experimental conditions.
CHROMA_COLLECTION = "ai4care_knowledge"

# ─── MODEL CONFIGS ────────────────────────────────────────────────────────────
# To add a model: add an entry below and set enabled=True.
# Only decoder-only GGUF models compatible with llama-cpp-python are supported.
MODEL_CONFIGS: dict[str, dict] = {
    "general_small": {
        "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "file": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "local": "general_small.gguf",
        "n_ctx": 2048,
        "enabled": True,
        "description": "TinyLlama 1.1B Chat (general, compact)",
    },
    "general_main": {
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "file": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "local": "general_main.gguf",
        "n_ctx": 4096,
        "enabled": True,
        "description": "Qwen2.5 1.5B Instruct (general, mid-size)",
    },
    "coder_small": {
        "repo": "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF",
        "file": "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
        "local": "coder_small.gguf",
        "n_ctx": 4096,
        "enabled": True,
        "description": "Qwen2.5-Coder 0.5B Instruct (code-specialised, compact)",
    },
    "coder_main": {
        "repo": "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        "file": "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        "local": "coder_main.gguf",
        "n_ctx": 4096,
        "enabled": True,
        "description": "Qwen2.5-Coder 1.5B Instruct (code-specialised, mid-size)",
    },
    # ── Additional models may be registered below ──────────────────────────
    # "phi3_mini": {
    #     "repo": "microsoft/Phi-3-mini-4k-instruct-gguf",
    #     "file": "Phi-3-mini-4k-instruct-q4.gguf",
    #     "local": "phi3_mini.gguf",
    #     "n_ctx": 4096,
    #     "enabled": False,
    #     "description": "Microsoft Phi-3 Mini 4K Instruct",
    # },
}

# The verbalized respondent arm needs a controlled constant
# We define respondent_main to map to Qwen2.5-7B-Instruct
MODEL_CONFIGS["respondent_main"] = {
    "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
    "file": "qwen2.5-7b-instruct-q4_k_m.gguf",
    "local": "respondent_main.gguf",
    "n_ctx": 4096,
    "enabled": False,
    "description": "Qwen2.5 7B Instruct (Verbalized Respondent)",
}

# ─── DYNAMIC GENERATED REGISTRY ──────────────────────────────────────────────
from estimator import LLMCapacityEngine

def _slugify(name: str) -> str:
    return name.lower().replace(" ", "-")

GENERATED_REGISTRY = {}
for m in LLMCapacityEngine.MODEL_CATALOG:
    name_slug = _slugify(m["name"])
    repo_gguf = m["repo"] + "-GGUF"
    if "/" not in repo_gguf:
        repo_gguf = m["repo"]  # fallback just in case
    GENERATED_REGISTRY[name_slug] = {
        "repo": repo_gguf,
        "n_ctx": min(m["max_context"], 32768)
    }



@dataclass
class ModelSpec:
    """Resolved model specification after validation."""
    model_id: str
    local_path: Path
    n_ctx: int
    description: str
    repo: str
    file: str
    enabled: bool
    # Filled after verification:
    load_success: bool = False
    generation_success: bool = False
    detected_chat_template: Optional[str] = None
    model_size_mb: float = 0.0
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_path": str(self.local_path),
            "model_size_mb": self.model_size_mb,
            "n_ctx": self.n_ctx,
            "description": self.description,
            "enabled": self.enabled,
            "load_success": self.load_success,
            "generation_success": self.generation_success,
            "detected_chat_template": self.detected_chat_template,
            "error_message": self.error_message,
        }


def get_enabled_models() -> list[str]:
    """Return model IDs that are enabled in the registry."""
    return [k for k, v in MODEL_CONFIGS.items() if v.get("enabled", False)]


def get_model_spec(model_id: str) -> ModelSpec:
    """Return a ModelSpec for the given model ID (does not verify)."""
    if model_id not in MODEL_CONFIGS:
        raise KeyError(f"Model '{model_id}' is not registered in MODEL_CONFIGS.")
    cfg = MODEL_CONFIGS[model_id]
    local_path = MODELS_DIR / cfg["local"]
    size_mb = local_path.stat().st_size / (1024 * 1024) if local_path.exists() else 0.0
    return ModelSpec(
        model_id=model_id,
        local_path=local_path,
        n_ctx=cfg.get("n_ctx", 2048),
        description=cfg.get("description", ""),
        repo=cfg.get("repo", ""),
        file=cfg.get("file", ""),
        enabled=cfg.get("enabled", False),
        model_size_mb=round(size_mb, 2),
    )


def _resolve_and_download_dynamic(model_id: str, quant: str = "Q4_K_M") -> ModelSpec:
    """Dynamically resolve and download a model from GENERATED_REGISTRY."""
    import json
    from huggingface_hub import list_repo_files, hf_hub_download
    import os

    if model_id not in GENERATED_REGISTRY:
        raise KeyError(f"Dynamic model '{model_id}' not found in GENERATED_REGISTRY")
    
    reg = GENERATED_REGISTRY[model_id]
    repo = reg["repo"]
    
    # Try looking in cache
    cache_file = DATA_DIR / f"repo_files_cache_{repo.replace('/', '_')}.json"
    files = []
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                files = json.load(f)
        except Exception:
            pass
            
    if not files:
        files = list_repo_files(repo)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(files, f)
            
    # Find the right file
    target_quant = quant.lower()
    chosen_file = None
    for f in files:
        f_lower = f.lower()
        if f_lower.endswith(".gguf") and target_quant in f_lower:
            chosen_file = f
            break
            
    if not chosen_file:
        raise RuntimeError(f"Could not find a .gguf matching {quant} in {repo}. Available: {files[:5]}...")

    local_path = MODELS_DIR / f"{model_id}_{quant}.gguf"
    
    if not local_path.exists() or local_path.stat().st_size <= 10_000:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        dl_path = hf_hub_download(repo_id=repo, filename=chosen_file)
        # symlink or copy to our .models directory
        if os.name == 'nt':
            import shutil
            shutil.copy2(dl_path, local_path)
        else:
            try:
                os.symlink(dl_path, local_path)
            except FileExistsError:
                pass
                
    size_mb = local_path.stat().st_size / (1024 * 1024)
    return ModelSpec(
        model_id=f"{model_id}_{quant}",
        local_path=local_path,
        n_ctx=reg["n_ctx"],
        description=f"Dynamic {model_id} ({quant})",
        repo=repo,
        file=chosen_file,
        enabled=True,
        model_size_mb=round(size_mb, 2)
    )

def get_runner(model_id: str, verbose: bool = False, quant: str = "Q4_K_M"):
    """Load and return a ChatRunner instance for the given model ID.
    Supports environment-based backend overrides (AI4CARE_BACKEND=openai).
    Raises RuntimeError explicitly if the model file is missing or backend fails.
    """
    import os
    backend = os.environ.get("AI4CARE_BACKEND", "local").lower()
    
    if backend == "openai":
        from inference import OpenAICompatibleRunner
        return OpenAICompatibleRunner(
            model_id=model_id,
            api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
            base_url=os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")
        )

    # Resolve spec from either static mapping or dynamic registry
    if model_id in MODEL_CONFIGS:
        spec = get_model_spec(model_id)
        if not spec.local_path.exists() or spec.local_path.stat().st_size <= 10_000:
            raise RuntimeError(
                f"[{model_id}] Model file not found or too small: {spec.local_path}\n"
                "Run 'python setup.py' to download models."
            )
    else:
        # Try dynamic resolution
        spec = _resolve_and_download_dynamic(model_id, quant)
    
    from inference import LlamaCppRunner
    n_threads = max(1, (os.cpu_count() or 4) - 1)
    return LlamaCppRunner(
        model_path=str(spec.local_path),
        n_ctx=spec.n_ctx,
        n_threads=n_threads,
        verbose=verbose
    )
