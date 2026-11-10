"""
model_registry.py -- Configurable model registry and loader.

The thesis interview and judge panels are derived from the estimator catalog
and resolved dynamically. MODEL_CONFIGS retains a small static registry for
manual/legacy demonstrations. Dynamic models must have an estimator catalog
entry and a discoverable GGUF quantization compatible with llama-cpp-python.

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

# ─── STATIC MANUAL/LEGACY MODEL CONFIGS ────────────────────────────────────────
# Main thesis panels come from estimator.py and do not need a hard-coded entry.
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
from estimator import HardwareDetector, LLMCapacityEngine

def _slugify(name: str) -> str:
    return name.lower().replace(" ", "-")

GENERATED_REGISTRY = {}
for m in LLMCapacityEngine.MODEL_CATALOG:
    name_slug = _slugify(m["name"])
    model_name = m["repo"].split("/", 1)[-1]
    repo_gguf = m["repo"] + "-GGUF"
    GENERATED_REGISTRY[name_slug] = {
        "repo": repo_gguf,
        # GGUF conversions are often published by a quantization publisher
        # under a separate repository. Keep the source repo as a final option
        # in case it contains GGUF files itself.
        "repos": list(dict.fromkeys([
            repo_gguf,
            f"bartowski/{model_name}-GGUF",
            f"bartowski/{m['repo'].split('/', 1)[0]}_{model_name}-GGUF",
            f"mradermacher/{model_name}-GGUF",
            f"mradermacher/{m['repo'].split('/', 1)[0]}_{model_name}-GGUF",
            m["repo"],
        ])),
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


def select_interview_models(
    hardware: Optional[dict] = None,
    count: int = 8,
    quant: str = "Q4_K_M",
    context_length: int = 2048,
) -> tuple[list[str], list[dict]]:
    """Select the best-fitting compact panel from estimator benchmark results.

    The estimator first filters models that fit the available memory at the
    requested quantization/context. Among fitting models under 4B parameters,
    selection is deterministic: largest models first, preferring a new family
    on the first pass, then filling remaining places by size.
    """
    if hardware is None:
        hardware = HardwareDetector.get_system_hardware()
    if count < 1:
        raise ValueError("count must be positive")

    catalog_by_name = {item["name"]: item for item in LLMCapacityEngine.MODEL_CATALOG}
    tiny_rows = LLMCapacityEngine.evaluate(
        hardware, quant_level=quant, context_length=context_length
    ).get("Tiny (<4B)", [])
    candidates = []
    for row in tiny_rows:
        model = catalog_by_name.get(row["model_name"])
        if model is None or row["max_concurrent"] < 1:
            continue
        candidates.append({
            "model_id": _slugify(model["name"]),
            "name": model["name"],
            "repo": model["repo"],
            "family": model["family"],
            "params_b": model["params_b"],
            "quant": quant,
            "context_length": context_length,
            "estimated_memory_gb": row["vram_per_instance_gb"],
            "max_concurrent_at_estimate": row["max_concurrent"],
            "selection_source": "estimator_hardware_fit_ranked",
        })

    candidates.sort(key=lambda item: (-item["params_b"], item["name"].lower()))
    selected: list[dict] = []
    used_families: set[str] = set()
    for candidate in candidates:
        if candidate["family"] not in used_families:
            selected.append(candidate)
            used_families.add(candidate["family"])
            if len(selected) == count:
                break
    if len(selected) < count:
        selected_ids = {item["model_id"] for item in selected}
        for candidate in candidates:
            if candidate["model_id"] in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(candidate["model_id"])
            if len(selected) == count:
                break
    if len(selected) < count:
        raise RuntimeError(
            f"The estimator found only {len(selected)} fitting models below 4B; "
            f"the thesis panel requires {count}. Reduce count or use hardware with more memory."
        )
    return [item["model_id"] for item in selected], selected


def get_enabled_models(hardware: Optional[dict] = None) -> list[str]:
    """Return the estimator-selected compact interview panel."""
    return select_interview_models(hardware=hardware)[0]


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
    """Load a cached estimator-selected GGUF or resolve and download one."""
    import json
    import os
    import re

    if model_id not in GENERATED_REGISTRY:
        raise KeyError(f"Dynamic model '{model_id}' not found in GENERATED_REGISTRY")
    
    reg = GENERATED_REGISTRY[model_id]
    local_path = MODELS_DIR / f"{model_id}_{quant}.gguf"
    metadata_path = MODELS_DIR / f"{model_id}_{quant}.json"
    if local_path.exists() and local_path.stat().st_size > 10_000:
        return ModelSpec(
            model_id=f"{model_id}_{quant}",
            local_path=local_path,
            n_ctx=reg["n_ctx"],
            description=f"Dynamic {model_id} ({quant})",
            repo=reg["repo"],
            file=local_path.name,
            enabled=True,
            model_size_mb=round(local_path.stat().st_size / (1024 * 1024), 2),
        )

    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            cached_path = MODELS_DIR / cached["path"]
            if cached_path.exists() and cached_path.stat().st_size > 10_000:
                return ModelSpec(
                    model_id=f"{model_id}_{quant}",
                    local_path=cached_path,
                    n_ctx=reg["n_ctx"],
                    description=f"Dynamic {model_id} ({quant})",
                    repo=cached.get("repo", reg["repo"]),
                    file=cached.get("file", cached_path.name),
                    enabled=True,
                    model_size_mb=round(cached_path.stat().st_size / (1024 * 1024), 2),
                )
        except (OSError, KeyError, json.JSONDecodeError, TypeError):
            pass

    # Check the local estimator-selected file before touching Hugging Face.
    # This allows benchmark runs on offline machines after the model is cached.
    from huggingface_hub import hf_hub_download, list_repo_files

    repos = reg.get("repos", [reg["repo"]])
    target_quant = quant.lower()
    chosen = None
    repo_errors = []
    for repo in repos:
        cache_file = DATA_DIR / f"repo_files_cache_{repo.replace('/', '_')}.json"
        files = []
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    files = json.load(f)
            except (OSError, json.JSONDecodeError):
                files = []

        if not files:
            try:
                files = list_repo_files(repo)
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(files, f)
            except Exception as exc:
                repo_errors.append(f"{repo}: {exc}")
                continue

        matches = [
            filename for filename in files
            if filename.lower().endswith(".gguf") and target_quant in filename.lower()
        ]
        if matches:
            # Prefer a single-file checkpoint when one is available.
            matches.sort(key=lambda filename: ("-00001-of-" in filename.lower(), filename.lower()))
            chosen_file = matches[0]
            split_match = re.match(r"^(.*)-\d{5}-of-(\d{5})\.gguf$", chosen_file, re.IGNORECASE)
            if split_match:
                shard_pattern = re.compile(
                    rf"^{re.escape(split_match.group(1))}-\d{{5}}-of-{split_match.group(2)}\.gguf$",
                    re.IGNORECASE,
                )
                shard_files = sorted(
                    filename for filename in matches if shard_pattern.match(filename)
                )
            else:
                shard_files = [chosen_file]
            chosen = (repo, chosen_file, shard_files)
            break

    if not chosen:
        details = "; ".join(repo_errors) if repo_errors else "no matching GGUF file found"
        raise RuntimeError(f"Could not resolve a {quant} GGUF for '{model_id}'. Checked {repos}: {details}")

    repo, chosen_file, shard_files = chosen
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if len(shard_files) > 1:
        shard_dir = MODELS_DIR / f"{model_id}_{quant}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        downloaded = {}
        for shard_file in shard_files:
            downloaded[shard_file] = Path(
                hf_hub_download(repo_id=repo, filename=shard_file, local_dir=str(shard_dir))
            )
        local_path = downloaded[chosen_file]
    else:
        dl_path = hf_hub_download(repo_id=repo, filename=chosen_file)
        if os.name == "nt":
            import shutil
            temporary_path = local_path.with_suffix(local_path.suffix + ".part")
            try:
                os.link(os.path.realpath(dl_path), temporary_path)
            except OSError:
                shutil.copy2(dl_path, temporary_path)
            os.replace(temporary_path, local_path)
        else:
            try:
                os.symlink(dl_path, local_path)
            except FileExistsError:
                pass

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "repo": repo,
            "file": chosen_file,
            "path": str(local_path.relative_to(MODELS_DIR)),
        }, f, indent=2)

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

def get_runner(
    model_id: str,
    verbose: bool = False,
    quant: str = "Q4_K_M",
    n_ctx: Optional[int] = None,
    n_threads: Optional[int] = None,
    n_gpu_layers: Optional[int] = None,
    n_batch: Optional[int] = None,
):
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

    if n_ctx is not None:
        if n_ctx <= 0:
            raise ValueError("n_ctx must be greater than zero")
        spec.n_ctx = min(n_ctx, spec.n_ctx)
    
    from inference import LlamaCppRunner
    from estimator import HardwareDetector

    if n_threads is None:
        n_threads = max(1, (os.cpu_count() or 4) - 1)
    if n_gpu_layers is None:
        override = os.environ.get("AI4CARE_GPU_LAYERS")
        if override is not None:
            n_gpu_layers = int(override)
        else:
            hardware = HardwareDetector.get_system_hardware()
            n_gpu_layers = -1 if hardware.get("gpu_count", 0) else 0
    if n_batch is None:
        n_batch = int(os.environ.get("AI4CARE_N_BATCH", "512"))
    return LlamaCppRunner(
        model_path=str(spec.local_path),
        n_ctx=spec.n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        verbose=verbose
    )
