import math
import os
import platform
from typing import Any, Dict, List, Optional

import psutil

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    pynvml = None
    HAS_NVML = False


class HardwareDetector:
    """Detect accelerator memory on NVIDIA, Apple Silicon, or CPU-only systems."""

    @staticmethod
    def get_system_hardware() -> Dict[str, Any]:
        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
        info = {
            "os": platform.system(),
            "machine": platform.machine(),
            "cpu_cores": psutil.cpu_count(logical=True),
            "ram_gb": ram_gb,
            "vram_gb": 0.0,
            "gpu_name": "CPU Only",
            "gpu_count": 0,
            "memory_kind": "system_ram",
            "is_unified_memory": False,
        }

        # Apple Silicon has a shared CPU/GPU memory pool; it is not fixed VRAM.
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            info.update({
                "gpu_name": "Apple Silicon GPU (Unified Memory)",
                "vram_gb": ram_gb,
                "gpu_count": 1,
                "memory_kind": "unified_memory",
                "is_unified_memory": True,
            })
            return info

        if HAS_NVML:
            try:
                pynvml.nvmlInit()
                n = pynvml.nvmlDeviceGetCount()
                if n:
                    total = 0
                    names = []
                    for i in range(n):
                        h = pynvml.nvmlDeviceGetHandleByIndex(i)
                        m = pynvml.nvmlDeviceGetMemoryInfo(h)
                        raw = pynvml.nvmlDeviceGetName(h)
                        name = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
                        total += m.total
                        names.append(name)
                    info.update({
                        "gpu_name": f"{n}x {names[0]}" if n > 1 else names[0],
                        "vram_gb": round(total / (1024 ** 3), 2),
                        "gpu_count": n,
                        "memory_kind": "dedicated_gpu",
                    })
                    return info
            except Exception:
                pass
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass

        if HAS_TORCH and torch is not None:
            try:
                if torch.cuda.is_available():
                    n = torch.cuda.device_count()
                    total = sum(torch.cuda.get_device_properties(i).total_memory for i in range(n))
                    info.update({
                        "gpu_name": f"{n}x {torch.cuda.get_device_name(0)}" if n > 1 else torch.cuda.get_device_name(0),
                        "vram_gb": round(total / (1024 ** 3), 2),
                        "gpu_count": n,
                        "memory_kind": "dedicated_gpu",
                    })
                    return info
            except Exception:
                pass

            try:
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    info.update({
                        "gpu_name": "Apple GPU (MPS / Unified Memory)",
                        "vram_gb": ram_gb,
                        "gpu_count": 1,
                        "memory_kind": "unified_memory",
                        "is_unified_memory": True,
                    })
                    return info
            except Exception:
                pass

        # CPU inference uses host RAM as the model-memory pool.
        info["vram_gb"] = ram_gb
        return info


def model(
    name: str,
    repo: str,
    family: str,
    params_b: float,
    layers: int,
    kv_heads: int,
    head_dim: int,
    max_context: int,
    *,
    active_params_b: Optional[float] = None,
    arch_type: str = "Decoder-only",
    memory_model: str = "standard_kv",
    num_experts: Optional[int] = None,
    experts_per_token: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "name": name,
        "repo": repo,
        "family": family,
        "params_b": params_b,
        "active_params_b": params_b if active_params_b is None else active_params_b,
        "layers": layers,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "max_context": max_context,
        "arch_type": arch_type,
        "memory_model": memory_model,
        "num_experts": num_experts,
        "experts_per_token": experts_per_token,
    }


class LLMCapacityEngine:
    """Memory estimator for models with standard MHA/GQA/MQA KV caches."""

    # Packed-weight approximations. Runtime-specific formats can differ slightly.
    WEIGHT_BYTES_PER_PARAM = {
        "FP16": 2.0,
        "BF16": 2.0,
        "Q8_0": 1.0625,
        "Q6_K": 0.8125,
        "Q5_K_M": 0.6875,
        "Q4_K_M": 0.625,
        "Q4_0": 0.5625,
    }

    KV_CACHE_BYTES = 2.0       # FP16/BF16 KV cache
    RUNTIME_OVERHEAD = 0.15    # workspace, activations, kernels, metadata, etc.
    MEMORY_RESERVE = 0.10      # leave headroom for the OS/runtime

    # Major official checkpoints that fit the standard KV-cache equation.
    MODEL_CATALOG = [
        # Qwen 2.5
        model("Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5", 0.49, 24, 2, 64, 32768),
        model("Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5", 1.54, 28, 2, 128, 32768),
        model("Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "Qwen2.5", 3.09, 36, 2, 128, 32768),
        model("Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "Qwen2.5", 7.61, 28, 4, 128, 32768),
        model("Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "Qwen2.5", 14.7, 48, 8, 128, 32768),
        model("Qwen2.5-32B-Instruct", "Qwen/Qwen2.5-32B-Instruct", "Qwen2.5", 32.5, 64, 8, 128, 32768),
        model("Qwen2.5-72B-Instruct", "Qwen/Qwen2.5-72B-Instruct", "Qwen2.5", 72.7, 80, 8, 128, 32768),

        # Qwen 3 dense + ordinary-KV MoE
        model("Qwen3-0.6B", "Qwen/Qwen3-0.6B", "Qwen3", 0.60, 28, 8, 128, 32768),
        model("Qwen3-1.7B", "Qwen/Qwen3-1.7B", "Qwen3", 1.78, 28, 8, 128, 40960),
        model("Qwen3-4B", "Qwen/Qwen3-4B", "Qwen3", 4.02, 36, 8, 128, 40960),
        model("Qwen3-8B", "Qwen/Qwen3-8B", "Qwen3", 8.19, 36, 8, 128, 40960),
        model("Qwen3-14B", "Qwen/Qwen3-14B", "Qwen3", 14.8, 40, 8, 128, 40960),
        model("Qwen3-32B", "Qwen/Qwen3-32B", "Qwen3", 32.8, 64, 8, 128, 40960),
        model("Qwen3-30B-A3B", "Qwen/Qwen3-30B-A3B", "Qwen3 MoE", 30.5, 48, 4, 128, 40960,
              active_params_b=3.3, arch_type="MoE Decoder-only", num_experts=128, experts_per_token=8),
        model("Qwen3-235B-A22B", "Qwen/Qwen3-235B-A22B", "Qwen3 MoE", 235.0, 94, 4, 128, 40960,
              active_params_b=22.0, arch_type="MoE Decoder-only", num_experts=128, experts_per_token=8),

        # Meta Llama 3.x
        model("Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-1B-Instruct", "Llama 3.2", 1.23, 16, 8, 64, 131072),
        model("Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct", "Llama 3.2", 3.21, 28, 8, 128, 131072),
        model("Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "Llama 3.1", 8.03, 32, 8, 128, 131072),
        model("Llama-3.1-70B-Instruct", "meta-llama/Llama-3.1-70B-Instruct", "Llama 3.1", 70.6, 80, 8, 128, 131072),
        model("Llama-3.1-405B-Instruct", "meta-llama/Llama-3.1-405B-Instruct", "Llama 3.1", 405.0, 126, 8, 128, 131072),
        model("Llama-3.3-70B-Instruct", "meta-llama/Llama-3.3-70B-Instruct", "Llama 3.3", 70.6, 80, 8, 128, 131072),

        # Mistral
        model("Mistral-7B-Instruct-v0.3", "mistralai/Mistral-7B-Instruct-v0.3", "Mistral", 7.25, 32, 8, 128, 32768),
        model("Mistral-Nemo-Instruct-2407", "mistralai/Mistral-Nemo-Instruct-2407", "Mistral", 12.2, 40, 8, 128, 131072),
        model("Mistral-Large-Instruct-2411", "mistralai/Mistral-Large-Instruct-2411", "Mistral", 123.0, 88, 8, 128, 131072),

        # Microsoft Phi
        model("Phi-3.5-mini-instruct", "microsoft/Phi-3.5-mini-instruct", "Phi", 3.82, 32, 32, 96, 131072),
        model("Phi-4-mini-instruct", "microsoft/Phi-4-mini-instruct", "Phi", 3.8, 32, 8, 128, 131072),
        model("Phi-4", "microsoft/phi-4", "Phi", 14.7, 40, 10, 128, 16384),

        # DeepSeek R1 distilled dense checkpoints
        model("DeepSeek-R1-Distill-Qwen-1.5B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek R1 Distill", 1.78, 28, 2, 128, 131072),
        model("DeepSeek-R1-Distill-Qwen-7B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "DeepSeek R1 Distill", 7.62, 28, 4, 128, 131072),
        model("DeepSeek-R1-Distill-Qwen-14B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "DeepSeek R1 Distill", 14.8, 48, 8, 128, 131072),
        model("DeepSeek-R1-Distill-Qwen-32B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "DeepSeek R1 Distill", 32.8, 64, 8, 128, 131072),
        model("DeepSeek-R1-Distill-Llama-8B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "DeepSeek R1 Distill", 8.03, 32, 8, 128, 131072),
        model("DeepSeek-R1-Distill-Llama-70B", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "DeepSeek R1 Distill", 70.6, 80, 8, 128, 131072),

        # Ministral 3 text towers use ordinary KV attention, but the full model
        # is multimodal. Keep them in the special list rather than undercounting.
    ]

    # Current important models that this simple formula must NOT pretend to estimate.
    SPECIAL_MODEL_CATALOG = [
        ("Qwen3.5-2B", "Hybrid linear + full attention cache"),
        ("Qwen3.5-9B", "Hybrid linear + full attention cache"),
        ("Qwen3.5-27B", "Hybrid linear attention + vision"),
        ("Qwen3.5-35B-A3B", "Hybrid linear attention + MoE + multimodal"),
        ("Qwen3.8-27B", "Hybrid linear + full attention cache"),
        ("Qwen3.8-2.4T-A95B", "Hybrid linear-attention MoE"),
        ("Gemma-4-12B", "Hybrid sliding/full attention + unified multimodal model"),
        ("Gemma-4-26B-A4B", "Hybrid attention + MoE + multimodal model"),
        ("Gemma-4-31B-it", "Hybrid sliding/full attention + vision"),
        ("Llama-4-Scout", "Native multimodal MoE"),
        ("Llama-4-Maverick", "Native multimodal MoE"),
        ("DeepSeek-V4-Flash", "Specialized latent/MLA-style attention cache"),
        ("DeepSeek-V4-Pro", "Specialized latent/MLA-style attention cache"),
        ("Mistral-Large-3", "Sparse MoE multimodal model"),
        ("GLM-5", "MoE + DSA/latent attention cache"),
        ("bge-large-en-v1.5", "Encoder-only embedding model"),
        ("FLAN-T5-XL", "Encoder-decoder; separate encoder/decoder accounting required"),
    ]

    @staticmethod
    def size_category(params_b: float) -> str:
        if params_b < 4:
            return "Tiny (<4B)"
        if params_b < 15:
            return "Small (4B-15B)"
        return "Large (>15B)"

    @classmethod
    def validate_catalog(cls) -> None:
        seen = set()
        for m in cls.MODEL_CATALOG:
            if m["name"] in seen:
                raise ValueError(f"Duplicate model: {m['name']}")
            seen.add(m["name"])
            if m["params_b"] <= 0 or m["layers"] <= 0 or m["kv_heads"] <= 0 or m["head_dim"] <= 0:
                raise ValueError(f"Invalid dimensions for {m['name']}")
            if m["active_params_b"] > m["params_b"]:
                raise ValueError(f"active_params_b > params_b for {m['name']}")

    @classmethod
    def estimate_weights_gb(cls, m: Dict[str, Any], quant_level: str) -> float:
        try:
            bpp = cls.WEIGHT_BYTES_PER_PARAM[quant_level]
        except KeyError as exc:
            raise ValueError(f"Unsupported quantization: {quant_level}") from exc
        return m["params_b"] * bpp

    @classmethod
    def estimate_kv_cache_gb(cls, m: Dict[str, Any], context_length: int) -> float:
        if context_length <= 0:
            raise ValueError("context_length must be > 0")
        if context_length > m["max_context"]:
            raise ValueError(
                f"{m['name']} max context is {m['max_context']:,}; requested {context_length:,}"
            )
        # K and V are both stored: 2 * L * KV_heads * head_dim * tokens * bytes.
        total = 2 * m["layers"] * m["kv_heads"] * m["head_dim"] * context_length * cls.KV_CACHE_BYTES
        return total / 1e9

    @classmethod
    def estimate_vram_gb(cls, m: Dict[str, Any], quant_level: str, context_length: int) -> float:
        weights = cls.estimate_weights_gb(m, quant_level)
        kv = cls.estimate_kv_cache_gb(m, context_length)
        return round((weights + kv) * (1 + cls.RUNTIME_OVERHEAD), 2)

    @classmethod
    def evaluate(cls, hardware: Dict[str, Any], quant_level: str, context_length: int):
        total_memory = hardware["vram_gb"]
        usable = total_memory * (1 - cls.MEMORY_RESERVE)
        results = {"Tiny (<4B)": [], "Small (4B-15B)": [], "Large (>15B)": []}

        for m in cls.MODEL_CATALOG:
            needed = cls.estimate_vram_gb(m, quant_level, context_length)
            concurrent = math.floor(usable / needed) if needed else 0
            results[cls.size_category(m["params_b"])].append({
                "model_name": m["name"],
                "params_b": m["params_b"],
                "active_params_b": m["active_params_b"],
                "arch_type": m["arch_type"],
                "vram_per_instance_gb": needed,
                "max_concurrent": concurrent,
            })
        return results

    @classmethod
    def find_two_model_combinations(
        cls,
        hardware: Dict[str, Any],
        quant_level: str,
        context_length: int,
    ) -> List[Dict[str, Any]]:
        """Find pairs of two different-sized models that fit at the same time."""
        usable = hardware["vram_gb"] * (1 - cls.MEMORY_RESERVE)
        model_rows = []

        for m in cls.MODEL_CATALOG:
            needed = cls.estimate_vram_gb(m, quant_level, context_length)
            model_rows.append({
                "model_name": m["name"],
                "category": cls.size_category(m["params_b"]),
                "params_b": m["params_b"],
                "vram_gb": needed,
            })

        combinations = []
        for i in range(len(model_rows)):
            a = model_rows[i]
            for j in range(i + 1, len(model_rows)):
                b = model_rows[j]

                # User requested different models AND different model-size categories.
                if a["model_name"] == b["model_name"]:
                    continue
                if a["category"] == b["category"]:
                    continue

                combined = a["vram_gb"] + b["vram_gb"]
                if combined <= usable:
                    combinations.append({
                        "model_a": a["model_name"],
                        "category_a": a["category"],
                        "vram_a_gb": a["vram_gb"],
                        "model_b": b["model_name"],
                        "category_b": b["category"],
                        "vram_b_gb": b["vram_gb"],
                        "combined_vram_gb": round(combined, 2),
                        "remaining_gb": round(usable - combined, 2),
                    })

        # Show the combinations that use the most of the available budget first.
        combinations.sort(key=lambda x: (-x["combined_vram_gb"], x["model_a"], x["model_b"]))
        return combinations


def print_report(hardware: Dict[str, Any], results, combinations, quant_level: str, context_length: int) -> None:
    usable = hardware["vram_gb"] * (1 - LLMCapacityEngine.MEMORY_RESERVE)
    print("=" * 112)
    print("LOCAL LLM CAPACITY REPORT")
    print("=" * 112)
    print(f"OS / Machine      : {hardware['os']} / {hardware['machine']}")
    print(f"CPU Cores         : {hardware['cpu_cores']}")
    print(f"System RAM        : {hardware['ram_gb']:.2f} GB")
    print(f"GPU               : {hardware['gpu_name']}")
    print(f"GPU Count         : {hardware['gpu_count']}")
    print(f"Memory Type       : {hardware['memory_kind']}")
    print(f"Memory Pool       : {hardware['vram_gb']:.2f} GB")
    print(f"Model Budget      : {usable:.2f} GB")
    print(f"Quantization      : {quant_level}")
    print(f"Context            : {context_length:,} tokens")
    print(f"KV Cache           : FP16/BF16, {LLMCapacityEngine.KV_CACHE_BYTES:.1f} bytes/value")
    print(f"Runtime Overhead   : {LLMCapacityEngine.RUNTIME_OVERHEAD * 100:.0f}%")
    print(f"Memory Reserve     : {LLMCapacityEngine.MEMORY_RESERVE * 100:.0f}%")
    print("=" * 112)

    for category, rows in results.items():
        print(f"\n--- {category.upper()} ---")
        print(f"{'Model':<38} | {'Params':>8} | {'Architecture':<20} | {'VRAM/model':>12} | {'Concurrent':>12}")
        print("-" * 112)
        for r in rows:
            status = f"{r['max_concurrent']}x" if r["max_concurrent"] else "Insufficient"
            print(
                f"{r['model_name']:<38} | {r['params_b']:>7.2f}B | "
                f"{r['arch_type']:<20} | {r['vram_per_instance_gb']:>9.2f} GB | {status:>12}"
            )

    print("\n--- TWO-MODEL COMBINATIONS (DIFFERENT SIZE CATEGORIES) ---")
    if not combinations:
        print("No two-model combination fits within the usable memory budget at this quantization/context length.")
    else:
        print(f"{'Model A':<34} | {'Size':<14} | {'GB':>8} | {'Model B':<34} | {'Size':<14} | {'GB':>8} | {'Total':>8} | {'Free':>8}")
        print("-" * 142)
        for c in combinations:
            print(
                f"{c['model_a']:<34} | {c['category_a']:<14} | {c['vram_a_gb']:>6.2f} | "
                f"{c['model_b']:<34} | {c['category_b']:<14} | {c['vram_b_gb']:>6.2f} | "
                f"{c['combined_vram_gb']:>6.2f} | {c['remaining_gb']:>6.2f}"
            )
        print(f"\nCompatible combinations found: {len(combinations)}")

    print("\n--- SPECIAL / NOT SAFE TO ESTIMATE WITH STANDARD KV MATH ---")
    for name, reason in LLMCapacityEngine.SPECIAL_MODEL_CATALOG:
        print(f"{name:<30} : {reason}")


if __name__ == "__main__":
    LLMCapacityEngine.validate_catalog()
    hardware = HardwareDetector.get_system_hardware()

    QUANT_PRECISION = os.getenv("LLM_QUANT", "Q4_K_M")
    CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT", "4096"))

    results = LLMCapacityEngine.evaluate(
        hardware,
        quant_level=QUANT_PRECISION,
        context_length=CONTEXT_WINDOW,
    )
    combinations = LLMCapacityEngine.find_two_model_combinations(
        hardware,
        quant_level=QUANT_PRECISION,
        context_length=CONTEXT_WINDOW,
    )
    print_report(hardware, results, combinations, QUANT_PRECISION, CONTEXT_WINDOW)
