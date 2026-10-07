"""
judge_selector.py -- Automated hardware-aware LLM judge selection engine.
Detects hardware constraints and selects the best/biggest models to act as judges.
"""

import logging
from typing import Any, Dict, List, Optional

from estimator import HardwareDetector, LLMCapacityEngine

logger = logging.getLogger(__name__)


def slugify(name: str) -> str:
    """Convert a model name to a slugified model_id."""
    return name.lower().replace(" ", "-")


def select_judges(
    hardware: Dict[str, Any],
    top_n: int = 3,
    quant: str = "auto",
    context_length: int = 4096,
    families: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Select the top `top_n` biggest models that fit in the hardware memory budget,
    preferring distinct model families.
    """
    total_memory = hardware.get("vram_gb", 0.0)
    usable_gb = total_memory * (1 - LLMCapacityEngine.MEMORY_RESERVE)
    
    candidates = []
    
    # Ordered list of quantizations from best to worst
    auto_quants = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"]
    
    for m in LLMCapacityEngine.MODEL_CATALOG:
        if families and m["family"] not in families:
            continue
            
        chosen_quant = None
        needed_vram = None
        
        if quant == "auto":
            for q in auto_quants:
                try:
                    needed = LLMCapacityEngine.estimate_vram_gb(m, q, context_length)
                    if needed <= usable_gb:
                        chosen_quant = q
                        needed_vram = needed
                        break
                except ValueError:
                    continue  # Invalid context length bounds -> skip
        else:
            try:
                needed = LLMCapacityEngine.estimate_vram_gb(m, quant, context_length)
                if needed <= usable_gb:
                    chosen_quant = quant
                    needed_vram = needed
            except ValueError:
                pass
                
        if chosen_quant and needed_vram is not None:
            candidates.append({
                "model_id": slugify(m["name"]),
                "name": m["name"],
                "repo": m["repo"],
                "family": m["family"],
                "params_b": m["params_b"],
                "quant": chosen_quant,
                "vram_gb": needed_vram,
                "context_length": context_length
            })
            
    if not candidates:
        logger.warning(f"No models fit the current usable hardware memory budget of {usable_gb:.2f} GB "
                       f"(Total: {total_memory:.2f} GB).")
        return []
        
    # Rank survivors by params_b descending (biggest first)
    candidates.sort(key=lambda x: x["params_b"], reverse=True)
    
    # Select with family diversity constraint
    selected = []
    used_families = set()
    
    # First pass: try to respect family constraints
    for c in candidates:
        if len(selected) >= top_n:
            break
        if c["family"] not in used_families:
            selected.append(c)
            used_families.add(c["family"])
            
    # Relaxation pass: if we couldn't find enough distinct families, fill from the top remaining
    if len(selected) < top_n:
        for c in candidates:
            if len(selected) >= top_n:
                break
            if c not in selected:
                selected.append(c)
                
    return selected
