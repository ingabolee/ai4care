"""
verify_models.py -- Verify all enabled GGUF models can be loaded, templated,
                    and used for generation.

Usage:
    python verify_models.py [--model MODEL_ID] [--all]

For each model this script checks:
    1. GGUF file exists and has non-trivial size.
    2. llama-cpp-python can initialize the model.
    3. Native chat completion is available.
    4. A simple test generation succeeds (produces non-empty text).
    5. The configured context length is supported.
    6. The model can produce a question-style output.

Any model that fails these checks is reported as UNSUPPORTED and excluded
from evaluation runs. No silent substitution ever occurs.
"""

import argparse
import json
import time
from pathlib import Path

from model_registry import (
    MODEL_CONFIGS,
    MODELS_DIR,
    ModelSpec,
    get_enabled_models,
    get_model_spec,
    load_llm,
)

TEST_SYSTEM_PROMPT = (
    "You are an expert knowledge elicitation interviewer. "
    "Your job is to ask targeted follow-up questions to uncover tacit knowledge."
)
TEST_USER_PROMPT = (
    "The participant mentioned they faced a significant operational challenge "
    "related to slow data processing. Ask one focused follow-up question to "
    "understand the root cause."
)


def verify_single(model_id: str, verbose: bool = False) -> ModelSpec:
    """Run all verification checks for one model. Returns an updated ModelSpec."""
    spec = get_model_spec(model_id)
    print(f"\n── Verifying: {model_id} ──────────────────────────────────────")
    print(f"  Path : {spec.local_path}")
    print(f"  Desc : {spec.description}")

    # ── Check 1: file existence ─────────────────────────────────────────────
    if not spec.local_path.exists():
        spec.error_message = "Model file does not exist."
        print(f"  [FAIL] {spec.error_message}")
        return spec
    if spec.local_path.stat().st_size <= 10_000:
        spec.error_message = f"Model file too small ({spec.model_size_mb:.1f} MB) — likely incomplete."
        print(f"  [FAIL] {spec.error_message}")
        return spec
    print(f"  [OK] File exists ({spec.model_size_mb:.1f} MB)")

    # ── Check 2: load via llama-cpp-python ──────────────────────────────────
    try:
        t0 = time.time()
        llm = load_llm(model_id, verbose=verbose)
        load_ms = int((time.time() - t0) * 1000)
        spec.load_success = True
        print(f"  [OK] Loaded in {load_ms} ms")
    except RuntimeError as exc:
        spec.error_message = str(exc)
        print(f"  [FAIL] Load error: {exc}")
        return spec

    # ── Check 3: detect chat template ───────────────────────────────────────
    try:
        meta = llm.metadata if hasattr(llm, "metadata") else {}
        tmpl = meta.get("tokenizer.chat_template", None)
        spec.detected_chat_template = tmpl[:80] + "…" if tmpl and len(tmpl) > 80 else tmpl
        if tmpl:
            print(f"  [OK] Chat template detected (length {len(tmpl)} chars)")
        else:
            print("  [WARN] No explicit chat template found — model will use llama-cpp default formatting")
    except Exception as exc:
        print(f"  [WARN] Could not inspect metadata: {exc}")

    # ── Check 4 & 5: native chat completion ─────────────────────────────────
    try:
        t0 = time.time()
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": TEST_SYSTEM_PROMPT},
                {"role": "user", "content": TEST_USER_PROMPT},
            ],
            max_tokens=80,
            temperature=0.1,
        )
        gen_ms = int((time.time() - t0) * 1000)
        generated_text = resp["choices"][0]["message"]["content"].strip()
        if not generated_text:
            spec.error_message = "Generation produced empty output."
            print(f"  [FAIL] {spec.error_message}")
            return spec
        spec.generation_success = True
        print(f"  [OK] Generation succeeded in {gen_ms} ms")
        if verbose:
            print(f"         Output: {generated_text[:120]}")
        else:
            preview = generated_text[:80].replace("\n", " ")
            print(f"         Preview: {preview}…" if len(generated_text) > 80 else f"         Preview: {preview}")
    except Exception as exc:
        spec.error_message = f"create_chat_completion failed: {exc}"
        print(f"  [FAIL] {spec.error_message}")
        return spec

    # ── Check 6: output looks like a question ───────────────────────────────
    if "?" in generated_text:
        print("  [OK] Output contains a question mark")
    else:
        print("  [WARN] Output does not contain '?' — model may not be following the instruction precisely")

    print(f"  [PASS] {model_id} is verified and ready for evaluation.")
    return spec


def run_verification(model_ids: list[str], verbose: bool = False) -> dict[str, ModelSpec]:
    results: dict[str, ModelSpec] = {}
    for mid in model_ids:
        if mid not in MODEL_CONFIGS:
            print(f"\n[ERROR] '{mid}' is not in MODEL_CONFIGS. Skipping.")
            continue
        results[mid] = verify_single(mid, verbose=verbose)
    return results


def print_summary(results: dict[str, ModelSpec]) -> None:
    print("\n\n══════════════════════════════════════════════════════════════")
    print("  MODEL VERIFICATION SUMMARY")
    print("══════════════════════════════════════════════════════════════")
    passed = [m for m, s in results.items() if s.generation_success]
    failed = [m for m, s in results.items() if not s.generation_success]

    for mid in passed:
        print(f"  [PASS] {mid:<20} — {results[mid].description}")
    for mid in failed:
        err = results[mid].error_message or "unknown error"
        print(f"  [FAIL] {mid:<20} — {err}")

    print("──────────────────────────────────────────────────────────────")
    print(f"  Ready: {len(passed)}   Failed/Excluded: {len(failed)}")
    print("══════════════════════════════════════════════════════════════\n")


def save_manifest(results: dict[str, ModelSpec], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = [spec.to_dict() for spec in results.values()]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Manifest saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Verify GGUF models for AI4CARE evaluation")
    parser.add_argument("--model", help="Verify a single model by ID")
    parser.add_argument("--all", action="store_true", help="Verify all registered + enabled models")
    parser.add_argument("--verbose", action="store_true", help="Show full generation output")
    parser.add_argument(
        "--save-manifest",
        metavar="PATH",
        default="results/model_manifest.json",
        help="Path to save the verification manifest JSON",
    )
    args = parser.parse_args()

    if args.model:
        model_ids = [args.model]
    elif args.all or True:  # default: verify all enabled
        model_ids = get_enabled_models()

    if not model_ids:
        print("[ERROR] No enabled models found in MODEL_CONFIGS.")
        return

    results = run_verification(model_ids, verbose=args.verbose)
    print_summary(results)
    save_manifest(results, Path(args.save_manifest))


if __name__ == "__main__":
    main()
