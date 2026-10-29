"""
tests/test_provenance.py -- Tests for run identity, resumption, and provenance.

Covers F2 requirements:
  - Stable unique run_ids
  - No collision between variants of same case
  - No collision between different cases
  - Ablation separation
  - Config change invalidation
  - Compatible resume works
  - Duplicate ID detection
  - Manifest preservation

Run with:
    python -m pytest tests/test_provenance.py -v
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate import (
    ExperimentConfig,
    derive_run_seed,
    load_traces,
    make_run_id,
    create_iteration_manifest,
)


# ─── make_run_id ─────────────────────────────────────────────────────────────

class TestMakeRunId:
    BASE_KWARGS = dict(
        iteration_id="0001",
        model_id="general_small",
        case_id="case_abc1234567",
        question_variant_id="qv_abc123",
        ablation="none",
        config_fingerprint="fp_" + "a" * 60,
    )

    def test_deterministic(self):
        id1 = make_run_id(**self.BASE_KWARGS)
        id2 = make_run_id(**self.BASE_KWARGS)
        assert id1 == id2

    def test_different_variants_no_collision(self):
        id1 = make_run_id(**{**self.BASE_KWARGS, "question_variant_id": "qv_aaa"})
        id2 = make_run_id(**{**self.BASE_KWARGS, "question_variant_id": "qv_bbb"})
        assert id1 != id2

    def test_different_full_case_ids_no_collision(self):
        # Two distinct case IDs that share the same 8-character prefix (the old bug)
        id1 = make_run_id(**{**self.BASE_KWARGS, "case_id": "case_abcdef00_x"})
        id2 = make_run_id(**{**self.BASE_KWARGS, "case_id": "case_abcdef00_y"})
        assert id1 != id2

    def test_different_ablation_no_collision(self):
        id_none = make_run_id(**{**self.BASE_KWARGS, "ablation": "none"})
        id_rag = make_run_id(**{**self.BASE_KWARGS, "ablation": "no_rag"})
        id_sm = make_run_id(**{**self.BASE_KWARGS, "ablation": "state_machine"})
        assert len({id_none, id_rag, id_sm}) == 3

    def test_different_model_no_collision(self):
        id_a = make_run_id(**{**self.BASE_KWARGS, "model_id": "general_small"})
        id_b = make_run_id(**{**self.BASE_KWARGS, "model_id": "general_main"})
        assert id_a != id_b

    def test_different_config_fingerprint_no_collision(self):
        id_a = make_run_id(**{**self.BASE_KWARGS, "config_fingerprint": "fp_" + "a" * 60})
        id_b = make_run_id(**{**self.BASE_KWARGS, "config_fingerprint": "fp_" + "b" * 60})
        assert id_a != id_b

    def test_output_is_hex_string(self):
        run_id = make_run_id(**self.BASE_KWARGS)
        assert all(c in "0123456789abcdef" for c in run_id)

    def test_output_length_is_16(self):
        run_id = make_run_id(**self.BASE_KWARGS)
        assert len(run_id) == 16


# ─── ExperimentConfig fingerprint ────────────────────────────────────────────

class TestConfigFingerprint:
    def test_same_config_same_fingerprint(self):
        c1 = ExperimentConfig(random_seed=42, ablation="none", model_ids=["general_small"])
        c2 = ExperimentConfig(random_seed=42, ablation="none", model_ids=["general_small"])
        assert c1.fingerprint() == c2.fingerprint()

    def test_different_ablation_different_fingerprint(self):
        c1 = ExperimentConfig(ablation="none")
        c2 = ExperimentConfig(ablation="no_rag")
        assert c1.fingerprint() != c2.fingerprint()

    def test_different_seed_different_fingerprint(self):
        c1 = ExperimentConfig(random_seed=42)
        c2 = ExperimentConfig(random_seed=99)
        assert c1.fingerprint() != c2.fingerprint()

    def test_different_model_list_different_fingerprint(self):
        c1 = ExperimentConfig(model_ids=["general_small"])
        c2 = ExperimentConfig(model_ids=["general_main"])
        assert c1.fingerprint() != c2.fingerprint()

    def test_fingerprint_is_sha256_hex(self):
        fp = ExperimentConfig().fingerprint()
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

    def test_judge_ids_excluded_from_fingerprint(self):
        # Changing judge IDs should NOT change the generation fingerprint
        c1 = ExperimentConfig(judge_ids=["general_main"])
        c2 = ExperimentConfig(judge_ids=["coder_main"])
        assert c1.fingerprint() == c2.fingerprint()


# ─── derive_run_seed ──────────────────────────────────────────────────────────

class TestDeriveRunSeed:
    def test_deterministic(self):
        s1 = derive_run_seed(1234, "general_small", "case_abc", "qv_xyz")
        s2 = derive_run_seed(1234, "general_small", "case_abc", "qv_xyz")
        assert s1 == s2

    def test_different_variant_different_seed(self):
        s1 = derive_run_seed(1234, "general_small", "case_abc", "qv_aaa")
        s2 = derive_run_seed(1234, "general_small", "case_abc", "qv_bbb")
        assert s1 != s2

    def test_different_model_different_seed(self):
        s1 = derive_run_seed(1234, "general_small", "case_abc", "qv_aaa")
        s2 = derive_run_seed(1234, "general_main", "case_abc", "qv_aaa")
        assert s1 != s2

    def test_returns_int(self):
        s = derive_run_seed(42, "m", "c", "v")
        assert isinstance(s, int)


# ─── Trace loading with config filter ────────────────────────────────────────

class TestLoadTraces:
    def _write_trace(self, path: Path, rec: dict) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def test_compatible_trace_loaded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tp = Path(tmpdir) / "traces.jsonl"
            trace = {
                "run_id": "abc123",
                "model_id": "general_small",
                "status": "success",
                "config_fingerprint": "fp_aaa",
                "benchmark_hash": "bh_aaa",
            }
            self._write_trace(tp, trace)

            import evaluate
            orig = evaluate.TRACES_FILE
            evaluate.TRACES_FILE = tp
            result = load_traces("fp_aaa", "bh_aaa")
            evaluate.TRACES_FILE = orig

            assert "abc123" in result

    def test_incompatible_config_not_loaded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tp = Path(tmpdir) / "traces.jsonl"
            trace = {
                "run_id": "def456",
                "model_id": "general_small",
                "status": "success",
                "config_fingerprint": "fp_old",
                "benchmark_hash": "bh_aaa",
            }
            self._write_trace(tp, trace)

            import evaluate
            orig = evaluate.TRACES_FILE
            evaluate.TRACES_FILE = tp
            result = load_traces("fp_new", "bh_aaa")
            evaluate.TRACES_FILE = orig

            assert "def456" not in result

    def test_incompatible_benchmark_not_loaded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tp = Path(tmpdir) / "traces.jsonl"
            trace = {
                "run_id": "ghi789",
                "model_id": "general_small",
                "status": "success",
                "config_fingerprint": "fp_aaa",
                "benchmark_hash": "bh_old",
            }
            self._write_trace(tp, trace)

            import evaluate
            orig = evaluate.TRACES_FILE
            evaluate.TRACES_FILE = tp
            result = load_traces("fp_aaa", "bh_new")
            evaluate.TRACES_FILE = orig

            assert "ghi789" not in result

    def test_duplicate_run_id_does_not_silently_corrupt(self, capsys):
        """Duplicate run_ids with conflicting status should trigger a warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tp = Path(tmpdir) / "traces.jsonl"
            trace_a = {"run_id": "dup001", "model_id": "model_a", "status": "success",
                       "config_fingerprint": "fp_aaa", "benchmark_hash": "bh_aaa"}
            trace_b = {"run_id": "dup001", "model_id": "model_b", "status": "failed",
                       "config_fingerprint": "fp_aaa", "benchmark_hash": "bh_aaa"}
            self._write_trace(tp, trace_a)
            self._write_trace(tp, trace_b)

            import evaluate
            orig = evaluate.TRACES_FILE
            evaluate.TRACES_FILE = tp
            load_traces("fp_aaa", "bh_aaa")
            evaluate.TRACES_FILE = orig

            captured = capsys.readouterr()
            # A warning must have been printed
            assert "WARNING" in captured.out or "duplicate" in captured.out.lower()


# ─── Manifest provenance ──────────────────────────────────────────────────────

class TestManifestProvenance:
    def test_manifest_contains_benchmark_hash(self):
        from evaluate import create_iteration_manifest, ExperimentConfig
        cases = [
            {"case_id": f"case_{i:04d}", "case_group_id": f"grp_{i}",
             "question_variant_id": f"qv_{i}", "variant_key": "overall",
             "answer": "ans", "category": "technology"}
            for i in range(10)
        ]
        config = ExperimentConfig(cases_per_iteration=5, sampling_mode="proportional")

        with tempfile.TemporaryDirectory() as tmpdir:
            import evaluate
            orig = evaluate.ITERATIONS_DIR
            evaluate.ITERATIONS_DIR = Path(tmpdir)
            manifest = create_iteration_manifest("0001", cases, 5, 42, config, "bh_test123")
            evaluate.ITERATIONS_DIR = orig

        assert manifest["benchmark_hash"] == "bh_test123"
        assert manifest["config_fingerprint"] == config.fingerprint()

    def test_same_model_receives_same_cases(self):
        """All models in an iteration should use the same manifest case selection."""
        from evaluate import create_iteration_manifest, ExperimentConfig
        cases = [
            {"case_id": f"case_{i:04d}", "case_group_id": f"grp_{i}",
             "question_variant_id": f"qv_{i}", "variant_key": "overall",
             "answer": "ans", "category": "technology"}
            for i in range(20)
        ]
        config = ExperimentConfig(cases_per_iteration=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            import evaluate
            orig = evaluate.ITERATIONS_DIR
            evaluate.ITERATIONS_DIR = Path(tmpdir)
            m1 = create_iteration_manifest("0001", cases, 10, 99, config, "bh_xxx")
            m2 = create_iteration_manifest("0001", cases, 10, 99, config, "bh_xxx")
            evaluate.ITERATIONS_DIR = orig

        # Same seed produces same selection
        assert [c["case_id"] for c in m1["cases"]] == [c["case_id"] for c in m2["cases"]]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
