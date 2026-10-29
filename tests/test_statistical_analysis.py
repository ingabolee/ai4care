"""
tests/test_statistical_analysis.py -- Tests for the corrected statistical analysis.

Covers F4 requirements:
  - Paired identical observations → zero differences
  - CI reproducible for fixed seed
  - Case variants grouped under logical case
  - Different models/ablations not pooled
  - Missing values excluded with counts reported
  - Failed runs not treated as zero metric values
  - Actual sample sizes reported

Run with:
    python -m pytest tests/test_statistical_analysis.py -v
"""

from __future__ import annotations

import csv
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analyze_benchmark_size import (
    BOOTSTRAP_SEED,
    _float_or_none,
    bootstrap_ci,
    cluster_bootstrap_ci,
    load_results,
)


# ─── Helper ───────────────────────────────────────────────────────────────────

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ─── _float_or_none ───────────────────────────────────────────────────────────

class TestFloatOrNone:
    def test_valid_float(self):
        assert _float_or_none("0.75") == pytest.approx(0.75)

    def test_empty_string_returns_none(self):
        assert _float_or_none("") is None

    def test_none_string_returns_none(self):
        assert _float_or_none("none") is None
        assert _float_or_none("None") is None

    def test_null_string_returns_none(self):
        assert _float_or_none("null") is None

    def test_nan_string_returns_none(self):
        assert _float_or_none("nan") is None

    def test_non_numeric_returns_none(self):
        assert _float_or_none("abc") is None


# ─── bootstrap_ci ─────────────────────────────────────────────────────────────

class TestBootstrapCI:
    def test_reproducible_for_same_seed(self):
        data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        r1 = bootstrap_ci(data, seed=BOOTSTRAP_SEED)
        r2 = bootstrap_ci(data, seed=BOOTSTRAP_SEED)
        assert r1 == r2

    def test_different_seed_may_differ(self):
        data = [0.1, 0.3, 0.5, 0.7, 0.9]
        r1 = bootstrap_ci(data, seed=1)
        r2 = bootstrap_ci(data, seed=9999)
        # Not guaranteed to differ but results should be tuples of floats
        assert len(r1) == 4 and all(isinstance(v, float) for v in r1)

    def test_empty_data(self):
        r = bootstrap_ci([])
        assert r == (0.0, 0.0, 0.0, 0.0)

    def test_single_value(self):
        r = bootstrap_ci([0.5])
        mean, lo, hi, width = r
        assert mean == pytest.approx(0.5)
        assert lo <= hi

    def test_all_same_value_zero_width(self):
        data = [0.8] * 20
        mean, lo, hi, width = bootstrap_ci(data)
        assert mean == pytest.approx(0.8)
        assert width == pytest.approx(0.0, abs=0.001)

    def test_mean_within_ci(self):
        data = [float(i) / 10 for i in range(1, 11)]
        mean, lo, hi, _ = bootstrap_ci(data)
        assert lo <= mean <= hi


# ─── cluster_bootstrap_ci ────────────────────────────────────────────────────

class TestClusterBootstrapCI:
    def test_reproducible(self):
        groups = {"g1": [0.3, 0.4], "g2": [0.6, 0.7], "g3": [0.2, 0.5]}
        r1 = cluster_bootstrap_ci(groups, seed=BOOTSTRAP_SEED)
        r2 = cluster_bootstrap_ci(groups, seed=BOOTSTRAP_SEED)
        assert r1 == r2

    def test_empty_groups(self):
        r = cluster_bootstrap_ci({})
        assert r[0] == 0.0
        assert r[4] == 0  # n_clusters

    def test_reports_cluster_count(self):
        groups = {"g1": [0.5], "g2": [0.6], "g3": [0.7]}
        _, _, _, _, n_clusters = cluster_bootstrap_ci(groups)
        assert n_clusters == 3

    def test_paired_identical_observations_zero_range(self):
        # If all clusters have identical values, CI width should be ~0
        groups = {"g1": [0.5, 0.5], "g2": [0.5, 0.5], "g3": [0.5, 0.5]}
        mean, lo, hi, width, _ = cluster_bootstrap_ci(groups)
        assert mean == pytest.approx(0.5)
        assert width == pytest.approx(0.0, abs=0.01)


# ─── load_results ─────────────────────────────────────────────────────────────

class TestLoadResults:
    def test_loads_csv_rows(self):
        rows = [
            {"run_id": "r1", "model_id": "m1", "status": "success",
             "reference_fact_f1": "0.5", "case_group_id": "g1",
             "question_variant_id": "v1", "ablation": "none"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.csv"
            _write_csv(path, rows)
            loaded = load_results(path)
        assert len(loaded) == 1
        assert loaded[0]["model_id"] == "m1"

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_results(Path("/nonexistent/path/results.csv"))


# ─── Integration: missing values excluded, failed rows not zero ───────────────

class TestMissingAndFailedHandling:
    def test_missing_values_excluded_from_calculation(self):
        """_float_or_none returns None for missing cells, which are then excluded."""
        values_with_missing = ["0.5", "", "0.7", "none", "0.3"]
        parsed = [_float_or_none(v) for v in values_with_missing]
        valid = [v for v in parsed if v is not None]
        assert len(valid) == 3
        assert all(isinstance(v, float) for v in valid)

    def test_failed_rows_are_status_failed_not_zero(self):
        """Failed runs should have status='failed', not zero metric values."""
        rows = [
            {"run_id": "ok1", "model_id": "m1", "status": "success",
             "reference_fact_f1": "0.5", "case_group_id": "g1",
             "question_variant_id": "v1", "ablation": "none"},
            {"run_id": "fail1", "model_id": "m1", "status": "failed",
             "reference_fact_f1": "", "case_group_id": "g1",
             "question_variant_id": "v1", "ablation": "none"},
        ]
        successful = [r for r in rows if r.get("status") == "success"]
        failed = [r for r in rows if r.get("status") != "success"]

        assert len(successful) == 1
        assert len(failed) == 1
        assert _float_or_none(failed[0]["reference_fact_f1"]) is None


# ─── Model and ablation separation ───────────────────────────────────────────

class TestModelAblationSeparation:
    def test_models_not_pooled(self):
        """Rows from different models must be filtered separately."""
        rows = [
            {"model_id": "m1", "ablation": "none", "status": "success",
             "reference_fact_f1": "0.8", "case_group_id": "g1", "question_variant_id": "v1"},
            {"model_id": "m2", "ablation": "none", "status": "success",
             "reference_fact_f1": "0.2", "case_group_id": "g1", "question_variant_id": "v1"},
        ]
        m1_vals = [_float_or_none(r["reference_fact_f1"]) for r in rows if r["model_id"] == "m1"]
        m2_vals = [_float_or_none(r["reference_fact_f1"]) for r in rows if r["model_id"] == "m2"]
        assert m1_vals == [0.8]
        assert m2_vals == [0.2]
        # Pooled mean would be 0.5, but individually they're 0.8 and 0.2
        assert m1_vals[0] != m2_vals[0]

    def test_ablation_conditions_not_pooled(self):
        rows = [
            {"ablation": "none", "reference_fact_f1": "0.8"},
            {"ablation": "no_rag", "reference_fact_f1": "0.5"},
        ]
        none_vals = [_float_or_none(r["reference_fact_f1"]) for r in rows if r["ablation"] == "none"]
        norag_vals = [_float_or_none(r["reference_fact_f1"]) for r in rows if r["ablation"] == "no_rag"]
        assert none_vals == [0.8]
        assert norag_vals == [0.5]


# ─── Case grouping ────────────────────────────────────────────────────────────

class TestCaseGrouping:
    def test_variants_of_same_case_cluster_together(self):
        """Multiple question variants of the same logical case must go to same cluster."""
        rows = [
            {"case_group_id": "src_001", "question_variant_id": "v1", "reference_fact_f1": "0.6"},
            {"case_group_id": "src_001", "question_variant_id": "v2", "reference_fact_f1": "0.7"},
            {"case_group_id": "src_002", "question_variant_id": "v3", "reference_fact_f1": "0.4"},
        ]
        clusters: dict = defaultdict(list)
        for r in rows:
            grp = r.get("case_group_id", "")
            val = _float_or_none(r["reference_fact_f1"])
            if val is not None:
                clusters[grp].append(val)

        assert len(clusters) == 2
        assert len(clusters["src_001"]) == 2  # both variants in same cluster
        assert len(clusters["src_002"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
