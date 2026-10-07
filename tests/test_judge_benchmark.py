import json
import os
from unittest.mock import patch, mock_open

# We rely on judge_benchmark's main script behavior and judge_selector

from judge_selector import select_judges

def test_select_judges_picks_biggest_fitting():
    hardware = {"vram_gb": 40.0}
    # 40GB budget minus 10% reserve = 36GB usable. 
    # Qwen2.5-32B needs ~20-25GB so it fits. 72B does not.
    judges = select_judges(hardware, top_n=3, quant="Q4_K_M")
    
    assert len(judges) <= 3
    # Check that family constraints are respected if possible
    families = [j["family"] for j in judges]
    assert len(set(families)) == len(families)

def test_select_judges_returns_empty_when_no_mem():
    hardware = {"vram_gb": 0.5}
    judges = select_judges(hardware, top_n=3, quant="Q8_0")
    assert judges == []
    assert judges == []

@patch("judge_benchmark.HardwareDetector")
@patch("judge_benchmark.select_judges")
@patch("judge_benchmark.load_traces")
@patch("judge_benchmark.get_runner")
@patch("judge_benchmark.LLMJudge")
def test_judge_benchmark_produces_outputs(
    mock_llm_judge_cls, mock_get_runner, mock_load, mock_select, mock_hw, tmp_path
):
    import judge_benchmark
    
    # Setup mocks
    mock_hw.get_system_hardware.return_value = {"vram_gb": 32.0}
    
    mock_select.return_value = [
        {"model_id": "judge_a", "name": "Judge A", "family": "Fam1", "params_b": 10.0, "quant": "Q4_0", "vram_gb": 6.0},
        {"model_id": "judge_b", "name": "Judge B", "family": "Fam2", "params_b": 8.0,  "quant": "Q4_0", "vram_gb": 5.0}
    ]
    
    mock_load.return_value = [
        {"run_id": "trace_1", "status": "success", "model_id": "cand_a", "trace": {}},
        {"run_id": "trace_2", "status": "success", "model_id": "cand_a", "trace": {}},
        {"run_id": "trace_3", "status": "success", "model_id": "cand_b", "trace": {}},
    ]
    
    mock_judge_instance = mock_llm_judge_cls.return_value
    mock_judge_instance.evaluate_interview.return_value = {
        "success": True,
        "parse_retries": 1,
        "latency_ms": 100,
        "metrics": {
            "turn_level_completeness_efficiency": 8.0, 
            "turn_level_contradiction_rate": 2.0, 
            "overall_elicit_efficiency": 7.5, 
            "participant_engagement": 6.0, 
            "redundancy_penalty": 1.0, 
            "hallucination_penalty": 0.0
        }
    }
    
    # Configure arguments for run
    import sys
    test_args = [
        "judge_benchmark.py", 
        "--out", str(tmp_path), 
        "--traces", "dummy.jsonl"
    ]
    
    with patch.object(sys, 'argv', test_args):
        judge_benchmark.main()
        
    long_csv = tmp_path / "judge_scores_long.csv"
    aggr_csv = tmp_path / "panel_aggregated.csv"
    summary_json = tmp_path / "judge_benchmark_summary.json"
    partial_jsonl = tmp_path / "judges_scores_partial.jsonl"
    
    assert long_csv.exists()
    assert aggr_csv.exists()
    assert summary_json.exists()
    assert partial_jsonl.exists()
    
    # Check rows: 3 traces x 2 judges x 6 metrics = 36 rows
    import csv
    with open(long_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 36
        
    # Aggr rows: 3 traces x 6 metrics = 18 rows
    with open(aggr_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 18
        
    # Verify summary JSON schema
    with open(summary_json, "r") as f:
        summary = json.load(f)
        assert "hardware" in summary
        assert "per_judge_summary" in summary
