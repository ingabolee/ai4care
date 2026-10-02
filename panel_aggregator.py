"""
panel_aggregator.py -- Evaluation MoE Aggregator
Handles parallel or sequential evaluation by a panel of judges
and computes central tendencies and disagreement statistics.
"""

import statistics
from typing import List, Dict
from llm_judge import LLMJudge, METRICS

class PanelAggregator:
    def __init__(self, judges: List[LLMJudge]):
        self.judges = judges

    def evaluate_with_panel(self, trace: dict) -> dict:
        """
        Sends the interview trace to all judges, aggregates the metrics.
        Never substitutes zeros for missing values, uses median for robustness.
        """
        results = []
        for judge in self.judges:
            # Note: Can be multi-processed, but done sequentially for simplicity here in PoC.
            res = judge.evaluate_interview(trace)
            results.append(res)
            
        final_metrics = {}
        disagreements = {}
        
        for m in METRICS:
            valid_scores = []
            for r in results:
                if r.get("success", False) and m in r.get("metrics", {}):
                    valid_scores.append(r["metrics"][m])
            
            if len(valid_scores) > 0:
                final_metrics[m] = statistics.median(valid_scores)
                if len(valid_scores) > 1:
                    disagreements[m] = statistics.stdev(valid_scores)
                else:
                    disagreements[m] = 0.0
            else:
                # Fallback purely if all judges fail
                final_metrics[m] = None
                disagreements[m] = 0.0

        return {
            "aggregated_metrics": final_metrics,
            "metric_stdev": disagreements,
            "judge_responses": results,
            "successful_judges": sum(1 for r in results if r.get("success", False)),
            "total_judges": len(self.judges)
        }
