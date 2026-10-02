"""
llm_judge.py -- Independent MoE Judge logic
Evaluates interview transcripts post-hoc to avoid runtime overhead.
"""

import json
import time
from typing import Optional
from copy import deepcopy

from inference import ChatRunner

METRICS = [
    "turn_level_completeness_efficiency",
    "turn_level_contradiction_rate",
    "overall_elicit_efficiency",
    "participant_engagement",
    "redundancy_penalty",
    "hallucination_penalty"
]

def _scrub_json(raw: str) -> str:
    """Extract raw JSON substring so json.loads() can succeed."""
    raw = raw.strip()
    # Find start and end brackets
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start:end+1]
    return raw

def sanitize_trace(trace: dict) -> list[dict]:
    """Removes all candidate model metadata, timing, and latency info.
    Only returns a simple text transcript."""
    transcript = []
    for turn in trace.get("turns", []):
        if turn["role"] in ("interviewer", "participant"):
            transcript.append({
                "role": turn["role"],
                "text": turn["text"]
            })
    return transcript

def format_transcript_text(transcript: list[dict]) -> str:
    lines = []
    for t in transcript:
        role = "Interviewer" if t["role"] == "interviewer" else "Participant"
        lines.append(f"{role}: {t['text']}")
    return "\n\n".join(lines)


class LLMJudge:
    def __init__(self, judge_id: str, runner: ChatRunner):
        self.judge_id = judge_id
        self.runner = runner

    def evaluate_interview(self, trace: dict) -> dict:
        """
        Evaluate a completed interview trace.
        Returns a dictionary mapping metric names to scores [0-10].
        """
        start_time = time.time()
        
        # 1. Sanitize to ensure blinded evaluation
        clean_transcript = sanitize_trace(trace)
        transcript_text = format_transcript_text(clean_transcript)
        
        system_prompt = (
            "You are an expert qualitative research evaluator. "
            "Review the provided interview transcript and score the interviewer's performance on a scale of 0 to 10 for key metrics.\\n\\n"
            "Return ONLY a JSON dictionary with these exact keys (float values 0-10):\\n"
            '{"turn_level_completeness_efficiency": 8.0, "turn_level_contradiction_rate": 0.0, '
            '"overall_elicit_efficiency": 7.5, "participant_engagement": 8.5, '
            '"redundancy_penalty": 1.0, "hallucination_penalty": 0.0}\\n\\n'
            "CRITICALLY: Do not use markdown, markdown code blocks, or explain your reasoning. Just return the JSON object."
        )
        
        user_msg = f"Transcript:\\n{transcript_text}\\n\\nProvide JSON scores:"
        
        metrics = {m: 5.0 for m in METRICS} # default mid scores
        parse_success = False
        error_msg = None
        
        try:
            resp = self.runner.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=200,
                temperature=0.1
            )
            raw_text = resp.get("text", "")
            json_text = _scrub_json(raw_text)
            parsed = json.loads(json_text)
            
            # Map values back, fallback to 5.0 for missing
            for m in METRICS:
                val = parsed.get(m, 5.0)
                try:
                    metrics[m] = float(val)
                except ValueError:
                    metrics[m] = 5.0
            parse_success = True
        except Exception as e:
            error_msg = f"Judge parsing failed: {e}"
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return {
            "judge_id": self.judge_id,
            "metrics": metrics,
            "success": parse_success,
            "error": error_msg,
            "latency_ms": latency_ms
        }
