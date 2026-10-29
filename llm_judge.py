"""
llm_judge.py -- Independent MoE Judge logic.

Evaluates interview transcripts post-hoc using a single structured metric:
    follow_up_question_quality  (ordinal integer 0 – 4)

Metric schema version: v2

The judge assesses observable transcript-level question quality only.
It does NOT claim to measure:
  - Actual human engagement (synthetic respondent is used)
  - Real-world factual correctness (judge lacks source facts)
  - Human-preference alignment

Strict validation:
  - Score must be a finite integer in {0, 1, 2, 3, 4}.
  - Missing key, non-integer value, out-of-range value, NaN, or infinity
    are ALL treated as invalid — score is set to None with a failure reason.
  - No midpoint defaults are applied for missing or malformed responses.
"""

import json
import math
import time
from typing import Optional
from copy import deepcopy

from inference import ChatRunner

# ─── METRIC DEFINITION ────────────────────────────────────────────────────────

METRIC = "follow_up_question_quality"
METRIC_VERSION = "v2"
VALID_SCORE_RANGE = (0, 4)   # inclusive, integer
JUDGE_SCHEMA_VERSION = "v2"

RUBRIC = """\
Score the OVERALL follow-up-question quality for this transcript on a 0–4 ordinal scale:

  0 — Unacceptable: Most follow-ups are irrelevant, invalid, or incoherent with the transcript.
  1 — Poor: Most follow-ups are generic, poorly connected to prior responses, or assume facts not in the transcript.
  2 — Mixed: Some follow-ups are relevant, but targeting, specificity, or conversational progression is inconsistent.
  3 — Good: Follow-ups are generally clear, relevant, specific, and appropriately connected to prior responses.
  4 — Excellent: Follow-ups are consistently clear, specific, answerable, non-leading, and coherent, advancing a useful line of inquiry.

Return ONLY a JSON object with exactly these two keys:
  "follow_up_question_quality": <integer 0, 1, 2, 3, or 4>
  "supporting_evidence": "<one to three sentences citing specific transcript turns>"

Do NOT include markdown, code fences, prose outside the JSON, or any additional keys.
"""

# ─── LEGACY METRIC LIST (kept for backward reference only — NOT active) ───────

_LEGACY_METRICS = [
    "turn_level_completeness_efficiency",
    "turn_level_contradiction_rate",
    "overall_elicit_efficiency",
    "participant_engagement",
    "redundancy_penalty",
    "hallucination_penalty",
]

# ─── FAILURE TYPE ENUM ────────────────────────────────────────────────────────

FAILURE_INFERENCE = "inference_failure"   # runner.generate() raised an exception
FAILURE_PARSE = "parse_failure"           # response not valid JSON / missing key
FAILURE_INVALID = "invalid_output"        # score present but out of range / non-integer


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _scrub_json(raw: str) -> str:
    """Extract the first balanced JSON object from raw text."""
    raw = raw.strip()
    start = raw.find("{")
    if start == -1:
        return raw
    depth = 0
    in_str = False
    escape_next = False
    for i, ch in enumerate(raw[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_str:
            escape_next = True
            continue
        if ch == '"':
            in_str = not in_str
        if not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return raw[start: i + 1]
    return raw[start:]


def sanitize_trace(trace: dict) -> list[dict]:
    """Strip all candidate model metadata, timing, and extraction data.

    Returns a minimal transcript with only role and text fields.
    Candidate model identifiers must not appear in the judge's input.
    """
    transcript = []
    for turn in trace.get("turns", []):
        if turn["role"] in ("interviewer", "participant"):
            transcript.append({
                "role": turn["role"],
                "text": turn["text"],
            })
    return transcript


def format_transcript_text(transcript: list[dict]) -> str:
    lines = []
    for t in transcript:
        role = "Interviewer" if t["role"] == "interviewer" else "Participant"
        lines.append(f"{role}: {t['text']}")
    return "\n\n".join(lines)


def _validate_score(parsed: dict) -> tuple[Optional[int], Optional[str], Optional[str]]:
    """Validate the parsed judge response.

    Returns (score, failure_type, failure_reason) where:
      - score is an int in {0,1,2,3,4} if valid, else None
      - failure_type is one of FAILURE_PARSE / FAILURE_INVALID / None
      - failure_reason is a human-readable explanation or None
    """
    key = METRIC
    if key not in parsed:
        return None, FAILURE_PARSE, f"Required key '{key}' missing from response"

    raw_val = parsed[key]

    # Must be numeric
    try:
        float_val = float(raw_val)
    except (TypeError, ValueError):
        return None, FAILURE_INVALID, f"Score value '{raw_val}' is not numeric"

    # Must be finite
    if not math.isfinite(float_val):
        return None, FAILURE_INVALID, f"Score value '{raw_val}' is not finite (NaN or Inf)"

    # Must be a whole number
    if float_val != int(float_val):
        return None, FAILURE_INVALID, (
            f"Score {float_val} is not an integer — ordinal scale requires integer values"
        )

    score = int(float_val)
    lo, hi = VALID_SCORE_RANGE
    if not (lo <= score <= hi):
        return None, FAILURE_INVALID, (
            f"Score {score} is outside valid range [{lo}, {hi}]"
        )

    return score, None, None


# ─── JUDGE CLASS ──────────────────────────────────────────────────────────────

class LLMJudge:
    def __init__(self, judge_id: str, runner: ChatRunner):
        self.judge_id = judge_id
        self.runner = runner

    def evaluate_interview(self, trace: dict) -> dict:
        """Evaluate a completed interview trace.

        Returns a dict with:
            judge_id         : str
            metric_version   : str  (METRIC_VERSION)
            score            : int or None
            supporting_evidence : str or None
            success          : bool
            failure_type     : str or None (FAILURE_INFERENCE / FAILURE_PARSE / FAILURE_INVALID)
            failure_reason   : str or None
            latency_ms       : int
        """
        start_time = time.time()

        clean_transcript = sanitize_trace(trace)
        transcript_text = format_transcript_text(clean_transcript)

        system_prompt = (
            "You are an expert qualitative research evaluator assessing AI interviewer transcripts.\n\n"
            + RUBRIC
        )
        user_msg = f"Transcript:\n\n{transcript_text}\n\nProvide your assessment as JSON:"

        score = None
        supporting_evidence = None
        success = False
        failure_type = None
        failure_reason = None
        raw_response = ""

        try:
            resp = self.runner.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=256,
                temperature=0.1,
            )
            raw_response = resp.get("text", "")

            json_text = _scrub_json(raw_response)
            try:
                parsed = json.loads(json_text)
            except (json.JSONDecodeError, ValueError) as exc:
                failure_type = FAILURE_PARSE
                failure_reason = f"JSON parse error: {exc}"
            else:
                score, failure_type, failure_reason = _validate_score(parsed)
                if score is not None:
                    success = True
                    supporting_evidence = str(parsed.get("supporting_evidence", ""))[:500]

        except Exception as exc:
            failure_type = FAILURE_INFERENCE
            failure_reason = f"Inference error: {exc}"

        latency_ms = int((time.time() - start_time) * 1000)

        return {
            "judge_id": self.judge_id,
            "metric_version": METRIC_VERSION,
            "score": score,
            "supporting_evidence": supporting_evidence,
            "success": success,
            "failure_type": failure_type,
            "failure_reason": failure_reason,
            "latency_ms": latency_ms,
        }
