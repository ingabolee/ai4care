"""
engine.py -- Interview session logic, structured LLM extraction, evidence-based
             state machine, adaptive gap selection, and question generation.
"""

from __future__ import annotations

import json
import time
import uuid
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

from knowledge import query, store_atomic_fact, store_interview_artifact
from inference import ChatRunner

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
MAX_TURNS = 15
COMPLETION_THRESHOLD = 0.75  # fraction of fields at 'supported' or above
RECENT_TURNS_WINDOW = 4       # turns kept verbatim for LLM
REDUNDANCY_THRESHOLD = 0.85   # cosine similarity threshold for flagging redundant questions
RAG_TOP_K = 5

OPENING = (
    "Welcome to the AI4CARE Knowledge Elicitation interview. "
    "I am here to help document tacit knowledge from your experience. "
    "To start: could you describe a significant project, technical change, or operational "
    "challenge your team has faced recently, and what prompted it?"
)

# ─── KNOWLEDGE SCHEMA ────────────────────────────────────────────────────────
SCHEMA_FIELDS = [
    "domain_area",
    "primary_challenge",
    "root_cause",
    "key_decisions",
    "technologies_used",
    "people_involved",
    "processes_affected",
    "operational_impact",
    "lessons_learned",
]

# Status levels (higher index = stronger)
STATUS_ORDER = ["unknown", "candidate", "supported", "verified", "conflicting"]


def _empty_field_state(field_name: str) -> dict:
    return {
        "field": field_name,
        "status": "unknown",
        "value": None,
        "confidence": 0.0,
        "evidence": [],
        "source_turn_ids": [],
        "support_count": 0,
        "conflicts": [],
    }


def _empty_knowledge_state() -> dict[str, dict]:
    return {f: _empty_field_state(f) for f in SCHEMA_FIELDS}


# ─── GAP SELECTION ────────────────────────────────────────────────────────────

def select_next_gap(state: dict[str, dict]) -> Optional[dict]:
    """Return the highest-priority knowledge gap or None if schema is complete."""
    best = None
    best_score = -1
    for fname, fstate in state.items():
        st = fstate["status"]
        if st == "verified":
            continue
        score = 0
        if st == "conflicting":
            score = 100
        elif st == "unknown":
            score = 80
        elif st == "candidate":
            score = 60 + (1 - fstate.get("confidence", 0)) * 10
        elif st == "supported":
            score = 30 + max(0, 2 - fstate.get("support_count", 0)) * 5
        if score > best_score:
            best_score = score
            reason = {
                "conflicting": "Conflicting evidence requires resolution.",
                "unknown": "No information collected yet.",
                "candidate": "Only weak evidence exists.",
                "supported": "Evidence exists but needs corroboration.",
            }.get(st, "")
            best = {
                "field": fname,
                "priority": round(score / 100, 2),
                "status": st,
                "reason": reason,
            }
    return best


# ─── STATE MACHINE TRANSITION ─────────────────────────────────────────────────

def apply_extraction_update(state: dict[str, dict], update: dict, turn_id: int) -> dict[str, dict]:
    """Merge a structured extraction update into the knowledge state machine."""
    fname = update.get("field", "").strip()
    if fname not in state:
        return state  # Unknown field — ignore

    new_value = str(update.get("value", "")).strip()
    new_evidence = str(update.get("evidence", "")).strip()
    new_conf = float(update.get("confidence", 0.5))

    fstate = state[fname]
    cur_status = fstate["status"]

    if cur_status == "unknown":
        fstate["status"] = "candidate"
        fstate["value"] = new_value
        fstate["confidence"] = new_conf
        fstate["evidence"].append({"text": new_evidence, "turn_id": turn_id})
        fstate["source_turn_ids"].append(turn_id)
        fstate["support_count"] = 1

    elif cur_status == "candidate":
        if _values_agree(fstate["value"], new_value):
            fstate["status"] = "supported"
            fstate["support_count"] += 1
            fstate["confidence"] = max(fstate["confidence"], new_conf)
            fstate["evidence"].append({"text": new_evidence, "turn_id": turn_id})
            fstate["source_turn_ids"].append(turn_id)
        else:
            fstate["conflicts"].append({"value": new_value, "evidence": new_evidence, "source_turn_ids": [turn_id]})
            fstate["status"] = "conflicting"

    elif cur_status == "supported":
        if _values_agree(fstate["value"], new_value):
            fstate["support_count"] += 1
            fstate["confidence"] = min(1.0, fstate["confidence"] + 0.1)
            fstate["evidence"].append({"text": new_evidence, "turn_id": turn_id})
            fstate["source_turn_ids"].append(turn_id)
            if fstate["support_count"] >= 3:
                fstate["status"] = "verified"
        else:
            fstate["conflicts"].append({"value": new_value, "evidence": new_evidence, "source_turn_ids": [turn_id]})
            fstate["status"] = "conflicting"

    elif cur_status == "conflicting":
        fstate["conflicts"].append({"value": new_value, "evidence": new_evidence, "source_turn_ids": [turn_id]})
        fstate["confidence"] = max(fstate.get("confidence", 0.0), new_conf)

    elif cur_status == "verified":
        if not _values_agree(fstate["value"], new_value):
            fstate["conflicts"].append({"value": new_value, "evidence": new_evidence, "source_turn_ids": [turn_id]})
        else:
            fstate["evidence"].append({"text": new_evidence, "turn_id": turn_id})

    return state


def _values_agree(a: Optional[str], b: str, min_overlap: float = 0.3) -> bool:
    """Rough semantic agreement check between two string values."""
    if not a:
        return False
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    if not a_tokens or not b_tokens:
        return False
    overlap = len(a_tokens & b_tokens) / max(len(a_tokens), len(b_tokens))
    return overlap >= min_overlap


# ─── SCHEMA COMPLETENESS ──────────────────────────────────────────────────────

def schema_completeness(state: dict[str, dict]) -> float:
    """Fraction of total weight achieved across all fields (0–1)."""
    max_score = len(SCHEMA_FIELDS) * 3  # 3 points per field at 'verified'
    score_map = {"unknown": 0, "candidate": 1, "supported": 2, "verified": 3, "conflicting": 1}
    actual = sum(score_map.get(fstate["status"], 0) for fstate in state.values())
    return round(actual / max_score, 4)


# ─── JSON SCRUBBER ────────────────────────────────────────────────────────────

def _scrub_json(raw: str) -> str:
    """Extract raw JSON substring so json.loads() can succeed."""
    raw = raw.strip()
    raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    for start_char, end_char in (("{" , "}"), ("[", "]")):
        start = raw.find(start_char)
        if start == -1:
            continue
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
                if ch == start_char:
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                    if depth == 0:
                        return raw[start: i + 1]
    return raw


# ─── SENTENCE-LEVEL FALLBACK EXTRACTOR ───────────────────────────────────────

_FIELD_KEYWORDS: dict[str, list[str]] = {
    "domain_area":        ["domain", "area", "field", "sector", "industry", "topic", "subject", "about", "type"],
    "primary_challenge":  ["challenge", "problem", "issue", "difficult", "struggle", "obstacle", "failure", "outage", "crisis"],
    "root_cause":         ["because", "caused", "due to", "reason", "trigger", "origin", "root", "resulted from", "led to"],
    "key_decisions":      ["decided", "chose", "implemented", "adopted", "migrated", "deployed", "changed", "switched", "moved"],
    "technologies_used":  ["python","docker","kubernetes","git","database","server","api","cloud","platform","software","system","tool","technology","framework"],
    "people_involved":    ["team", "manager", "engineer", "developer", "staff", "employee", "lead", "analyst", "nurse", "doctor", "admin", "director"],
    "processes_affected": ["process", "workflow", "procedure", "deployment", "testing", "monitoring", "review", "onboarding", "training", "release"],
    "operational_impact": ["impact", "result", "improved", "reduced", "saved", "better", "faster", "outcome", "effect", "benefit"],
    "lessons_learned":    ["learned", "lesson", "takeaway", "next time", "going forward", "recommend", "should", "avoid", "improve"],
}


def _sentence_level_fallback(text: str, turn_id: int) -> list[dict]:
    """Extract facts from plain text when LLM JSON fails."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    updates: list[dict] = []
    covered_fields: set[str] = set()

    for sent in sentences:
        sent = sent.strip()
        if len(sent.split()) < 4:
            continue
        sent_lower = sent.lower()
        best_field = None
        best_score = 0
        for field, keywords in _FIELD_KEYWORDS.items():
            if field in covered_fields:
                continue
            score = sum(1 for kw in keywords if kw in sent_lower)
            if score > best_score:
                best_score = score
                best_field = field
        if best_field and best_score >= 1:
            updates.append({
                "field": best_field,
                "value": sent[:200],
                "evidence": sent[:200],
                "confidence": min(0.3 + best_score * 0.1, 0.6),
            })
            covered_fields.add(best_field)
    return updates



# ─── INTERVIEW SESSION ────────────────────────────────────────────────────────

class InterviewSession:
    """Fully stateful interview session for one model/episode pair."""

    def __init__(
        self,
        episode_id: str,
        model_id: str,
        runner: ChatRunner,
        rag_enabled: bool = True,
        persist_runtime: bool = False,
        ablation: str = "none"
    ):
        self.episode_id = episode_id
        self.model_id = model_id
        self.runner = runner
        self.rag_enabled = rag_enabled
        self.persist_runtime = persist_runtime
        self.ablation = ablation
        
        self.turn_count = 0
        self.is_done = False

        # Knowledge state machine
        self.knowledge_state: dict[str, dict] = _empty_knowledge_state()
        self.completeness: float = 0.0

        # Output tracking
        self.turns: list[dict] = []
        self.question_history: list[str] = []

        self.opening_question = OPENING
        self.turns.append({
            "role": "interviewer",
            "text": OPENING,
            "turn_id": 0,
            "target_field": None,
            "question_type": "opening",
        })

        self.atomic_facts: list[dict] = []
        self.storage_success: bool = False
        self.storage_error: Optional[str] = None
        self.artifact_id: Optional[str] = None

    def respond(self, participant_text: str) -> dict:
        """Process one participant answer and return the updated session state."""
        if self.is_done:
            return self._state()

        self.turn_count += 1
        turn_id = self.turn_count

        record = {"role": "participant", "text": participant_text, "turn_id": turn_id}
        self.turns.append(record)

        # Snapshot fields that were unknown before, now surfaced
        previously_unknown = {
            f for f, s in self.knowledge_state.items() if s["status"] == "unknown"
        }

        extract_result = self._extract(participant_text, turn_id)
        self.completeness = schema_completeness(self.knowledge_state)

        # Compute tacit-knowledge novelty: fields newly extracted this turn
        still_unknown = {
            f for f, s in self.knowledge_state.items() if s["status"] == "unknown"
        }
        newly_surfaced = list(previously_unknown - still_unknown)
        novel_facts_this_turn = len(newly_surfaced)

        record["extraction"] = extract_result.get("updates", [])
        record["knowledge_state_after"] = deepcopy(self.knowledge_state)
        record["completeness_after"] = self.completeness
        record["newly_surfaced_fields"] = newly_surfaced
        record["novel_facts_count"] = novel_facts_this_turn

        # RAG Logic
        rag = []
        retrieval_latency_ms = 0
        if self.rag_enabled:
            t_ret = time.perf_counter()
            rag = query(participant_text, n=RAG_TOP_K, where={"namespace": "corpus"})
            if not rag:
                rag = query(participant_text, n=RAG_TOP_K)
            retrieval_latency_ms = int((time.perf_counter() - t_ret) * 1000)
            
            # Format explicitly for trace
            for r in rag:
                r["retrieved_context_supports_target"] = False # Default placeholder to be updated later by tests/logic

        if self.turn_count >= MAX_TURNS or self.completeness >= COMPLETION_THRESHOLD:
            return self._complete()

        # Generate next question
        t_gen = time.perf_counter()
        question_result = self._generate_question(rag)
        generation_latency_ms = int((time.perf_counter() - t_gen) * 1000)

        question_text = question_result.get("question", "Could you elaborate further?")
        target_field = question_result.get("target_field", None)
        question_type = question_result.get("question_type", "open")
        information_goal = question_result.get("information_goal", "Gain more context.")
        gap_reason = question_result.get("reason", "")
        
        knowledge_gap_targeted = str(target_field) if target_field else None

        redundant, similarity = self._is_redundant(question_text)

        q_record = {
            "role": "interviewer",
            "text": question_text,
            "turn_id": turn_id,
            "target_field": target_field,
            "question_type": question_type,
            "information_goal": information_goal,
            "knowledge_state_before_question": deepcopy(self.knowledge_state),
            "knowledge_gap_targeted": knowledge_gap_targeted,
            "gap_selection_reason": gap_reason,
            "rag": rag,
            "redundant": redundant,
            "similarity_to_previous": round(similarity, 4),
            "generation_latency_ms": generation_latency_ms,
        }
        self.turns.append(q_record)
        self.question_history.append(question_text)

        return self._state()

    def _extract(self, text: str, turn_id: int) -> dict:
        t0 = time.time()
        
        system_prompt = (
            "You are a knowledge extraction specialist. "
            "Extract factual information from the participant's answer and return ONLY valid JSON.\n\n"
            "Return this exact structure:\n"
            '{"updates": [{"field": "...", "value": "...", "evidence": "...", "confidence": 0.85}]}\n\n'
            f"Supported fields: {', '.join(SCHEMA_FIELDS)}\n\n"
            "Rules:\n"
            "- Only extract explicitly stated facts.\n"
            "- 'evidence' must be a direct quote or close paraphrase from the answer.\n"
            "- confidence ranges 0.0 to 1.0 based on how clearly the fact was stated.\n"
            "- Use [] for updates if nothing applies.\n"
            "- Return ONLY the JSON object. No markdown. No explanation."
        )
        user_msg = f"Participant answer:\n{text}"
        
        retry_msg = (
            f"Text: {text[:400]}\n\n"
            "Output JSON only: "
            '{"updates": [{"field": "primary_challenge", "value": "short description", '
            '"evidence": "quote from text", "confidence": 0.7}]}\n'
            "Replace field name and value with what you found. Use [] if nothing."
        )

        updates = []
        extraction_method = "empty"
        error_msg = None

        for attempt, msg in enumerate([user_msg, retry_msg]):
            try:
                resp = self.runner.generate(
                    messages=[
                        {"role": "system", "content": system_prompt if attempt == 0 else "Return JSON only."},
                        {"role": "user", "content": msg},
                    ],
                    max_tokens=400,
                    temperature=0.0,
                )
                raw_output = resp.get("text", "")
                raw_json = _scrub_json(raw_output)

                parsed = json.loads(raw_json)
                if isinstance(parsed, list):
                    parsed = {"updates": parsed}
                if not isinstance(parsed, dict) or "updates" not in parsed:
                    raise ValueError("Missing 'updates' key")

                valid_updates = [
                    u for u in parsed["updates"]
                    if isinstance(u, dict) and u.get("field") in SCHEMA_FIELDS and u.get("value")
                ]
                if valid_updates:
                    updates = valid_updates
                    extraction_method = "llm"
                    break
                else:
                    error_msg = f"Attempt {attempt+1}: No valid updates"
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                error_msg = f"Attempt {attempt+1}: {type(exc).__name__}: {exc}"
            except Exception as exc:
                error_msg = f"Attempt {attempt+1}: Unexpected: {exc}"
                break

        if not updates:
            fallback = _sentence_level_fallback(text, turn_id)
            if fallback:
                updates = fallback
                extraction_method = "fallback"

        latency_ms = int((time.time() - t0) * 1000)

        for upd in updates:
            if isinstance(upd, dict) and "field" in upd and "value" in upd:
                apply_extraction_update(self.knowledge_state, upd, turn_id)
                self.atomic_facts.append({
                    "fact_id": f"{self.episode_id}_t{turn_id:02d}_{upd['field']}",
                    "episode_id": self.episode_id,
                    "turn_id": turn_id,
                    "field": upd["field"],
                    "value": upd.get("value", ""),
                    "evidence": upd.get("evidence", ""),
                    "confidence": upd.get("confidence", 0.5),
                    "status": self.knowledge_state.get(upd["field"], {}).get("status", "candidate"),
                    "source_turn_ids": [turn_id],
                })

        return {
            "updates": updates,
            "success": extraction_method in ("llm", "fallback"),
            "extraction_method": extraction_method,
            "error": error_msg,
            "latency_ms": latency_ms,
        }

    def _generate_question(self, rag: list[dict]) -> dict:
        recent_turns = self._recent_turns_text()
        rag_snippets = "\n".join(f"- {r['text'][:200]}" for r in rag[:3])
        
        state_summary = {
            fname: {"status": fstate["status"], "value": fstate["value"]}
            for fname, fstate in self.knowledge_state.items()
        }

        if self.ablation == "state_machine":
            # State-machine ablation: free-form follow-up based on context without explicit gap reasoning
            system_prompt = (
                "You are an expert knowledge elicitation interviewer. "
                "Generate a follow-up question based on the recent conversation flow. "
                "Return ONLY valid JSON with this structure:\n"
                '{"question": "...", "target_field": null, "question_type": "open", "information_goal": "..."}\n'
                "Do not use explicit gap-tracking, just respond naturally to what the participant just said."
            )
            user_msg = f"Recent conversation:\n{recent_turns}\n\nBackground:\n{rag_snippets}\n\nGenerate next question."
        else:
            # Normal adaptive behavior
            gap = select_next_gap(self.knowledge_state)
            gap_text = json.dumps(gap) if gap else "None"
            
            system_prompt = (
                "You are an expert knowledge elicitation interviewer. "
                "Generate ONE precise follow-up question targeting the identified knowledge gap. "
                "Return ONLY valid JSON.\n\n"
                "Return this exact structure:\n"
                '{"question": "...", "target_field": "...", "question_type": "clarification|probe|confirmation|expansion", "information_goal": "...", "reason": "..."}\n\n'
                "Rules:\n"
                "- The question must directly target the specified gap field.\n"
                "- Use the participant's recent answers to contextualise the question.\n"
                "- Use retrieved knowledge snippets to inform, not to lead."
            )
            user_msg = (
                f"Knowledge gap to target:\n{gap_text}\n\n"
                f"Current knowledge state:\n{json.dumps(state_summary, indent=2)}\n\n"
                f"Recent conversation:\n{recent_turns}\n\n"
                f"Background:\n{rag_snippets if rag_snippets else 'None'}\n\n"
                f"Previous questions: {self.question_history[-3:]}\n\n"
                "Generate next question."
            )

        try:
            resp = self.runner.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=200,
                temperature=0.7,
            )
            raw_json = _scrub_json(resp.get("text", ""))
            parsed = json.loads(raw_json)
            question = parsed.get("question", "").strip()
            if not question:
                raise ValueError("Empty question")
            
            if self.ablation == "state_machine":
                parsed["reason"] = "state_machine_ablation_enabled"
                
            return parsed
        except (json.JSONDecodeError, ValueError):
            return {
                "question": "Could you tell me more about that?",
                "target_field": None,
                "question_type": "open",
                "information_goal": "Fallback clarification due to parse error",
                "reason": "fallback",
            }

    def _is_redundant(self, question: str) -> tuple[bool, float]:
        if not self.question_history:
            return False, 0.0
            
        q_lower = question.lower()
        max_overlap = 0.0
        for prev in self.question_history:
            words_q = set(q_lower.split())
            words_p = set(prev.lower().split())
            if words_q and words_p:
                overlap = len(words_q & words_p) / max(len(words_q), len(words_p))
                max_overlap = max(max_overlap, overlap)
                if overlap >= REDUNDANCY_THRESHOLD:
                    return True, overlap
        return False, max_overlap

    def _recent_turns_text(self) -> str:
        recent = [t for t in self.turns if t["role"] in ("participant", "interviewer")]
        recent = recent[-RECENT_TURNS_WINDOW:]
        lines = []
        for t in recent:
            role = "Interviewer" if t["role"] == "interviewer" else "Participant"
            lines.append(f"{role}: {t['text'][:300]}")
        return "\n".join(lines)

    def _complete(self) -> dict:
        self.is_done = True

        if self.persist_runtime:
            try:
                for fact in self.atomic_facts:
                    store_atomic_fact(
                        fact_id=fact["fact_id"],
                        interview_id=self.episode_id,
                        turn_id=fact["turn_id"],
                        field=fact["field"],
                        value=fact["value"],
                        evidence=fact["evidence"],
                        confidence=fact["confidence"],
                        status=fact["status"],
                        model_id=self.model_id,
                        source_turn_ids=fact["source_turn_ids"],
                        namespace="interview_fact",
                    )
    
                summary_parts = []
                for fname, fstate in self.knowledge_state.items():
                    if fstate["value"]:
                        summary_parts.append(f"{fname.replace('_', ' ').capitalize()}: {fstate['value']}")
                summary_text = ". ".join(summary_parts) + f". Completed in {self.turn_count} turns."
    
                self.artifact_id = store_interview_artifact(
                    run_id=self.episode_id,
                    model_id=self.model_id,
                    turns=self.turn_count,
                    summary_text=summary_text,
                )
                self.storage_success = True
            except Exception as exc:
                self.storage_success = False
                self.storage_error = str(exc)
        else:
            self.storage_success = True
            self.artifact_id = "mem_" + self.episode_id

        completion_msg = (
            f"Interview complete ({self.turn_count} turns). "
            f"Completeness: {self.completeness:.0%}. "
            f"Artifact: {self.artifact_id}"
        )
        self.turns.append({"role": "system", "text": completion_msg, "turn_id": self.turn_count})
        return self._state()

    def _state(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "model_id": self.model_id,
            "ablation": self.ablation,
            "rag_enabled": self.rag_enabled,
            "turns": self.turns,
            "turn_count": self.turn_count,
            "is_done": self.is_done,
            "knowledge_state": self.knowledge_state,
            "completeness": self.completeness,
            "atomic_facts": self.atomic_facts,
            "storage_success": self.storage_success,
            "storage_error": self.storage_error,
            "artifact_id": self.artifact_id,
        }
