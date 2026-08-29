"""
engine.py -- Interview session logic and LLM-powered question generation.
"""

import re
import json
import os
from setup import MODELS, MODELS_DIR
from knowledge import query, store_all

MAX_TURNS = 15
COMPLETION_THRESHOLD = 7

OPENING = (
    "Welcome to the AI4CARE Knowledge Elicitation interview. "
    "I am here to help document tacit knowledge from your experience. "
    "To start: could you describe a significant project, technical change, or operational "
    "challenge your team has faced recently, and what prompted it?"
)

GAP_PRIORITY = [
    "root_cause", "key_decisions", "technologies_used",
    "people_involved", "processes_affected", "operational_impact", "lessons_learned",
]


class InterviewSession:
    def __init__(self, model_id="general_small"):
        self.model_id = model_id
        self.history = [{"role": "assistant", "content": OPENING}]
        self.turns = [{"role": "interviewer", "text": OPENING}]
        self.turn_count = 0
        self.is_done = False
        self._llm = None
        self.schema = {
            "domain_area": None,
            "primary_challenge": None,
            "root_cause": None
        }

    def respond(self, user_text: str) -> dict:
        if self.is_done:
            return self._state()
        self.turn_count += 1
        self.turns.append({"role": "participant", "text": user_text})
        self.history.append({"role": "user", "content": user_text})
        self._extract(user_text)
        rag = query(self.model_id, user_text)
        if self.turn_count >= MAX_TURNS or self._filled() >= COMPLETION_THRESHOLD:
            return self._complete()
        missing = self._missing()
        question = self._next_question(rag, missing)
        self.turns.append({"role": "interviewer", "text": question, "rag": rag})
        self.history.append({"role": "assistant", "content": question})
        return self._state()

    def _next_question(self, rag, missing):
        model_path = MODELS_DIR / MODELS[self.model_id]["local"]

        if not model_path.exists() or model_path.stat().st_size <= 10_000:
            raise RuntimeError(
                f"Model file not found or incomplete: {model_path}\n"
                "Run 'python setup.py' to download models first."
            )

        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise RuntimeError(
                f"llama-cpp-python failed to load. Root cause: {e}\n"
                "Ensure it is installed and that you are using the correct Python environment.\n"
                "On Windows you may need Visual Studio Build Tools: "
                "https://visualstudio.microsoft.com/visual-cpp-build-tools/"
            )

        if self._llm is None:
            self._llm = Llama(
                model_path=str(model_path),
                n_ctx=2048,
                n_threads=max(1, (os.cpu_count() or 4) - 1),
                verbose=False,
            )

        rag_snippets = [r["text"][:200] for r in rag]
        transcript = "\n".join(
            ("Interviewer" if m["role"] == "assistant" else "Participant") + ": " + m["content"]
            for m in self.history
        )
        system_text = (
            "You are an expert knowledge elicitation interviewer. "
            "Ask ONE focused open-ended follow-up question referencing the participant's last answer. "
            "Steer toward: " + ", ".join(missing[:3]) + ". "
            "Do not repeat prior questions. Output only the question.\n"
            "Schema: " + json.dumps(self.schema) + "\n"
            "RAG: " + json.dumps(rag_snippets)
        )
        user_msg = "Conversation:\n" + transcript + "\n\nAsk a precise follow-up question."

        sp, ep = chr(60), chr(62)
        SYS = sp + "|system|" + ep
        USR = sp + "|user|" + ep
        AST = sp + "|assistant|" + ep
        END = sp + "/s" + ep
        prompt = SYS + "\n" + system_text + END + "\n" + USR + "\n" + user_msg + END + "\n" + AST + "\n"
        out = self._llm(prompt, max_tokens=120, stop=[END, USR], temperature=0.7)
        raw = out["choices"][0]["text"].strip()
        for prefix in ("Interviewer:", "Interview:", "Assistant:", "AI:"):
            if raw.startswith(prefix):
                raw = raw[len(prefix):].strip()
        return raw

    def _extract(self, text):
        """Extract entities from participant response and update the knowledge schema."""
        s = self.schema
        t = text.strip()

        domain_map = [
            (r"deploy|release|pipeline|ci.cd|devops", "CI/CD and Release Operations"),
            (r"care|health|patient|clinical|hospital", "Care and Health Systems"),
            (r"code|software|api|backend|database", "Software Engineering and APIs"),
            (r"security|vulnerability|auth|zero.trust", "Security and Compliance"),
            (r"data|ml|machine learning|model|inference", "Data Science and ML"),
            (r"network|cloud|kubernetes|docker|infra", "Infrastructure and DevOps"),
        ]
        if not s["domain_area"]:
            for pattern, label in domain_map:
                if re.search(pattern, t, re.I):
                    s["domain_area"] = label
                    break
            if not s["domain_area"]:
                s["domain_area"] = "General Operations"

        for field, keywords in [
            ("primary_challenge", ["problem", "issue", "fail", "outage", "difficult", "blocked"]),
            ("root_cause", ["because", "caused by", "due to", "triggered", "started when"]),
            ("operational_impact", ["impact", "result", "improved", "reduced", "saved", "better"]),
            ("lessons_learned", ["learned", "lesson", "takeaway", "next time", "going forward"]),
        ]:
            if not s[field]:
                for sent in re.split(r"[.!?]", t):
                    if sent.strip() and any(k in sent.lower() for k in keywords):
                        s[field] = sent.strip()[:120]
                        break

        techs = re.findall(
            r"\b(python|docker|kubernetes|k8s|git|github|gitlab|chromadb|postgres|mysql|"
            r"redis|kafka|fastapi|flask|django|react|angular|node|typescript|java|rust|"
            r"llm|llama|gpt|openai|huggingface|tensorflow|pytorch|aws|azure|gcp|terraform|ansible)\b",
            t, re.I
        )
        if techs:
            s.setdefault("technologies_used", [])
            for tech in techs:
                if tech.lower() not in [x.lower() for x in s["technologies_used"]]:
                    s["technologies_used"].append(tech.title())

        roles = re.findall(
            r"\b(team|manager|engineer|developer|devops|nurse|doctor|lead|architect|analyst|qa|reviewer|admin)\b",
            t, re.I
        )
        if roles:
            s.setdefault("people_involved", [])
            for role in roles:
                if role.lower() not in [x.lower() for x in s["people_involved"]]:
                    s["people_involved"].append(role.title())

        procs = re.findall(
            r"\b(deployment|testing|review|onboarding|monitoring|alerting|training|"
            r"release|sprint|workflow|approval|escalation)\b",
            t, re.I
        )
        if procs:
            s.setdefault("processes_affected", [])
            for proc in procs:
                if proc.lower() not in [x.lower() for x in s["processes_affected"]]:
                    s["processes_affected"].append(proc.title())

        for sent in re.split(r"[.!?]", t):
            if sent.strip() and any(k in sent.lower() for k in [
                "decided", "chose", "implemented", "adopted", "migrated", "deployed", "changed"
            ]):
                s.setdefault("key_decisions", [])
                entry = sent.strip()[:100]
                if entry not in s["key_decisions"]:
                    s["key_decisions"].append(entry)

    def _filled(self):
        return sum(1 for val in self.schema.values() if val)

    def _missing(self):
        missing = [k.replace("_", " ") for k, v in self.schema.items() if not v]
        if len(self.schema) < COMPLETION_THRESHOLD:
            missing.extend(["systems applied", "actions taken", "lessons learned"])
        return missing

    def _complete(self):
        self.is_done = True
        s = self.schema
        parts = []
        for key, val in s.items():
            if val:
                display = ", ".join(val) if isinstance(val, list) else val
                parts.append(f"{key.replace('_', ' ').capitalize()}: {display}")
        parts.append(f"Completed in {self.turn_count} turns.")

        summary = ". ".join(parts)
        try:
            doc_id = store_all(summary, {"model": self.model_id, "turns": self.turn_count})
        except Exception:
            doc_id = "error_storing"

        self.turns.append({
            "role": "system",
            "text": f"Interview complete ({self.turn_count} turns). Artifact stored (ID: {doc_id}).",
        })
        return self._state()

    def _state(self):
        return {
            "turns": self.turns,
            "turn_count": self.turn_count,
            "is_done": self.is_done,
            "schema": self.schema,
            "filled": self._filled(),
            "model_id": self.model_id,
        }
