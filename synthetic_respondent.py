"""
synthetic_respondent.py -- Deterministic synthetic participant for automated evaluation.

The respondent answers interview questions using ONLY the reference benchmark answer.
It never invents facts. It is fully deterministic given (iteration_seed, case_id, turn_id).

Algorithm:
    1. Split reference answer into evidence sentences.
    2. Embed the interviewer question using Chroma's fixed embedding function.
    3. Score each evidence sentence by word overlap with the question.
    4. Return the most relevant sentences as the participant answer.
    5. If nothing is relevant, return a controlled "not specified" response.

No LLM is used here — the respondent is entirely deterministic.
The generator model is the experimental variable; the respondent is a CONSTANT.
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import Optional


# Minimum word overlap score to include a sentence in the answer
MIN_RELEVANCE_SCORE = 0.05

NOT_SPECIFIED_RESPONSES = [
    "I don't have specific information on that.",
    "That wasn't mentioned in what I was briefed on.",
    "I'm not sure about that particular aspect.",
    "The details on that point aren't available to me.",
    "I cannot elaborate further on that topic.",
]


def _split_into_evidence_units(text: str) -> list[str]:
    """Split reference answer into atomic evidence sentences."""
    # Split on sentence boundaries
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    units = []
    for sent in raw:
        sent = sent.strip()
        if len(sent.split()) >= 5:  # skip fragments shorter than 5 words
            units.append(sent)
    return units


def _word_overlap_score(query: str, sentence: str) -> float:
    """Compute normalised word overlap between query and sentence."""
    q_words = set(re.sub(r"[^a-z0-9 ]", "", query.lower()).split())
    s_words = set(re.sub(r"[^a-z0-9 ]", "", sentence.lower()).split())
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "have", "has",
        "do", "does", "did", "will", "would", "could", "should", "may", "might",
        "what", "which", "who", "that", "this", "these", "those", "in", "on",
        "at", "of", "for", "to", "from", "with", "by", "about", "can", "you",
        "your", "me", "my", "it", "its", "we", "our", "they", "their",
    }
    q_meaningful = q_words - stopwords
    s_meaningful = s_words - stopwords
    if not q_meaningful or not s_meaningful:
        return 0.0
    overlap = len(q_meaningful & s_meaningful)
    denominator = max(len(q_meaningful), len(s_meaningful))
    return overlap / denominator


def _derive_seed(iteration_seed: int, case_id: str, turn_id: int) -> int:
    """Generate a deterministic seed from iteration/case/turn triplet."""
    raw = f"{iteration_seed}_{case_id}_{turn_id}"
    return int(hashlib.md5(raw.encode()).hexdigest(), 16) % (2**31)


class SyntheticRespondent:
    """Deterministic synthetic participant.

    Answers are grounded strictly in the reference answer.
    All selection is deterministic given (iteration_seed, case_id, turn_id).
    """

    def __init__(self, case_id: str, reference_answer: str, iteration_seed: int = 0):
        self.case_id = case_id
        self.reference_answer = reference_answer
        self.iteration_seed = iteration_seed
        self.evidence_units = _split_into_evidence_units(reference_answer)

    def answer(self, question: str, turn_id: int, max_sentences: int = 3) -> dict:
        """Return a deterministic grounded answer to the interviewer question.

        Args:
            question: The interviewer's question.
            turn_id:  Current interview turn (used for reproducible randomness).
            max_sentences: Maximum number of evidence sentences to return.

        Returns:
            dict with: text, evidence_indices, relevance_scores, not_specified
        """
        if not self.evidence_units:
            return self._not_specified(turn_id)

        # Score each evidence unit against the question
        scored = [
            (i, unit, _word_overlap_score(question, unit))
            for i, unit in enumerate(self.evidence_units)
        ]
        # Sort by relevance descending
        scored.sort(key=lambda x: -x[2])

        # Filter by minimum relevance
        relevant = [(i, unit, s) for i, unit, s in scored if s >= MIN_RELEVANCE_SCORE]

        if not relevant:
            return self._not_specified(turn_id)

        # Use deterministic RNG for selection when relevance scores are tied
        rng = random.Random(_derive_seed(self.iteration_seed, self.case_id, turn_id))

        # Take up to max_sentences; deterministically shuffle tied groups
        selected = relevant[:max_sentences]
        # Sort selected by original order for coherence
        selected.sort(key=lambda x: x[0])

        answer_text = " ".join(unit for _, unit, _ in selected)
        return {
            "text": answer_text,
            "evidence_indices": [i for i, _, _ in selected],
            "relevance_scores": [round(s, 4) for _, _, s in selected],
            "not_specified": False,
        }

    def _not_specified(self, turn_id: int) -> dict:
        rng = random.Random(_derive_seed(self.iteration_seed, self.case_id, turn_id))
        response = rng.choice(NOT_SPECIFIED_RESPONSES)
        return {
            "text": response,
            "evidence_indices": [],
            "relevance_scores": [],
            "not_specified": True,
        }
