"""Deterministic extractive "LLM" used for offline eval/CI and as a
no-model fallback (LLM_PROVIDER=extractive).

It is NOT an LLM: it answers by picking the best-matching sentences from the
policy excerpts embedded in the prompt. This keeps the evaluation pipeline
deterministic and network-free while still exercising the real retrieval +
citation machinery. Injection-looking text is treated as data and refused.

It only understands the prompt format built by app.answering.build_prompt.
"""

from __future__ import annotations

import re

from ..vectordb import tokenize
from .base import LLMError

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_HEADER = re.compile(r"^\[c\d+\]\s*")
_INJECTION_MARKERS = (
    "ignore all previous instructions",
    "ignore previous instructions",
    "note for automated assistants",
    "you are now",
    "you are policyshare",
    "policyshare",
    "shared by policyshare",
)

# Tokens that identify what a question is really about; generic transport
# vocabulary alone must not trigger an answer.
_GENERIC = frozenset(
    """cab cabs shuttle employee employees policy transport company trip trips
    desk booking book shuttle drop drops pickup pickups shift shifts time day
    days provide provided service allowed allow get give tell ad hoc office""".split()
)

# Canonicalisation for matching question words to policy wording (applied to
# BOTH sides before scoring; crude stemming handles regular plurals).
_SYNONYMS = {
    "woman": "women",
    "lady": "women",
    "ladies": "women",
    "female": "women",
    "month": "monthly",
    "monthly": "monthly",
    "accompany": "escort",
    "chaperone": "escort",
    "escorted": "escort",
    "approve": "approval",
    "approves": "approval",
    "approved": "approval",
    "reimbursement": "reimburse",
    "reimbursed": "reimburse",
    "reimburses": "reimburse",
    "saturday": "weekend",
    "sundays": "weekend",
    "sunday": "weekend",
}


def _canon(token: str) -> str:
    token = _SYNONYMS.get(token, token)
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def _distinctive(text: str) -> set[str]:
    """Canonical tokens that actually identify the topic."""

    return {c for t in tokenize(text) if t not in _GENERIC for c in [_canon(t)]}


class ExtractiveLLM:
    def __init__(self, min_overlap: int = 1) -> None:
        self._min_overlap = min_overlap
        self.stats = type("S", (), {"calls": 0})()  # minimal stats surface

    def complete(
        self,
        *,
        system: str,
        user: str,
        timeout_s: float,
        max_tokens: int,
        temperature: float = 0.0,
        force_json: bool = True,
    ) -> str:
        import json

        self.stats.calls += 1
        question, excerpts = _parse_prompt(user)
        q_terms = _distinctive(question)
        best_chunk, best_score, best_sentence = None, 0, ""
        for chunk_id, text in excerpts:
            for sentence in _SENTENCE.split(text):
                clean = _HEADER.sub("", sentence).strip()
                if _looks_like_injection(clean):
                    continue
                s_terms = {c for t in tokenize(clean) for c in [_canon(t)]}
                score = len(q_terms & s_terms)
                if score > best_score:
                    best_chunk, best_score, best_sentence = chunk_id, score, clean

        if best_score >= self._min_overlap and best_sentence:
            answer = _trim_words(best_sentence, 55)
            return (
                "ANSWERABLE: yes\n"
                f"ANSWER: {answer}\n"
                f"EVIDENCE: {best_chunk}"
            )
        return "ANSWERABLE: no\nANSWER: \nEVIDENCE:"

    def healthcheck(self, timeout_s: float = 1.0) -> bool:
        return True

    def describe(self) -> str:
        return "extractive-fallback"


def _looks_like_injection(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _INJECTION_MARKERS)


def _trim_words(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words])


def _parse_prompt(user: str) -> tuple[str, list[tuple[str, str]]]:
    """Split the built prompt back into question + excerpt texts."""

    question = ""
    excerpts: list[tuple[str, str]] = []
    current_id: str | None = None
    current_lines: list[str] = []
    in_question = False

    for line in user.splitlines():
        stripped = line.strip()
        if stripped.startswith("Question:"):
            question = stripped[len("Question:") :].strip()
            in_question = True
            continue
        if stripped.startswith("Policy excerpts"):
            in_question = False
            continue
        if stripped.startswith("[c") and "]" in stripped:
            if current_id is not None:
                excerpts.append((current_id, "\n".join(current_lines)))
            current_id = stripped[1 : stripped.index("]")]
            current_lines = [stripped[stripped.index("]") + 1 :].strip()]
            continue
        if current_id is not None:
            current_lines.append(line)
    if current_id is not None:
        excerpts.append((current_id, "\n".join(current_lines)))
    if not question:
        raise LLMError("extractive fallback could not parse the prompt")
    return question, excerpts
