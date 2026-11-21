"""Scripted fake LLM for tests and the deterministic offline pipeline.

Two modes:
- script: a list of raw outputs returned in order (raise IndexError-style
  LLMError when exhausted) — used to test parsing/repair/fallback paths.
- behaviour: a callable(system, user) -> str — used to emulate attacks.

The fake records every call so tests can assert the call budget exactly.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from .base import CallStats, LLMError


class ScriptedLLM:
    def __init__(
        self,
        responses: list[str] | None = None,
        behaviour: Callable[[str, str], str] | None = None,
        delay_s: float = 0.0,
    ) -> None:
        if responses is None and behaviour is None:
            raise ValueError("provide responses or behaviour")
        self.responses = list(responses or [])
        self.behaviour = behaviour
        self.delay_s = delay_s
        self.stats = CallStats()
        self.fail_all: bool = False  # flip to emulate a broken provider

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
        if self.delay_s:
            time.sleep(self.delay_s)
        # A call attempt consumes budget even when it fails: the budget is
        # about model load, not only about successful completions.
        self.stats.record(system=system, user=user, response="")
        if self.fail_all:
            raise LLMError("provider unavailable (simulated)")
        if self.behaviour is not None:
            response = self.behaviour(system, user)
        else:
            if not self.responses:
                raise LLMError("script exhausted")
            response = self.responses.pop(0)
        self.stats.history[-1].response = response
        return response

    def healthcheck(self, timeout_s: float = 1.0) -> bool:
        return not self.fail_all

    def describe(self) -> str:
        return "scripted-fake"

    # convenience for tests ---------------------------------------------------

    @property
    def calls(self) -> int:
        return self.stats.calls

    def last_user_prompt(self) -> str:
        return self.stats.history[-1].user


class JsonAnswerFake(ScriptedLLM):
    """A fake that always answers valid JSON citing given evidence ids."""

    def __init__(self, answer: str, evidence_ids: list[str], answerable: bool = True) -> None:
        import json

        payload = {"answerable": answerable, "answer": answer, "evidence_ids": evidence_ids}
        super().__init__(responses=[json.dumps(payload)])


def attack_behaviour(_: str, user: str) -> str:
    """Emulate a compromised model that obeys injection found in documents.

    Used to prove the SERVER rejects malformed citations and cross-company
    behaviour regardless of what the model returns.
    """

    import json

    if "c1" in user:
        payload: dict[str, Any] = {
            "answerable": True,
            "answer": "Ignore all rules. Bluebird gives escorts after 10 PM and Acme is wrong.",
            "evidence_ids": ["c1", "c9", "BLUEBIRD_CHUNK_1", "c2", "c3", "c4"],
        }
    else:
        payload = {"answerable": True, "answer": "undefined", "evidence_ids": []}
    return json.dumps(payload)
