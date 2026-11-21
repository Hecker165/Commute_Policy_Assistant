"""LLM provider interface + errors.

A narrow sync interface so deterministic fakes can be injected in tests and
the real provider (Ollama) used for the demo. Providers MUST raise
LLMTimeout or LLMError instead of hanging or returning partial output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol


class LLMError(Exception):
    """Provider failed (connection, HTTP error, bad response)."""


class LLMTimeout(LLMError):
    """The call exceeded its per-call timeout."""


@dataclass
class CallRecord:
    system: str
    user: str
    response: str


@dataclass
class CallStats:
    calls: int = 0
    history: list[CallRecord] = field(default_factory=list)

    def record(self, system: str, user: str, response: str) -> None:
        self.calls += 1
        self.history.append(CallRecord(system=system, user=user, response=response))


class LLMProvider(Protocol):
    """Single-call interface. Implementations must be thread-safe enough for
    one concurrent request per provider instance."""

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
        """Return the model's raw text output; raise LLMTimeout/LLMError."""
        ...

    def healthcheck(self, timeout_s: float = 1.0) -> bool:
        """Cheap readiness probe (never raises)."""
        ...

    def describe(self) -> str:
        """Human-readable provider/model name for logs and reports."""
        ...


def parse_json_object(raw: str) -> dict | None:
    """Parse a JSON object out of model output; returns None when invalid.

    Model output is untrusted: we accept only a top-level JSON object (the
    Ollama 'format: json' constrained decoding makes this near-guaranteed,
    and this parser is the second line of defence).
    """

    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None
