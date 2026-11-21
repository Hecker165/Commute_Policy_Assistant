"""Ollama provider: local, free, JSON-constrained decoding.

Uses Ollama's `format: "json"` (grammar-constrained sampling) so the model
cannot emit non-JSON output — the single biggest reliability win for small
models. Temperature 0 and a hard per-call timeout keep runs deterministic
and bounded. Connection errors/timeouts surface as LLMError/LLMTimeout.
"""

from __future__ import annotations

import logging

import httpx

from .base import LLMError, LLMTimeout

logger = logging.getLogger(__name__)


class OllamaLLM:
    def __init__(self, base_url: str, model: str, warm_keep_alive: str = "30m") -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._keep_alive = warm_keep_alive
        # Health probes get their own short timeout; calls set theirs per request.
        self._client = httpx.Client(timeout=httpx.Timeout(5.0, connect=2.0))

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
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 4096,
            },
        }
        if force_json:
            payload["format"] = "json"
        try:
            response = self._client.post(f"{self._base_url}/api/chat", json=payload, timeout=timeout_s)
        except httpx.TimeoutException as exc:
            raise LLMTimeout(f"ollama call exceeded {timeout_s}s") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama connection failed: {type(exc).__name__}") from exc

        if response.status_code != 200:
            raise LLMError(f"ollama returned HTTP {response.status_code}")
        try:
            data = response.json()
            content = data["message"]["content"]
        except (ValueError, KeyError) as exc:
            raise LLMError("ollama response shape unexpected") from exc
        if not isinstance(content, str):
            raise LLMError("ollama content not text")
        return content

    def healthcheck(self, timeout_s: float = 1.0) -> bool:
        try:
            response = self._client.get(f"{self._base_url}/api/tags", timeout=timeout_s)
            if response.status_code != 200:
                return False
            names = [m.get("name", "") for m in response.json().get("models", [])]
            return any(n == self._model or n.split(":")[0] == self._model.split(":")[0] for n in names)
        except Exception:  # noqa: BLE001 - healthcheck never raises
            return False

    def describe(self) -> str:
        return f"ollama:{self._model}"
