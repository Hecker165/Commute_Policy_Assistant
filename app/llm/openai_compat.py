"""Optional OpenAI-compatible provider (e.g. Groq free tier).

Disabled by default; used only when LLM_PROVIDER=openai and OPENAI_API_KEY
is present in the environment. Never commit the key.
"""

from __future__ import annotations

import httpx

from .base import LLMError, LLMTimeout


class OpenAICompatLLM:
    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set")
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(5.0, connect=2.0),
        )
        self._model = model

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
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if force_json:
            payload["response_format"] = {"type": "json_object"}
        try:
            response = self._client.post("/chat/completions", json=payload, timeout=timeout_s)
        except httpx.TimeoutException as exc:
            raise LLMTimeout(f"openai-compatible call exceeded {timeout_s}s") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"openai-compatible call failed: {type(exc).__name__}") from exc
        if response.status_code != 200:
            raise LLMError(f"openai-compatible returned HTTP {response.status_code}")
        try:
            return response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError) as exc:
            raise LLMError("openai-compatible response shape unexpected") from exc

    def healthcheck(self, timeout_s: float = 1.0) -> bool:
        try:
            return self._client.get("/models", timeout=timeout_s).status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def describe(self) -> str:
        return f"openai-compat:{self._model}"
