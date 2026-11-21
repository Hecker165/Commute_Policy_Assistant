"""Hosted OpenAI-compatible provider tests (offline, via mock transport)."""

from __future__ import annotations

import json

import httpx
import pytest

from app.config import Settings
from app.llm import build_llm
from app.llm.base import LLMError
from app.llm.openai_compat import OpenAICompatLLM

BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _chat_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": content}}]},
        request=httpx.Request("POST", f"{BASE}chat/completions"),
    )


def test_happy_path_returns_content():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return _chat_response('{"answerable": true, "answer": "ok", "evidence_ids": ["c1"]}')

    llm = OpenAICompatLLM(BASE, "gemini-2.0-flash", "test-key", transport=httpx.MockTransport(handler))
    out = llm.complete(system="s", user="u", timeout_s=1.0, max_tokens=50)
    assert '"answerable": true' in out
    assert calls[0]["model"] == "gemini-2.0-flash"
    assert calls[0]["response_format"] == {"type": "json_object"}


def test_response_format_400_falls_back_to_plain_call():
    """Providers that reject response_format get one retry without it."""

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if "response_format" in body:
            return httpx.Response(
                400,
                json={"error": {"message": "response_format is not supported"}},
                request=request,
            )
        return _chat_response('{"answerable": false, "answer": "", "evidence_ids": []}')

    llm = OpenAICompatLLM(BASE, "gemini-2.0-flash", "test-key", transport=httpx.MockTransport(handler))
    out = llm.complete(system="s", user="u", timeout_s=1.0, max_tokens=50)
    assert '"answerable": false' in out
    assert len(calls) == 2
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]


def test_http_error_maps_to_llm_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad key"}, request=request)

    llm = OpenAICompatLLM(BASE, "gemini-2.0-flash", "bad", transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError):
        llm.complete(system="s", user="u", timeout_s=1.0, max_tokens=50)


def test_missing_key_raises_clear_error(settings):
    provider_settings = settings.model_copy(update={"llm_provider": "openai"})
    with pytest.raises(LLMError, match="no API key"):
        build_llm(provider_settings)


def test_key_aliases_are_accepted(settings):
    for field in ("openai_api_key", "google_api_key", "gemini_api_key"):
        provider_settings = settings.model_copy(
            update={"llm_provider": "openai", field: "alias-key"}
        )
        provider = build_llm(provider_settings)
        assert provider.describe() == f"openai-compat:{settings.llm_model}"
