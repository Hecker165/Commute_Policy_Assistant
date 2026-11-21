"""Provider factory."""

from __future__ import annotations

from ..config import Settings
from .base import LLMProvider
from .extractive import ExtractiveLLM
from .fake import ScriptedLLM
from .ollama import OllamaLLM
from .openai_compat import OpenAICompatLLM


def build_llm(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "ollama":
        return OllamaLLM(base_url=settings.ollama_base_url, model=settings.llm_model)
    if settings.llm_provider == "openai":
        api_key = (
            settings.openai_api_key or settings.google_api_key or settings.gemini_api_key
        )
        return OpenAICompatLLM(
            base_url=settings.openai_base_url,
            model=settings.llm_model,
            api_key=api_key,
        )
    if settings.llm_provider == "extractive":
        return ExtractiveLLM()
    if settings.llm_provider == "fake":
        # Bare 'fake' is only useful in tests; a run without a script would
        # fail loudly, which is the correct behaviour for a misconfiguration.
        return ScriptedLLM(responses=[])
    raise ValueError(f"unknown llm provider: {settings.llm_provider}")
