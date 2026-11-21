"""Typed application configuration.

All knobs are environment variables (or a local .env file); every value has a
fixed default so the service starts with zero configuration. Nothing here is
read from the LLM or from requests: the model never influences configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Hard limits required by the case study. These are constants, not settings:
# a request must never be able to raise its own LLM call or citation budget.
MAX_LLM_CALLS_PER_REQUEST = 2
MAX_CITED_CHUNKS = 3

RETRIEVAL_MODE = Literal["semantic", "hybrid"]
LLM_PROVIDERS = Literal["ollama", "fake", "extractive", "openai"]
EMBEDDING_PROVIDERS = Literal["default", "hash"]


class Settings(BaseSettings):
    """Environment-driven settings with safe defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- storage -----------------------------------------------------------
    data_dir: Path = Path("./data/runtime")
    policies_dir: Path = Path("./data/policies")

    # --- auth --------------------------------------------------------------
    # JWT_SECRET (env) wins; otherwise a random secret is created once at
    # DATA_DIR/jwt_secret and reused. The secret never lives in source code.
    jwt_secret: str | None = None
    token_ttl_s: int = 8 * 3600

    # --- embedding / retrieval ---------------------------------------------
    embedding_provider: EMBEDDING_PROVIDERS = "default"  # default = Chroma ONNX MiniLM-L6-v2
    retrieval_mode: RETRIEVAL_MODE = "semantic"  # "hybrid" = the measured bonus
    retrieval_k: int = 4  # evidence candidates fetched from the vector store
    evidence_top_k: int = 3  # chunks handed to the LLM (== max citations)
    retrieval_min_similarity: float = 0.30  # cosine similarity floor for semantic hits
    lexical_candidates: int = 6  # lexical list size used by hybrid fusion
    rrf_k: int = 60  # reciprocal-rank-fusion constant

    # --- LLM ----------------------------------------------------------------
    llm_provider: LLM_PROVIDERS = "ollama"
    llm_model: str = "llama3.2:3b"  # measured: warm answers in 1-2s on 12 CPUs, best small-model reasoning
    ollama_base_url: str = "http://127.0.0.1:11434"
    openai_base_url: str = "https://api.groq.com/openai/v1"
    openai_api_key: str = ""  # optional adapter only; empty by default, never committed
    llm_timeout_s: float = 2.2  # per-call cap, shorter than the request budget
    llm_max_tokens: int = 220
    llm_temperature: float = 0.0

    # --- request budgets -----------------------------------------------------
    # Overall deadline for a question is REQUEST_BUDGET_S; p95 target is 3s.
    request_budget_s: float = 2.7
    max_question_chars: int = 500
    max_upload_bytes: int = 2_000_000

    # --- chunking -------------------------------------------------------------
    chunk_max_chars: int = 700
    chunk_overlap_chars: int = 120

    # --- misc -----------------------------------------------------------------
    log_level: str = "INFO"
    version: str = Field(default="1.0.0", exclude=True)

    @field_validator("data_dir", "policies_dir")
    @classmethod
    def _resolve(cls, v: Path) -> Path:
        return Path(v).resolve()

    @property
    def jwt_secret_file(self) -> Path:
        return self.data_dir / "jwt_secret"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def catalog_path(self) -> Path:
        return self.data_dir / "catalog.sqlite3"

    @property
    def model_cache_dir(self) -> Path:
        """Shared embedding-model cache, independent of the data dir, so the
        API runtime and isolated eval/benchmark stores download the model once."""

        return self.data_dir.parent / "model_cache"


def load_settings() -> Settings:
    """Load settings; give an actionable error when required dirs are bad."""

    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
