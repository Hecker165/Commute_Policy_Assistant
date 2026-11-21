"""Shared test fixtures.

Tests are fully offline: EMBEDDING_PROVIDER=hash (deterministic hashed
n-gram vectors, no model download), a ScriptedLLM stands in for the model,
and every test gets an isolated temporary SQLite + Chroma store.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("EMBEDDING_PROVIDER", "hash")
os.environ.setdefault("RETRIEVAL_MIN_SIMILARITY", "0.10")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("JWT_SECRET", "test-signing-secret")

# Tests must be hermetic: a developer machine may carry hosted-provider keys
# in its environment; pop them so key-handling tests exercise the empty case.
for _key in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
    os.environ.pop(_key, None)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.answering import AnswerService  # noqa: E402
from app.auth import issue_token  # noqa: E402
from app.bootstrap import build_context  # noqa: E402
from app.config import Settings  # noqa: E402
from app.llm.fake import ScriptedLLM, attack_behaviour  # noqa: E402
from app.main import create_app  # noqa: E402
from app.seed import seed_policies  # noqa: E402


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    # _env_file=None keeps tests hermetic: a developer's local .env (e.g. with a
    # hosted API key) must never leak into test configuration.
    return Settings(
        data_dir=tmp_path / "data",
        policies_dir=PROJECT_ROOT / "data" / "policies",
        jwt_secret="test-signing-secret",
        llm_provider="fake",
        request_budget_s=5.0,
        llm_timeout_s=4.0,
        _env_file=None,  # type: ignore[call-arg]
    )


@pytest.fixture()
def context(settings: Settings):
    ctx = build_context(settings)
    scripted = ScriptedLLM(responses=[])
    ctx.llm = scripted
    ctx.answer_service = AnswerService(ctx.retriever, scripted, settings)
    ctx.scripted = scripted  # type: ignore[attr-defined]
    yield ctx
    ctx.close()


@pytest.fixture()
def seeded_context(context):
    seed_policies(context.settings.policies_dir, context.ingest_service)
    return context


def make_token(settings: Settings, company: str = "acme", role: str = "employee") -> str:
    return issue_token(
        settings.jwt_secret or os.environ["JWT_SECRET"],
        sub="test-user",
        company_id=company,
        role=role,
        ttl_s=3600,
    )


@pytest.fixture()
def client(settings: Settings):
    """API test client with a scripted LLM swapped into the live context."""

    scripted = ScriptedLLM(responses=[])
    app = create_app(settings)
    with TestClient(app) as test_client:
        ctx = test_client.app.state.context
        ctx.llm = scripted
        ctx.answer_service = AnswerService(ctx.retriever, scripted, settings)
        ctx.scripted = scripted  # type: ignore[attr-defined]
        yield test_client


def auth_headers(settings: Settings, company: str = "acme", role: str = "employee") -> dict:
    return {"Authorization": f"Bearer {make_token(settings, company, role)}"}


__all__ = [
    "settings",
    "context",
    "seeded_context",
    "client",
    "make_token",
    "auth_headers",
    "attack_behaviour",
    "ScriptedLLM",
    "PROJECT_ROOT",
]
