"""FastAPI application factory.

Lifespan: build the shared context (SQLite + Chroma + LLM provider), seed
policies idempotently, register middleware. Request ids are generated here
and echoed in every response + error envelope.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import __version__
from .api import routes
from .api.deps import new_request_id
from .auth import AuthError
from .bootstrap import build_context
from .config import Settings, load_settings
from .errors import MSG_INVALID_REQUEST, MSG_RETRY, error_response, unauthorized
from .logging_setup import company_var, request_id_var, setup_logging
from .seed import SEED_COMPANIES, seed_policies

logger = logging.getLogger(__name__)


def _warm_llm(context) -> None:  # noqa: ANN001 - AppContext
    """Load the model into memory in the background so the first real
    request doesn't pay the cold-start cost (Ollama keeps it resident)."""

    import threading

    def warm() -> None:
        try:
            context.llm.complete(
                system="Reply with JSON.",
                user='{"answerable": false, "answer": "", "evidence_ids": []}',
                timeout_s=90.0,
                max_tokens=8,
            )
            logger.info("llm warmed up")
        except Exception as exc:  # noqa: BLE001 - warming is best-effort
            logger.warning("llm warm-up failed: %s", type(exc).__name__)

    threading.Thread(target=warm, daemon=True, name="llm-warmup").start()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(settings.log_level)
        context = build_context(settings)
        app.state.context = context
        for company_id, display_name in SEED_COMPANIES.items():
            context.catalog.ensure_company(company_id, display_name)
        try:
            seed_policies(settings.policies_dir, context.ingest_service)
        except Exception:  # noqa: BLE001 - startup must surface, not hide
            logger.exception("policy seeding failed")
            raise
        logger.info(
            "service ready (llm=%s, embedding=%s, retrieval=%s)",
            context.llm.describe(),
            settings.embedding_provider,
            settings.retrieval_mode,
        )
        _warm_llm(context)
        yield
        context.close()

    app = FastAPI(
        title="Commute Policy Assistant",
        description="Answers employee questions from their own company's current transport policy.",
        version=__version__,
        lifespan=lifespan,
    )
    app.include_router(routes.router)

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # noqa: ANN202
        request_id = new_request_id()
        request.state.request_id = request_id
        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            request_id_var.reset(token)

    @app.middleware("http")
    async def company_log_context(request: Request, call_next):  # noqa: ANN202
        # Attach the verified company (if any) to log lines only.
        try:
            auth = request.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                from .auth import verify_token

                principal = verify_token(app.state.context.jwt_secret, auth[7:].strip())
                company_var.set(principal.company_id)
        except Exception:  # noqa: BLE001 - logging context is best-effort
            pass
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):  # noqa: ANN202
        request_id = getattr(request.state, "request_id", new_request_id())
        return error_response(422, "invalid_request", MSG_INVALID_REQUEST, request_id)

    @app.exception_handler(AuthError)
    async def auth_error_handler(request: Request, exc: AuthError):  # noqa: ANN202
        request_id = getattr(request.state, "request_id", new_request_id())
        return unauthorized(request_id)

    @app.exception_handler(HTTPException)
    async def http_handler(request: Request, exc: HTTPException):  # noqa: ANN202
        request_id = getattr(request.state, "request_id", new_request_id())
        code = {404: "not_found", 403: "forbidden"}.get(exc.status_code, "error")
        return error_response(exc.status_code, code, str(exc.detail), request_id)

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):  # noqa: ANN202
        request_id = getattr(request.state, "request_id", new_request_id())
        logger.exception("unhandled error")
        return error_response(503, "service_unavailable", MSG_RETRY, request_id)

    return app


app = create_app()
