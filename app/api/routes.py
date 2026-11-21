"""HTTP routes: health, question answering, admin ingestion, active policy.

Failure mapping (stable envelopes from app.errors):
- 401 invalid/expired token, 403 wrong role, 422 invalid request
- 503 for retrieval/LLM timeout or dependency failure: a plain "try again"
  message, never a hang or a partial answer.
"""

from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile

from ..answering import AnswerPipelineError
from ..bootstrap import AppContext, monotonic_deadline
from ..config import MAX_LLM_CALLS_PER_REQUEST
from ..errors import (
    ApiError,
    MSG_RETRY,
    error_response,
)
from .deps import admin_role, current_principal, get_context, new_request_id
from .schemas import (
    ActivePolicyResponse,
    AnswerResponse,
    CitationOut,
    HealthResponse,
    IngestResponse,
    QuestionRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/healthz", response_model=HealthResponse)
def healthz(context: AppContext = Depends(get_context)) -> HealthResponse:
    components = {
        "catalog": "ok" if context.catalog.healthcheck() else "error",
        "vectors": "ok" if context.vectordb.healthcheck() else "error",
        "llm": "ok" if context.llm.healthcheck(timeout_s=0.6) else "unavailable",
    }
    status = "ok" if all(v == "ok" for v in components.values()) else (
        "degraded" if "error" in components.values() else "starting"
    )
    return HealthResponse(
        status=status,
        components=components,
        version=context.settings.version,
    )


@router.post("/v1/questions", response_model=AnswerResponse)
async def ask_question(
    request: Request,
    body: QuestionRequest,
    context: AppContext = Depends(get_context),
    principal=Depends(current_principal),
) -> AnswerResponse:
    request_id = getattr(request.state, "request_id", new_request_id())
    started = time.perf_counter()

    if len(body.question) > context.settings.max_question_chars:
        return error_response(
            422, "invalid_request", "question exceeds the length limit", request_id
        )

    settings = context.settings
    deadline = monotonic_deadline(settings)
    try:
        # Blocking pipeline in a worker thread, hard-capped by the request
        # budget so a stalled dependency can never hang the client.
        result = await asyncio.wait_for(
            asyncio.to_thread(
                context.answer_service.answer,
                question=body.question,
                company_id=principal.company_id,
                deadline=deadline,
            ),
            timeout=settings.request_budget_s,
        )
    except (asyncio.TimeoutError, TimeoutError):
        logger.warning("request deadline exceeded")
        return error_response(503, "timeout", MSG_RETRY, request_id)
    except AnswerPipelineError as exc:
        logger.warning("answer pipeline failed: %s", exc)
        return error_response(503, "service_unavailable", MSG_RETRY, request_id)
    except ApiError as exc:
        return error_response(exc.status_code, exc.code, exc.message, request_id)

    logger.info(
        "question answered covered=%s citations=%s llm_calls=%s retrieval_ms=%s total_ms=%s",
        result.covered,
        len(result.citations),
        min(result.llm_calls, MAX_LLM_CALLS_PER_REQUEST),
        result.retrieval.latency_ms,
        round((time.perf_counter() - started) * 1000, 1),
    )
    return AnswerResponse(
        request_id=request_id,
        covered=result.covered,
        answer=result.answer,
        citations=[
            CitationOut(
                document=c.document,
                version=c.version,
                section=c.section,
                chunk_id=c.chunk_id,
            )
            for c in result.citations
        ],
    )


@router.post("/v1/admin/policies", response_model=IngestResponse)
async def ingest_policy(
    request: Request,
    file: UploadFile,
    version: str = Form(...),
    effective_at: str = Form(...),
    context: AppContext = Depends(get_context),
    principal=Depends(admin_role),
) -> IngestResponse:
    request_id = getattr(request.state, "request_id", new_request_id())
    content = await file.read()
    from ..policies import IngestError, PolicyValidationError

    try:
        result = await asyncio.to_thread(
            context.ingest_service.ingest,
            company_id=principal.company_id,  # tenant comes from the JWT only
            version_id=version,
            effective_at=effective_at,
            content=content,
            source_name=file.filename or "",
        )
    except (PolicyValidationError, IngestError) as exc:
        # Rejected input; the previously active policy is untouched.
        logger.warning("ingestion rejected: %s", type(exc).__name__)
        return error_response(422, "invalid_policy", str(exc), request_id)
    except Exception:  # noqa: BLE001 - unexpected dependency failure
        logger.exception("ingestion failed")
        return error_response(503, "ingestion_failed", MSG_RETRY, request_id)

    logger.info(
        "policy ingested version=%s sections=%s chunks=%s superseded=%s",
        result.version_id,
        result.section_count,
        result.chunk_count,
        result.superseded_version,
    )
    return IngestResponse(
        request_id=request_id,
        company_id=result.company_id,
        document_title=result.document_title,
        version=result.version_id,
        effective_at=result.effective_at,
        status="active",
        sections=result.section_count,
        chunks=result.chunk_count,
        superseded_version=result.superseded_version,
    )


@router.get("/v1/admin/policies/active", response_model=ActivePolicyResponse)
def active_policy(
    context: AppContext = Depends(get_context),
    principal=Depends(admin_role),
) -> ActivePolicyResponse:
    row = context.catalog.get_active_version(principal.company_id)
    if row is None:
        raise HTTPException(status_code=404, detail="no active policy for this company")
    chunks = context.catalog.count_chunks(principal.company_id, row.version_id)
    return ActivePolicyResponse(
        company_id=row.company_id,
        document_title=row.title,
        version=row.version_id,
        effective_at=row.effective_at,
        status=row.status,
        sections=row.section_count,
        chunks=chunks,
        activated_at=row.activated_at,
    )
