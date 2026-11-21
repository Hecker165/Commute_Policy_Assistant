"""Stable error envelope shared by every failure path.

Every non-200 response uses {"error": {"code", "message"}, "request_id"}.
Messages are fixed strings: they never echo stack traces, tenant data or
model output back to the caller.
"""

from __future__ import annotations

from typing import Any

from fastapi import status
from fastapi.responses import JSONResponse


class ApiError(Exception):
    """Internal error carrying a stable HTTP code + envelope code."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def error_response(
    status_code: int,
    code: str,
    message: str,
    request_id: str,
    extra: dict[str, Any] | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {
        "request_id": request_id,
        "error": {"code": code, "message": message},
    }
    if extra:
        body.update(extra)
    return JSONResponse(status_code=status_code, content=body)


# Fixed user-facing messages: one wording everywhere, no model-authored error text.
MSG_NOT_COVERED = (
    "Your company's current transport policy does not cover this, so I can't "
    "answer it from the policy. Please contact the transport desk for help."
)
MSG_RETRY = (
    "We couldn't answer right now. Please try again in a moment, or contact "
    "the transport desk."
)
MSG_INVALID_REQUEST = "The request was invalid."
MSG_UNAUTHORIZED = "A valid token is required."
MSG_FORBIDDEN = "Your role does not allow this action."


def unauthorized(request_id: str) -> JSONResponse:
    return error_response(status.HTTP_401_UNAUTHORIZED, "invalid_token", MSG_UNAUTHORIZED, request_id)


def forbidden(request_id: str) -> JSONResponse:
    return error_response(status.HTTP_403_FORBIDDEN, "forbidden", MSG_FORBIDDEN, request_id)
