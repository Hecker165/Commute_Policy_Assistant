"""FastAPI dependencies: request ids, auth, app context access."""

from __future__ import annotations

import uuid

from fastapi import Depends, Request

from ..auth import AuthError, Principal, ROLES, verify_token
from ..bootstrap import AppContext
from ..errors import MSG_UNAUTHORIZED

_MISSING = object()


def get_context(request: Request) -> AppContext:
    return request.app.state.context  # type: ignore[no-any-return]


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def current_principal(request: Request) -> Principal:
    """Verify the bearer token; company/role come ONLY from verified claims."""

    context: AppContext = request.app.state.context
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        raise AuthError("missing bearer token")
    token = header[len("Bearer ") :].strip()
    if not token:
        raise AuthError("empty bearer token")
    return verify_token(context.jwt_secret, token)


def require_role(*allowed: str):
    """Dependency factory: 403 when the verified role is not allowed."""

    allowed_set = set(allowed)

    def dependency(principal: Principal = Depends(current_principal)) -> Principal:
        if principal.role not in allowed_set:
            from fastapi import HTTPException

            raise HTTPException(status_code=403, detail=MSG_UNAUTHORIZED)
        return principal

    return dependency


employee_role = require_role("employee", "admin")
admin_role = require_role("admin")

__all__ = [
    "get_context",
    "new_request_id",
    "current_principal",
    "require_role",
    "employee_role",
    "admin_role",
    "ROLES",
]
