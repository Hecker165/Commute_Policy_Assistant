"""JWT authentication: verification lives entirely in plain code.

The company and role are read ONLY from the verified token claims. Nothing
from a request body can ever change them. HS256 only; expired, malformed,
unsigned or wrong-signature tokens are rejected.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from pathlib import Path

import jwt

from .config import Settings

ALGORITHM = "HS256"
ROLES = ("employee", "admin")
REQUIRED_CLAIMS = ("sub", "company_id", "role", "iat", "exp")


class AuthError(Exception):
    """Raised for any token problem; callers map this to HTTP 401."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"invalid token: {reason}")
        self.reason = reason


@dataclass(frozen=True)
class Principal:
    sub: str
    company_id: str
    role: str
    iat: int
    exp: int

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def ensure_jwt_secret(settings: Settings) -> str:
    """Return the signing secret, creating a random local one on first use.

    Precedence: JWT_SECRET env (tests/local overrides) > secret file in the
    data dir. The file is created with restrictive permissions and is
    git-ignored; the secret is never logged.
    """

    if settings.jwt_secret:
        return settings.jwt_secret
    path: Path = settings.jwt_secret_file
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        secret = path.read_text(encoding="utf-8").strip()
        if secret:
            return secret
    secret = secrets.token_urlsafe(48)
    path.write_text(secret, encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:  # pragma: no cover - Windows ignores POSIX modes
        pass
    return secret


def issue_token(
    secret: str,
    *,
    sub: str,
    company_id: str,
    role: str,
    ttl_s: int,
) -> str:
    """Issue a short-lived demo token (CLI helper only; not an HTTP API)."""

    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    if not company_id or not sub:
        raise ValueError("sub and company_id are required")
    import time

    now = int(time.time())
    payload = {
        "sub": sub,
        "company_id": company_id,
        "role": role,
        "iat": now,
        "exp": now + int(ttl_s),
        "iss": "commute-policy-assistant",
    }
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def verify_token(secret: str, token: str) -> Principal:
    """Verify signature, expiry and required claims; return a Principal."""

    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=[ALGORITHM],  # rejects alg=none / alg confusion
            options={
                "require": list(REQUIRED_CLAIMS),
                "verify_signature": True,
                "verify_exp": True,
                "verify_iat": True,
            },
            leeway=0,
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("expired") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError(type(exc).__name__) from exc

    role = payload.get("role")
    if role not in ROLES:
        raise AuthError("unknown role")
    if not isinstance(payload.get("company_id"), str) or not payload["company_id"]:
        raise AuthError("missing company_id")
    if not isinstance(payload.get("sub"), str) or not payload["sub"]:
        raise AuthError("missing sub")
    return Principal(
        sub=payload["sub"],
        company_id=payload["company_id"],
        role=role,
        iat=int(payload["iat"]),
        exp=int(payload["exp"]),
    )


def principal_from_json(raw: str) -> Principal:
    """Test helper: build a Principal directly (bypasses JWT, used by fakes)."""

    data = json.loads(raw)
    return Principal(
        sub=data["sub"],
        company_id=data["company_id"],
        role=data["role"],
        iat=data.get("iat", 0),
        exp=data.get("exp", 0),
    )
