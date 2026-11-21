"""JWT auth: verification, expiry, roles, tenant claim source."""

from __future__ import annotations

import pytest

from app.auth import AuthError, issue_token, verify_token

SECRET = "unit-test-secret"


def _token(**overrides) -> str:
    kwargs = dict(sub="u1", company_id="acme", role="employee", ttl_s=600)
    kwargs.update(overrides)
    return issue_token(SECRET, **kwargs)


def test_valid_token_yields_principal():
    principal = verify_token(SECRET, _token())
    assert principal.company_id == "acme"
    assert principal.role == "employee"
    assert principal.sub == "u1"
    assert principal.is_admin is False


def test_wrong_signature_rejected():
    with pytest.raises(AuthError):
        verify_token("another-secret", _token())


def test_expired_token_rejected():
    import jwt
    import time

    payload = {
        "sub": "u1",
        "company_id": "acme",
        "role": "employee",
        "iat": int(time.time()) - 7200,
        "exp": int(time.time()) - 3600,
    }
    token = jwt.encode(payload, SECRET, algorithm="HS256")
    with pytest.raises(AuthError):
        verify_token(SECRET, token)


def test_alg_none_rejected():
    import jwt

    payload = {
        "sub": "u1",
        "company_id": "acme",
        "role": "employee",
        "iat": 1,
        "exp": 2**31,
    }
    unsigned = jwt.encode(payload, key=None, algorithm="none")
    with pytest.raises(AuthError):
        verify_token(SECRET, unsigned)


def test_missing_claims_rejected():
    import jwt
    import time

    now = int(time.time())
    token = jwt.encode({"sub": "u1", "exp": now + 600}, SECRET, algorithm="HS256")
    with pytest.raises(AuthError):
        verify_token(SECRET, token)


def test_garbage_token_rejected():
    with pytest.raises(AuthError):
        verify_token(SECRET, "not-a-jwt")
    with pytest.raises(AuthError):
        verify_token(SECRET, "")


def test_unknown_role_rejected():
    import time

    import jwt

    payload = {
        "sub": "u1",
        "company_id": "acme",
        "role": "superuser",
        "iat": int(time.time()),
        "exp": int(time.time()) + 600,
    }
    token = jwt.encode(payload, SECRET, algorithm="HS256")
    with pytest.raises(AuthError):
        verify_token(SECRET, token)


def test_admin_flag():
    assert verify_token(SECRET, _token(role="admin")).is_admin is True
