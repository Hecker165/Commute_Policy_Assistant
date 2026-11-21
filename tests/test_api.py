"""API end-to-end: status codes, envelopes, tenant scoping, timeouts."""

from __future__ import annotations

import json

from app.llm.fake import attack_behaviour

from conftest import auth_headers, make_token

ESCORT_JSON = json.dumps(
    {
        "answerable": True,
        "answer": "A security escort is mandatory for women employees on drops after 8:00 PM.",
        "evidence_ids": ["c1"],
    }
)
NOT_ANSWERABLE = json.dumps({"answerable": False, "answer": "", "evidence_ids": []})

QUESTION = {"question": "Do women employees get an escort for a 9:30 PM drop?"}


def _answer_ok(client, settings):
    client.app.state.context.scripted.responses = [ESCORT_JSON]
    response = client.post("/v1/questions", json=QUESTION, headers=auth_headers(settings))
    assert response.status_code == 200
    return response.json()


def test_healthz(client):
    data = client.get("/healthz").json()
    assert data["status"] in {"ok", "degraded", "starting"}
    assert set(data["components"]) == {"catalog", "vectors", "llm"}


def test_question_requires_token(client):
    response = client.post("/v1/questions", json=QUESTION)
    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == "invalid_token"
    assert body["request_id"]


def test_garbage_token_rejected(client):
    response = client.post(
        "/v1/questions", json=QUESTION, headers={"Authorization": "Bearer junk"}
    )
    assert response.status_code == 401


def test_company_scoped_answer(client, settings):
    data = _answer_ok(client, settings)
    assert data["covered"] is True
    assert data["answer"]
    assert 1 <= len(data["citations"]) <= 3
    citation = data["citations"][0]
    assert citation["document"] == "Commute and Transport Policy"
    assert citation["version"] == "2026.1"
    assert citation["section"].startswith("4.")
    assert citation["chunk_id"].startswith("acme:2026.1:")
    assert data["request_id"]


def test_other_company_token_answers_from_its_own_policy(client, settings):
    client.app.state.context.scripted.responses = [
        json.dumps(
            {
                "answerable": True,
                "answer": "An escort is mandatory on drops after 10:00 PM.",
                "evidence_ids": ["c1"],
            }
        )
    ]
    response = client.post(
        "/v1/questions",
        json={"question": "Do women employees get an escort for a 9:30 PM drop?"},
        headers=auth_headers(settings, company="bluebird"),
    )
    data = response.json()
    assert data["covered"] is True
    assert data["citations"][0]["chunk_id"].startswith("bluebird:2026.1:")


def test_question_body_cannot_override_company(client, settings):
    """A client-sent company_id must be rejected (422), never honoured."""

    response = client.post(
        "/v1/questions",
        json={**QUESTION, "company_id": "bluebird"},
        headers=auth_headers(settings, company="acme"),
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_oversized_question_rejected(client, settings):
    response = client.post(
        "/v1/questions",
        json={"question": "x" * 600},
        headers=auth_headers(settings),
    )
    assert response.status_code == 422


def test_blank_question_rejected(client, settings):
    response = client.post(
        "/v1/questions", json={"question": "   "}, headers=auth_headers(settings)
    )
    assert response.status_code == 422


def test_not_covered_flow(client, settings):
    client.app.state.context.scripted.responses = [NOT_ANSWERABLE]
    response = client.post(
        "/v1/questions",
        json={"question": "Can I bring my pet parrot on the shuttle?"},
        headers=auth_headers(settings),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["covered"] is False
    assert data["citations"] == []
    assert "transport desk" in data["answer"]
    assert client.app.state.context.scripted.calls <= 1


def test_llm_failure_returns_503_retry(client, settings):
    client.app.state.context.scripted.fail_all = True
    response = client.post("/v1/questions", json=QUESTION, headers=auth_headers(settings))
    assert response.status_code == 503
    data = response.json()
    assert data["error"]["code"] in {"timeout", "service_unavailable"}
    assert "try again" in data["error"]["message"].lower()


def test_slow_llm_hits_deadline_returns_503(client, settings):
    # Rebuild context with a tiny budget for a fast test.
    context = client.app.state.context
    context.settings.request_budget_s = 0.3
    context.scripted.delay_s = 2.0
    response = client.post("/v1/questions", json=QUESTION, headers=auth_headers(settings))
    assert response.status_code == 503
    assert "try again" in response.json()["error"]["message"].lower()


def test_admin_endpoints_require_admin_role(client, settings):
    response = client.get("/v1/admin/policies/active", headers=auth_headers(settings, role="employee"))
    assert response.status_code == 403

    response = client.get("/v1/admin/policies/active", headers=auth_headers(settings, role="admin"))
    assert response.status_code == 200
    data = response.json()
    assert data["company_id"] == "acme"
    assert data["version"] == "2026.1"
    assert data["status"] == "active"


def test_admin_ingestion_activates_new_version(client, settings):
    update_path = (
        settings.policies_dir / "updates" / "acme-2026.2.md"
    )
    with update_path.open("rb") as fh:
        response = client.post(
            "/v1/admin/policies",
            files={"file": ("acme-2026.2.md", fh, "text/markdown")},
            data={"version": "2026.2", "effective_at": "2026-04-01"},
            headers=auth_headers(settings, role="admin"),
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"
    assert data["version"] == "2026.2"
    assert data["superseded_version"] == "2026.1"
    assert data["sections"] == 8

    active = client.get(
        "/v1/admin/policies/active", headers=auth_headers(settings, role="admin")
    ).json()
    assert active["version"] == "2026.2"

    # The old rule must stop answering; new rule answers with new citation version.
    client.app.state.context.scripted.responses = [
        json.dumps(
            {
                "answerable": True,
                "answer": "A security escort is mandatory for drops after 7:30 PM.",
                "evidence_ids": ["c1"],
            }
        )
    ]
    result = client.post(
        "/v1/questions",
        json={"question": "Do women employees get an escort for a 9:30 PM drop?"},
        headers=auth_headers(settings),
    ).json()
    assert result["citations"][0]["version"] == "2026.2"


def test_admin_ingestion_bad_file_rejected_and_old_policy_stays(client, settings):
    response = client.post(
        "/v1/admin/policies",
        files={"file": ("policy.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"version": "2026.2", "effective_at": "2026-04-01"},
        headers=auth_headers(settings, role="admin"),
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_policy"

    active = client.get(
        "/v1/admin/policies/active", headers=auth_headers(settings, role="admin")
    ).json()
    assert active["version"] == "2026.1"


def test_employee_cannot_ingest(client, settings):
    response = client.post(
        "/v1/admin/policies",
        files={"file": ("x.md", b"# T\n\n## 1. A\nText", "text/markdown")},
        data={"version": "2026.2", "effective_at": "2026-04-01"},
        headers=auth_headers(settings, role="employee"),
    )
    assert response.status_code == 403


def test_admin_ingestion_scoped_to_admins_company(client, settings):
    """A bluebird admin cannot touch acme's policy: tenant comes from the JWT."""

    update_path = settings.policies_dir / "updates" / "acme-2026.2.md"
    with update_path.open("rb") as fh:
        response = client.post(
            "/v1/admin/policies",
            files={"file": ("acme-2026.2.md", fh, "text/markdown")},
            data={"version": "2026.2", "effective_at": "2026-04-01"},
            headers=auth_headers(settings, company="bluebird", role="admin"),
        )
    assert response.status_code == 200
    assert response.json()["company_id"] == "bluebird"
    # acme untouched
    acme_active = client.get(
        "/v1/admin/policies/active", headers=auth_headers(settings, company="acme", role="admin")
    ).json()
    assert acme_active["version"] == "2026.1"
