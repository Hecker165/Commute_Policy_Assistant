"""Ingestion: activation, atomicity, idempotency, failure rollback."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.policies import IngestError, PolicyValidationError

UPDATE_MD = (Path(__file__).resolve().parents[1] / "data" / "policies" / "updates" / "acme-2026.2.md").read_text(
    encoding="utf-8"
)
BASE_MD = (Path(__file__).resolve().parents[1] / "data" / "policies" / "acme-2026.1.md").read_text(encoding="utf-8")

GOOD = """# Commute and Transport Policy

Company: Acme (fictional)

## 1. Rules

Cab pickup is at the gate. The pickup window is 5 minutes.
"""


def _ingest(context, company="acme", version="2026.2", content=GOOD, effective="2026-04-01"):
    return context.ingest_service.ingest(
        company_id=company,
        version_id=version,
        effective_at=effective,
        content=content.encode("utf-8"),
        source_name=f"{company}-{version}.md",
    )


def test_new_version_activates_and_supersedes(seeded_context):
    result = _ingest(seeded_context, version="2026.2", content=UPDATE_MD)
    assert result.superseded_version == "2026.1"
    active = seeded_context.catalog.get_active_version("acme")
    assert active.version_id == "2026.2"
    # only the new version is retrievable
    evidence, _ = seeded_context.retriever.retrieve("escort for a 9:30 PM drop", "acme")
    assert evidence, "new version must answer"
    assert all(e.version_id == "2026.2" for e in evidence)


def test_failed_ingestion_keeps_previous_active_version(seeded_context, monkeypatch):
    def broken_upsert(*args, **kwargs):
        raise RuntimeError("chroma exploded mid-write")

    monkeypatch.setattr(seeded_context.vectordb, "upsert_chunks", broken_upsert)
    with pytest.raises(IngestError):
        _ingest(seeded_context, version="2026.2", content=UPDATE_MD)

    # active pointer untouched; old version still answers
    active = seeded_context.catalog.get_active_version("acme")
    assert active.version_id == "2026.1"
    failed = seeded_context.catalog.get_version("acme", "2026.2")
    assert failed.status == "failed"
    evidence, _ = seeded_context.retriever.retrieve("escort for a 9:30 PM drop", "acme")
    assert all(e.version_id == "2026.1" for e in evidence)
    assert evidence, "old version must remain answerable"


def test_verification_failure_rolls_back(seeded_context, monkeypatch):
    real_count = seeded_context.vectordb.count
    monkeypatch.setattr(
        seeded_context.vectordb,
        "count",
        lambda company, version: real_count(company, version) - 1,  # pretend a vector is missing
    )
    with pytest.raises(IngestError):
        _ingest(seeded_context, version="2026.2", content=UPDATE_MD)
    assert seeded_context.catalog.get_active_version("acme").version_id == "2026.1"


def test_activation_failure_keeps_old_version(seeded_context, monkeypatch):
    def broken_activate(company_id, version_id):
        raise RuntimeError("sqlite busy")

    monkeypatch.setattr(seeded_context.catalog, "activate_version", broken_activate)
    with pytest.raises(IngestError):
        _ingest(seeded_context, version="2026.2", content=UPDATE_MD)
    assert seeded_context.catalog.get_active_version("acme").version_id == "2026.1"


def test_reingest_same_content_is_idempotent(seeded_context):
    first = _ingest(seeded_context, version="2026.2", content=UPDATE_MD)
    assert first.idempotent is False
    count_after_first = seeded_context.vectordb.count("acme", "2026.2")

    second = _ingest(seeded_context, version="2026.2", content=UPDATE_MD)
    assert second.idempotent is True
    assert seeded_context.vectordb.count("acme", "2026.2") == count_after_first
    assert seeded_context.catalog.get_active_version("acme").version_id == "2026.2"


def test_same_version_different_content_rejected(seeded_context):
    _ingest(seeded_context, version="2026.2", content=UPDATE_MD)
    with pytest.raises(PolicyValidationError):
        _ingest(seeded_context, version="2026.2", content=GOOD)


def test_invalid_inputs_rejected(seeded_context):
    with pytest.raises(PolicyValidationError):
        _ingest(seeded_context, version="bad version id!")
    with pytest.raises(PolicyValidationError):
        _ingest(seeded_context, effective="01-04-2026")
    with pytest.raises(PolicyValidationError):
        _ingest(seeded_context, content="")
    with pytest.raises(PolicyValidationError):
        seeded_context.ingest_service.ingest(
            company_id="acme",
            version_id="2026.2",
            effective_at="2026-04-01",
            content=b"\xff\xfe invalid utf8",
            source_name="x.md",
        )


def test_unsupported_extension_rejected(seeded_context):
    with pytest.raises(PolicyValidationError):
        seeded_context.ingest_service.ingest(
            company_id="acme",
            version_id="2026.2",
            effective_at="2026-04-01",
            content=GOOD.encode("utf-8"),
            source_name="policy.pdf",
        )


def test_oversized_upload_rejected(seeded_context):
    with pytest.raises(PolicyValidationError):
        seeded_context.ingest_service.ingest(
            company_id="acme",
            version_id="2026.2",
            effective_at="2026-04-01",
            content=b"# T\n\n## 1. X\n" + b"a" * 3_000_000,
            source_name="x.md",
        )


def test_concurrent_activation_single_winner(seeded_context):
    import threading

    results: list[str] = []
    errors: list[str] = []

    other = GOOD.replace("gate", "gate B")  # different content, same company

    def worker(version: str, content: str) -> None:
        try:
            seeded_context.ingest_service.ingest(
                company_id="acme",
                version_id=version,
                effective_at="2026-05-01",
                content=content.encode("utf-8"),
                source_name="acme-%s.md" % version,
            )
            results.append(version)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{version}: {type(exc).__name__}")

    threads = [
        threading.Thread(target=worker, args=("2027.1", GOOD)),
        threading.Thread(target=worker, args=("2027.2", other)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    active = seeded_context.catalog.get_active_version("acme")
    # Both ingests complete (different versions), but exactly ONE is active.
    assert active.version_id in {"2027.1", "2027.2"}
    rows = seeded_context.catalog._conn.execute(
        "SELECT COUNT(*) AS n FROM policy_versions WHERE company_id='acme' AND status='active'"
    ).fetchone()
    assert rows["n"] == 1
