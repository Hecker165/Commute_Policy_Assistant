"""Tenant-safe retrieval: isolation, inactive versions, thresholds, limits."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.policies import parse_policy, chunk_policy

UPDATE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "policies" / "updates" / "acme-2026.2.md"
)

BLUEBIRD_TEMPTING = """# Commute and Transport Policy

Company: Bluebird (fictional)

## 4. Night Drops and Security Escorts

For women employees a security escort is mandatory on any drop after 8:00 PM. Escort details are identical to what you would expect: security escort after 8:00 PM for a 9:30 PM drop.
"""


def test_retrieval_only_returns_active_company_version(seeded_context):
    for company in ("acme", "bluebird", "copper"):
        evidence, stats = seeded_context.retriever.retrieve(
            "security escort for a 9:30 PM drop", company
        )
        assert evidence, f"{company} should retrieve its night-drop section"
        assert all(e.company_id == company for e in evidence)
        assert all(e.version_id == "2026.1" for e in evidence)
        assert stats.k <= seeded_context.settings.retrieval_k


def test_cross_company_chunk_never_leaks(seeded_context):
    """Stage vectors for another company (and an inactive version) whose text
    mimics the query closely; the filters must keep them unreachable."""

    parsed = parse_policy(BLUEBIRD_TEMPTING)
    chunks = chunk_policy(
        parsed, company_id="bluebird", version_id="2099.1", max_chars=700, overlap_chars=120
    )
    from app.catalog import ChunkRow

    seeded_context.catalog.replace_chunks(
        "bluebird",
        "2099.1",
        [
            ChunkRow(
                chunk_id=c.chunk_id,
                company_id=c.company_id,
                version_id=c.version_id,
                section_id=c.section_id,
                section_label=c.section_label,
                document_title=c.document_title,
                chunk_index=c.chunk_index,
                text=c.text,
            )
            for c in chunks
        ],
    )
    seeded_context.vectordb.upsert_chunks(company_id="bluebird", version_id="2099.1", chunks=chunks)
    assert seeded_context.catalog.get_active_version("bluebird").version_id == "2026.1"

    for company in ("acme", "copper"):
        evidence, _ = seeded_context.retriever.retrieve(
            "escort after 8:00 PM for a 9:30 PM drop", company
        )
        assert evidence
        assert all(e.company_id == company for e in evidence)
        assert all(not e.chunk_id.startswith("bluebird:") for e in evidence)
    # Even within bluebird, the staged inactive version must not answer.
    evidence, _ = seeded_context.retriever.retrieve(
        "escort after 8:00 PM for a 9:30 PM drop", "bluebird"
    )
    assert all(not e.chunk_id.startswith("bluebird:2099.1") for e in evidence)
    assert all(e.version_id == "2026.1" for e in evidence)


def test_inactive_version_never_retrieved(seeded_context):
    seeded_context.ingest_service.ingest(
        company_id="acme",
        version_id="2026.2",
        effective_at="2026-04-01",
        content=UPDATE_PATH.read_bytes(),
        source_name="acme-2026.2.md",
    )
    evidence, _ = seeded_context.retriever.retrieve("security escort for a 9:30 PM drop", "acme")
    assert evidence
    assert all(e.version_id == "2026.2" for e in evidence), "only the active version may answer"


def test_no_active_version_returns_empty(seeded_context):
    evidence, _ = seeded_context.retriever.retrieve("anything at all", "unknown-company")
    assert evidence == []


def test_similarity_threshold_filters_noise(seeded_context):
    settings = seeded_context.settings
    original = settings.retrieval_min_similarity
    settings.retrieval_min_similarity = 0.999  # impossibly high
    try:
        evidence, _ = seeded_context.retriever.retrieve("escort for a 9:30 PM drop", "acme")
        assert evidence == []
    finally:
        settings.retrieval_min_similarity = original


def test_hybrid_mode_respects_tenant_filter(seeded_context):
    seeded_context.settings.retrieval_mode = "hybrid"
    for company in ("acme", "bluebird", "copper"):
        evidence, _ = seeded_context.retriever.retrieve(
            "who approves an airport trip", company
        )
        assert evidence, f"hybrid mode should retrieve for {company}"
        assert all(e.company_id == company for e in evidence)
    seeded_context.settings.retrieval_mode = "semantic"
