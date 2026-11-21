"""Policy parsing + deterministic chunking."""

from __future__ import annotations

import pytest

from app.policies import PolicyValidationError, chunk_policy, parse_policy

DOC = """# Commute and Transport Policy

Company: Acme (fictional)

## 1. Purpose and Scope

This policy covers rostered commute.

## 2. Eligibility

Employees in grades L1 to L5 within 15 km are eligible. Interns are not eligible.

## 3. Night Drops

For women employees a security escort is mandatory on any drop after 8:00 PM.
"""

BAD = "just text, no headings at all"
EMPTY = "   "


def test_parse_sections_and_title():
    parsed = parse_policy(DOC)
    assert parsed.title == "Commute and Transport Policy"
    assert [s.id for s in parsed.sections] == ["1", "2", "3"]
    assert parsed.sections[1].label == "2. Eligibility"
    assert "not eligible" in parsed.sections[1].text


def test_parse_rejects_headless_and_empty():
    with pytest.raises(PolicyValidationError):
        parse_policy(BAD)
    with pytest.raises(PolicyValidationError):
        parse_policy(EMPTY)


def test_chunk_ids_deterministic_and_stable():
    parsed = parse_policy(DOC)
    chunks_a = chunk_policy(parsed, company_id="acme", version_id="2026.1", max_chars=700, overlap_chars=120)
    chunks_b = chunk_policy(parsed, company_id="acme", version_id="2026.1", max_chars=700, overlap_chars=120)
    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]
    assert chunks_a[0].chunk_id.startswith("acme:2026.1:s1:")
    # same content, different version -> different ids, no collision
    chunks_v2 = chunk_policy(parsed, company_id="acme", version_id="2026.2", max_chars=700, overlap_chars=120)
    assert {c.chunk_id for c in chunks_v2}.isdisjoint({c.chunk_id for c in chunks_a})


def test_chunking_splits_long_sections_with_overlap():
    long_text = " ".join(f"Sentence number {i} explains a rule in detail." for i in range(60))
    doc = f"# T\n\n## 1. Rules\n\n{long_text}\n"
    parsed = parse_policy(doc)
    chunks = chunk_policy(parsed, company_id="acme", version_id="2026.1", max_chars=500, overlap_chars=80)
    assert len(chunks) > 1
    assert all(len(c.text) <= 500 for c in chunks)
    assert "Sentence number 59" in "".join(c.text for c in chunks)


def test_parse_real_fixture_files():
    from pathlib import Path

    policies_dir = Path(__file__).resolve().parents[1] / "data" / "policies"
    for name in ("acme-2026.1.md", "bluebird-2026.1.md", "copper-2026.1.md"):
        parsed = parse_policy((policies_dir / name).read_text(encoding="utf-8"))
        assert parsed.title == "Commute and Transport Policy"
        assert len(parsed.sections) == 8
