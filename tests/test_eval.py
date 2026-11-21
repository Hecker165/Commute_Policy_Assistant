"""Evaluation runner: six-field output, determinism, scoring correctness."""

from __future__ import annotations

import json
from pathlib import Path

from app.eval import run_eval

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES = PROJECT_ROOT / "eval" / "cases.jsonl"

REQUIRED_FIELDS = {"id", "input", "expected", "actual", "pass", "notes"}


def _run(settings, tmp_path, with_update=False, tag="a"):
    summary = run_eval(
        settings=settings,
        cases_path=CASES,
        output_path=tmp_path / f"results-{tag}.jsonl",
        summary_path=tmp_path / f"summary-{tag}.json",
        mode="offline",
        retrieval_mode=None,
        with_update=with_update,
        limit=None,
    )
    records = [
        json.loads(line)
        for line in (tmp_path / f"results-{tag}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return summary, records


def test_eval_output_has_exactly_six_fields(settings, tmp_path):
    summary, records = _run(settings, tmp_path)
    assert records, "must produce records"
    assert summary["cases_total"] == len(records)
    for record in records:
        assert set(record) == REQUIRED_FIELDS


def test_offline_run_is_deterministic(settings, tmp_path):
    first_summary, first_records = _run(settings, tmp_path, tag="a")
    second_summary, second_records = _run(settings, tmp_path, tag="b")
    assert first_summary["cases_passed"] == second_summary["cases_passed"]
    assert [r["pass"] for r in first_records] == [r["pass"] for r in second_records]
    assert [r["actual"]["answer"] for r in first_records] == [
        r["actual"]["answer"] for r in second_records
    ]


def test_eval_scores_and_no_tenant_leaks(settings, tmp_path):
    summary, _records = _run(settings, tmp_path)
    assert summary["answer_correctness"] is not None
    assert summary["retrieval_recall_at3"] is not None
    assert summary["citation_validity"] is not None
    assert summary["tenant_leaks"] == 0, "cross-company leakage must be zero"
    assert summary["mode"] == "offline"


def test_gold_cases_meet_the_brief():
    cases = [
        json.loads(line)
        for line in CASES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(cases) >= 10
    companies = {c["input"]["company"] for c in cases}
    assert companies == {"acme", "bluebird", "copper"}
    kinds = [c.get("expected", {}).get("kind", "answer") for c in cases]
    assert "not_covered" in kinds
    assert "update" in kinds
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "case ids must be unique"
    assert any("paraphrase" in i for i in ids)
    assert sum(i.startswith("trap-") for i in ids) >= 2
    assert any("injection" in i for i in ids)
    for case in cases:
        assert set(case) >= {"id", "input", "expected"}
        assert set(case["input"]) == {"company", "question"}
