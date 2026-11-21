"""Repeatable JSONL evaluation runner.

Input : eval/cases.jsonl  (immutable gold cases; never modified by runs)
Output: one JSONL line per case with EXACTLY the required fields
        id, input, expected, actual, pass, notes
        plus a metrics summary (eval/summary.json by default).

Modes:
- offline (default): LLM = deterministic extractive fallback -> same results
  on every run, no network; embeddings come from the store in use.
- live: LLM = the configured real provider (Ollama) -> the submitted results.

Scoring is deterministic plain code (no LLM judge):
- answer correctness: covered flag + required/forbidden phrases (any_of
  variants) + correct section citations, resolved per active version when a
  case has by_version expectations.
- not-covered: covered=false, no citations, message points to the desk.
- retrieval recall@3: expected section among the top-3 retrieved sections
  (covered cases only).
- citation validity: 1..3 citations, all from the authenticated company and
  the active version.
- tenant leakage: any citation/chunk from another company or inactive
  version -> counted; must be zero.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
import time
from pathlib import Path

from .bootstrap import build_context
from .config import Settings, load_settings
from .llm.extractive import ExtractiveLLM
from .logging_setup import setup_logging

logger = logging.getLogger(__name__)

OUTPUT_FIELDS = ("id", "input", "expected", "actual", "pass", "notes")


# --- normalisation -----------------------------------------------------------


def normalize(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("p.m.", "pm").replace("a.m.", "am")
    text = " ".join(text.split())
    return text


def phrase_present(answer_norm: str, phrase: str) -> bool:
    return normalize(phrase) in answer_norm


def check_phrases(answer_norm: str, items) -> list[str]:  # noqa: ANN001 - gold spec
    """Returns list of unsatisfied item descriptions."""

    missing = []
    for item in items or []:
        if isinstance(item, dict) and "any_of" in item:
            if not any(phrase_present(answer_norm, v) for v in item["any_of"]):
                missing.append(f"any_of({item['any_of']})")
        elif isinstance(item, str):
            if not phrase_present(answer_norm, item):
                missing.append(item)
    return missing


def forbidden_present(answer_norm: str, phrases) -> list[str]:
    return [p for p in phrases or [] if phrase_present(answer_norm, p)]


# --- expectation resolution ---------------------------------------------------


def resolve_expected(expected: dict, active_version: str) -> dict:
    if "by_version" in expected:
        table = expected["by_version"]
        if active_version in table:
            return table[active_version]
        # Unknown active version: fail visibly instead of silently matching.
        return {"_error": f"no expectations for active version {active_version}"}
    return expected


def score_case(case: dict, actual: dict, active_version: str) -> tuple[bool, list[str]]:
    """Deterministic scorer. Returns (pass, failed_checks)."""

    expected = resolve_expected(case.get("expected", {}), active_version)
    if "_error" in expected:
        return False, [expected["_error"]]

    # A case may accept several valid behaviours (e.g. the injection case:
    # an authenticated-company answer OR an explicit refusal).
    if "any_of_expectations" in expected:
        variants = expected["any_of_expectations"]
        results = [_score_single(case, v, actual, active_version) for v in variants]
        best = min(results, key=lambda r: len(r[1]))
        return best[0], best[1] if not best[0] else ["ok"]
    return _score_single(case, expected, actual, active_version)


def _score_single(case: dict, expected: dict, actual: dict, active_version: str) -> tuple[bool, list[str]]:
    failures: list[str] = []
    kind = expected.get("kind", "answer")
    covered = actual["covered"]
    answer_norm = normalize(actual.get("answer", ""))
    citations = actual.get("citations", [])
    company = case["input"]["company"]

    if kind == "not_covered" or expected.get("covered") is False:
        if covered is not False:
            failures.append("expected covered=false")
        if citations:
            failures.append("not-covered case must have no citations")
        if "transport desk" not in answer_norm:
            failures.append("must point to the transport desk")
        return not failures, failures

    # answer / update kinds
    if covered is not True:
        failures.append("expected covered=true")
        return False, failures
    if not citations:
        failures.append("no citations")
    if len(citations) > 3:
        failures.append("more than 3 citations")
    cite_company = expected.get("cite_company", company)
    for c in citations:
        if c.get("company_id") != cite_company:
            failures.append(f"citation from wrong company: {c.get('chunk_id')}")
        if "versions" in expected and c.get("version") not in expected["versions"]:
            failures.append(f"citation version {c.get('version')} not in {expected['versions']}")
        elif "versions" not in expected and c.get("version") != active_version:
            failures.append(f"citation from inactive version {c.get('version')}")
    if expected.get("sections"):
        cited_sections = {c.get("section", "").split(".")[0] for c in citations}
        if not any(str(s) in cited_sections for s in expected["sections"]):
            failures.append(f"cited sections {sorted(cited_sections)} not in {expected['sections']}")

    failures += [f"missing phrase {m}" for m in check_phrases(answer_norm, expected.get("must_include"))]
    failures += [f"forbidden phrase {p}" for p in forbidden_present(answer_norm, expected.get("must_not_include"))]
    return not failures, failures


def recall_hit(expected: dict, retrieved_sections: list[str]) -> bool | None:
    """True/False for covered cases; None when not applicable."""

    sections = expected.get("recall_sections", expected.get("sections"))
    if not sections:
        return None
    cited = {str(s).split(".")[0] for s in retrieved_sections}
    return any(str(s) in cited for s in sections)


# --- runner --------------------------------------------------------------------


def run_case(case: dict, context, llm_override) -> dict:
    started = time.perf_counter()
    company = case["input"]["company"]
    question = case["input"]["question"]
    kind = case.get("expected", {}).get("kind", "answer")
    error: str | None = None

    ingested = None
    if kind == "update":
        ingest_spec = case["expected"]
        try:
            path = Path(ingest_spec["ingest_file"])
            result = context.ingest_service.ingest(
                company_id=company,
                version_id=ingest_spec["ingest_version"],
                effective_at=ingest_spec["effective_at"],
                content=path.read_bytes(),
                source_name=path.name,
            )
            ingested = result.version_id
        except Exception as exc:  # noqa: BLE001
            error = f"ingestion failed: {type(exc).__name__}"

    active = context.catalog.get_active_version(company)
    active_version = active.version_id if active else "none"

    actual: dict = {
        "covered": None,
        "answer": "",
        "citations": [],
        "retrieved_sections": [],
        "llm_calls": 0,
        "retrieval_latency_ms": None,
        "total_ms": None,
        "active_version": active_version,
        "mode": llm_override,
        "ingested_version": ingested,
        "raw_model_output": "",
    }

    if error is None:
        try:
            answer = context.answer_service.answer(
                question=question,
                company_id=company,
                deadline=time.monotonic() + context.settings.request_budget_s,
            )
            actual.update(
                covered=answer.covered,
                answer=answer.answer,
                citations=[
                    {
                        "company_id": c.company_id,
                        "document": c.document,
                        "version": c.version,
                        "section": c.section,
                        "chunk_id": c.chunk_id,
                    }
                    for c in answer.citations
                ],
                retrieved_sections=answer.evidence_sections,
                llm_calls=answer.llm_calls,
                retrieval_latency_ms=answer.retrieval.latency_ms,
                raw_model_output=answer.raw_model_output,
            )
        except Exception as exc:  # noqa: BLE001 - record the failure honestly
            error = f"answer failed: {type(exc).__name__}"
            actual["answer"] = ""

    actual["total_ms"] = round((time.perf_counter() - started) * 1000, 1)

    if error is not None:
        return {"pass": False, "notes": error, "actual": actual}

    expected = resolve_expected(case.get("expected", {}), active_version)
    ok, failures = score_case(case, actual, active_version)
    notes = "ok" if ok else "; ".join(failures)
    if actual["llm_calls"] > 2:
        ok, notes = False, "call budget exceeded"
    return {"pass": ok, "notes": notes, "actual": actual}


def run_eval(
    *,
    settings: Settings,
    cases_path: Path,
    output_path: Path,
    summary_path: Path | None,
    mode: str,  # "offline" | "live"
    retrieval_mode: str | None,
    with_update: bool,
    limit: int | None,
) -> dict:
    context = build_context(settings)
    from .seed import SEED_COMPANIES, seed_policies

    for company_id, display_name in SEED_COMPANIES.items():
        context.catalog.ensure_company(company_id, display_name)
    seed_policies(settings.policies_dir, context.ingest_service)  # idempotent
    if mode == "offline":
        # Swap the LLM for the deterministic extractor (same retrieval stack).
        from .answering import AnswerService

        context.answer_service = AnswerService(context.retriever, ExtractiveLLM(), settings)
    if retrieval_mode:
        settings.retrieval_mode = retrieval_mode

    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        cases = cases[:limit]

    records = []
    latencies: list[float] = []
    recall_results: list[bool] = []
    citation_valid: list[bool] = []
    tenant_leaks = 0
    passed = 0

    for case in cases:
        if case.get("expected", {}).get("kind") == "update" and not with_update:
            continue
        outcome = run_case(case, context, mode)
        record = {
            "id": case["id"],
            "input": case["input"],
            "expected": case["expected"],
            "actual": outcome["actual"],
            "pass": outcome["pass"],
            "notes": outcome["notes"],
        }
        assert set(record) == set(OUTPUT_FIELDS), "output must have exactly the six fields"
        records.append(record)
        passed += 1 if outcome["pass"] else 0

        actual = outcome["actual"]
        if actual.get("total_ms"):
            latencies.append(actual["total_ms"])
        if actual.get("covered") is True:
            expected = resolve_expected(case.get("expected", {}), actual["active_version"])
            hit = recall_hit(expected, actual.get("retrieved_sections", []))
            if hit is not None:
                recall_results.append(hit)
            citations = actual.get("citations", [])
            company = case["input"]["company"]
            valid = bool(citations) and all(
                c.get("company_id") == company and c.get("version") == actual["active_version"]
                for c in citations
            )
            citation_valid.append(valid)
            tenant_leaks += sum(1 for c in citations if c.get("company_id") != company)
        elif actual.get("covered") is False:
            citation_valid.append(True)  # not-covered: no citations is correct

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = len(records)
    lat_sorted = sorted(latencies) or [0.0]

    def pct(p: float) -> float:
        if not latencies:
            return 0.0
        idx = max(0, math.ceil(p / 100 * len(lat_sorted)) - 1)
        return lat_sorted[idx]

    summary = {
        "mode": mode,
        "llm": "extractive-fallback" if mode == "offline" else context.llm.describe(),
        "retrieval_mode": settings.retrieval_mode,
        "embedding_provider": settings.embedding_provider,
        "cases_total": total,
        "cases_passed": passed,
        "answer_correctness": round(passed / total, 3) if total else None,
        "retrieval_recall_at3": (
            round(sum(recall_results) / len(recall_results), 3) if recall_results else None
        ),
        "citation_validity": (
            round(sum(citation_valid) / len(citation_valid), 3) if citation_valid else None
        ),
        "tenant_leaks": tenant_leaks,
        "latency_ms_p50": round(statistics.median(lat_sorted), 1) if latencies else None,
        "latency_ms_p95": round(pct(95), 1) if latencies else None,
        "output_file": str(output_path),
    }
    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    context.close()
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.eval", description="Run the gold evaluation set")
    parser.add_argument("--cases", type=Path, default=Path("eval/cases.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("eval/results.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("eval/summary.json"))
    parser.add_argument("--mode", choices=("offline", "live"), default="offline")
    parser.add_argument("--retrieval", choices=("semantic", "hybrid"), default=None)
    parser.add_argument("--with-update", action="store_true", help="include the version-update case (mutates active versions)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="isolated store for the run (default: data/eval-runtime, so the API "
        "server's store is never touched; Chroma must not be shared across processes)",
    )
    args = parser.parse_args(argv)

    setup_logging("WARNING")
    settings = load_settings()
    if args.data_dir is not None:
        settings = settings.model_copy(update={"data_dir": args.data_dir.resolve()})
    elif settings.data_dir.name != "eval-runtime":
        settings = settings.model_copy(update={"data_dir": settings.data_dir.parent / "eval-runtime"})
    summary = run_eval(
        settings=settings,
        cases_path=args.cases,
        output_path=args.output,
        summary_path=None if args.no_summary else args.summary,
        mode=args.mode,
        retrieval_mode=args.retrieval,
        with_update=args.with_update,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["cases_passed"] == summary["cases_total"] else 1


if __name__ == "__main__":
    sys.exit(main())
