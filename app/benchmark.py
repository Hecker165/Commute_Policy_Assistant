"""Warm latency benchmark: end-to-end answer time with the real LLM included.

Method: `--warmup` requests warm the model/server, then `--requests`
timed requests go through the SAME in-process pipeline the API uses
(retrieval + LLM call + validation). Prints p50/p95 plus machine info.
No fabricated numbers: failures are counted and reported.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from .bootstrap import build_context
from .config import load_settings
from .logging_setup import setup_logging

WARMUP_QUESTIONS = [
    ("acme", "Do women employees get an escort for a 9:30 PM drop?"),
    ("bluebird", "Can I book a cab on Saturday?"),
    ("copper", "Who approves an airport trip?"),
]
TIMED_QUESTIONS = WARMUP_QUESTIONS + [
    ("acme", "What happens if I no-show twice in a month?"),
    ("acme", "Can I take my dog in the office cab?"),
    ("bluebird", "Who approves an airport trip?"),
    ("copper", "Can I book a cab on Saturday?"),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.benchmark")
    parser.add_argument("--requests", type=int, default=14, help="timed requests after warm-up")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--json", type=str, default=None, help="write results to this file")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="isolated store for the run (default: data/eval-runtime)",
    )
    args = parser.parse_args(argv)

    setup_logging("ERROR")
    settings = load_settings()
    if args.data_dir is not None:
        settings = settings.model_copy(update={"data_dir": args.data_dir.resolve()})
    elif settings.data_dir.name != "eval-runtime":
        settings = settings.model_copy(update={"data_dir": settings.data_dir.parent / "eval-runtime"})
    context = build_context(settings)
    print(f"provider={context.llm.describe()} retrieval={settings.retrieval_mode} "
          f"budget={settings.request_budget_s}s llm_timeout={settings.llm_timeout_s}s")

    failures = 0
    for i in range(args.warmup):
        company, question = WARMUP_QUESTIONS[i % len(WARMUP_QUESTIONS)]
        context.answer_service.answer(
            question=question,
            company_id=company,
            deadline=time.monotonic() + settings.request_budget_s,
        )

    timings: list[float] = []
    for i in range(args.requests):
        company, question = TIMED_QUESTIONS[i % len(TIMED_QUESTIONS)]
        started = time.perf_counter()
        try:
            context.answer_service.answer(
                question=question,
                company_id=company,
                deadline=time.monotonic() + settings.request_budget_s,
            )
        except Exception as exc:  # noqa: BLE001 - count and report failures honestly
            failures += 1
            print(f"  request {i + 1}: FAILED {type(exc).__name__}")
            continue
        timings.append((time.perf_counter() - started) * 1000)

    provider_name = context.llm.describe()
    context.close()

    if not timings:
        print("all requests failed; no timings to report")
        return 1

    ordered = sorted(timings)
    p50 = statistics.median(ordered)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    result = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "provider": provider_name,
        "retrieval_mode": settings.retrieval_mode,
        "model_file": settings.llm_model,
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "processor": platform.processor(),
        },
        "warmup_requests": args.warmup,
        "timed_requests": args.requests,
        "failures": failures,
        "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1),
        "min_ms": round(min(ordered), 1),
        "max_ms": round(max(ordered), 1),
        "target_p95_ms": 3000,
        "meets_target": p95 <= 3000,
    }
    print(json.dumps(result, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(result, indent=2))
    return 0 if result["meets_target"] else 2


if __name__ == "__main__":
    sys.exit(main())
