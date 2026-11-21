"""CLI: issue demo tokens and ingest policy updates.

`ingest` prefers the RUNNING SERVER's admin API (minting a local admin token
from the shared secret) so the server stays the single writer to the vector
store — Chroma is not safe to write from two processes at once. Only when no
server is listening does it use the same IngestService locally.

Examples:
    python -m app.cli issue-token --company acme --role employee
    python -m app.cli ingest --company acme --version 2026.2 \\
        --file data/policies/updates/acme-2026.2.md --effective-at 2026-04-01
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .bootstrap import build_context
from .auth import issue_token
from .config import load_settings
from .logging_setup import setup_logging
from .seed import SEED_COMPANIES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.cli", description="Commute Policy Assistant CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    token = sub.add_parser("issue-token", help="print a local demo JWT")
    token.add_argument("--company", required=True, choices=sorted(SEED_COMPANIES))
    token.add_argument("--role", required=True, choices=("employee", "admin"))
    token.add_argument("--sub", default="demo-user")
    token.add_argument("--ttl", type=int, default=8 * 3600)

    ingest = sub.add_parser("ingest", help="ingest a new policy version (activates it)")
    ingest.add_argument("--company", required=True, choices=sorted(SEED_COMPANIES))
    ingest.add_argument("--version", required=True)
    ingest.add_argument("--file", required=True, type=Path)
    ingest.add_argument("--effective-at", required=True)
    ingest.add_argument(
        "--server",
        default=os.environ.get("API_BASE_URL", "http://127.0.0.1:8000"),
        help="running API to proxy through (default: http://127.0.0.1:8000)",
    )

    args = parser.parse_args(argv)
    settings = load_settings()
    setup_logging(settings.log_level)

    if args.command == "issue-token":
        context = build_context(settings)
        try:
            print(
                issue_token(
                    context.jwt_secret,
                    sub=args.sub,
                    company_id=args.company,
                    role=args.role,
                    ttl_s=args.ttl,
                )
            )
        finally:
            context.close()
        return 0

    if args.command == "ingest":
        path: Path = args.file
        if not path.exists():
            print(f"file not found: {path}", file=sys.stderr)
            return 2
        return _ingest(args, path, settings)

    return 2


def _ingest(args, path: Path, settings) -> int:  # noqa: ANN001 - argparse namespace / Settings
    """Proxy through the running server when one is listening (single-writer
    discipline); otherwise ingest directly with the same service."""

    admin_token = issue_token(
        _local_secret(settings),
        sub="cli",
        company_id=args.company,
        role="admin",
        ttl_s=60,
    )
    try:
        import httpx

        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{args.server.rstrip('/')}/v1/admin/policies",
                files={"file": (path.name, path.read_bytes(), "text/markdown")},
                data={"version": args.version, "effective_at": args.effective_at},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        if response.status_code == 200:
            data = response.json()
            print(
                f"activated {data['company_id']} v{data['version']}: "
                f"{data['sections']} sections, {data['chunks']} chunks, "
                f"superseded={data.get('superseded_version') or 'none'}"
            )
            return 0
        detail = response.json().get("error", {}).get("message", response.text)
        if response.status_code in (401, 403):
            detail += (
                " (the server on this port signs tokens with a different secret than the "
                "local DATA_DIR; pass --server/--api URL of the matching instance or stop it)"
            )
        print(f"ingestion rejected by the API ({response.status_code}): {detail}", file=sys.stderr)
        return 1
    except (httpx.ConnectError, httpx.ConnectTimeout):
        pass  # no server listening: safe to write the store directly
    except Exception as exc:  # noqa: BLE001 - server rejected the upload
        print(f"ingestion failed via API: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    context = build_context(settings)
    try:
        result = context.ingest_service.ingest(
            company_id=args.company,
            version_id=args.version,
            effective_at=args.effective_at,
            content=path.read_bytes(),
            source_name=path.name,
        )
    except Exception as exc:  # validation/ingest errors are safe to print
        print(f"ingestion failed: {exc}", file=sys.stderr)
        return 1
    finally:
        context.close()
    print(
        f"activated {args.company} v{result.version_id}: "
        f"{result.section_count} sections, {result.chunk_count} chunks, "
        f"superseded={result.superseded_version or 'none'}"
    )
    return 0


def _local_secret(settings) -> str:  # noqa: ANN001
    from .auth import ensure_jwt_secret

    return ensure_jwt_secret(settings)


if __name__ == "__main__":
    raise SystemExit(main())
