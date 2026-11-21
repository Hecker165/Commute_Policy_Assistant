"""CLI + API hardening smoke tests (token issuance, secret handling)."""

from __future__ import annotations

import json

from app.auth import verify_token
from app.cli import main as cli_main


def test_cli_issue_token_roundtrip(settings, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "cli-data"))
    exit_code = cli_main(
        ["issue-token", "--company", "acme", "--role", "employee", "--sub", "emp-42"]
    )
    assert exit_code == 0
    token = capsys.readouterr().out.strip()
    principal = verify_token(settings.jwt_secret, token)
    assert principal.company_id == "acme"
    assert principal.role == "employee"
    assert principal.sub == "emp-42"


CLI_ARGS = [
    "ingest",
    "--server",
    "http://127.0.0.1:59999",  # nothing listens here in tests -> direct path
]


def test_cli_ingest_activates_version(settings, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "cli-data"))
    update = settings.policies_dir / "updates" / "acme-2026.2.md"
    exit_code = cli_main(
        CLI_ARGS
        + [
            "--company",
            "acme",
            "--version",
            "2026.2",
            "--file",
            str(update),
            "--effective-at",
            "2026-04-01",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "activated acme v2026.2" in out
    assert "superseded=2026.1" not in out  # nothing seeded before, so nothing superseded
    assert "superseded=none" in out


def test_cli_ingest_rejects_bad_version(settings, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "cli-data"))
    exit_code = cli_main(
        CLI_ARGS
        + [
            "--company",
            "acme",
            "--version",
            "not a version",
            "--file",
            str(settings.policies_dir / "acme-2026.1.md"),
            "--effective-at",
            "2026-04-01",
        ]
    )
    assert exit_code == 1


def test_jwt_secret_file_created_with_random_secret(tmp_path):
    from app.config import Settings
    from app.auth import ensure_jwt_secret

    settings = Settings(data_dir=tmp_path / "d1", jwt_secret=None)
    first = ensure_jwt_secret(settings)
    second = ensure_jwt_secret(settings)
    assert first == second  # persisted, not regenerated
    assert len(first) >= 40  # random, long
    assert (settings.jwt_secret_file).exists()
    other = Settings(data_dir=tmp_path / "d2", jwt_secret=None)
    assert ensure_jwt_secret(other) != first  # per-install randomness
