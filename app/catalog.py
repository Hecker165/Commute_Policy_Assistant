"""SQLite policy catalog: the authoritative store for active-version state.

Design notes:
- One row per (company, version). A partial UNIQUE index guarantees at most
  one ACTIVE version per company, enforced by SQLite itself.
- Activation is a single transaction (retire old -> activate new). If it
  fails, nothing changed, so the previous policy keeps answering.
- Chunk text is also stored here so lexical (hybrid/offline) retrieval can
  run without the vector store; the vector store remains authoritative for
  embeddings only.
- WAL mode + a process-level lock serialize writers; readers never block.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

STATUS_STAGED = "staged"
STATUS_ACTIVE = "active"
STATUS_RETIRED = "retired"
STATUS_FAILED = "failed"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    company_id   TEXT PRIMARY KEY,
    display_name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS policy_versions (
    company_id    TEXT NOT NULL,
    version_id    TEXT NOT NULL,
    title         TEXT NOT NULL,
    effective_at  TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    section_count INTEGER NOT NULL,
    chunk_count   INTEGER NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('staged','active','retired','failed')),
    created_at    TEXT NOT NULL,
    activated_at  TEXT,
    PRIMARY KEY (company_id, version_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_per_company
    ON policy_versions (company_id) WHERE status = 'active';
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id       TEXT PRIMARY KEY,
    company_id     TEXT NOT NULL,
    version_id     TEXT NOT NULL,
    section_id     TEXT NOT NULL,
    section_label  TEXT NOT NULL,
    document_title TEXT NOT NULL,
    chunk_index    INTEGER NOT NULL,
    text           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_by_version
    ON chunks (company_id, version_id);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class VersionRow:
    company_id: str
    version_id: str
    title: str
    effective_at: str
    source_sha256: str
    section_count: int
    chunk_count: int
    status: str
    activated_at: str | None


@dataclass(frozen=True)
class ChunkRow:
    chunk_id: str
    company_id: str
    version_id: str
    section_id: str
    section_label: str
    document_title: str
    chunk_index: int
    text: str


class Catalog:
    """Small synchronous wrapper around the SQLite catalog."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=10)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def healthcheck(self) -> bool:
        with self._lock:
            row = self._conn.execute("SELECT 1").fetchone()
            return row is not None

    # --- companies ---------------------------------------------------------

    def ensure_company(self, company_id: str, display_name: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO companies (company_id, display_name) VALUES (?, ?) "
                "ON CONFLICT (company_id) DO NOTHING",
                (company_id, display_name),
            )
            self._conn.commit()

    # --- versions ----------------------------------------------------------

    def get_version(self, company_id: str, version_id: str) -> VersionRow | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM policy_versions WHERE company_id=? AND version_id=?",
                (company_id, version_id),
            ).fetchone()
        return _row_to_version(row) if row else None

    def get_active_version(self, company_id: str) -> VersionRow | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM policy_versions WHERE company_id=? AND status='active'",
                (company_id,),
            ).fetchone()
        return _row_to_version(row) if row else None

    def upsert_version(
        self,
        *,
        company_id: str,
        version_id: str,
        title: str,
        effective_at: str,
        source_sha256: str,
        section_count: int,
        chunk_count: int,
        status: str,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO policy_versions
                    (company_id, version_id, title, effective_at, source_sha256,
                     section_count, chunk_count, status, created_at, activated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (company_id, version_id) DO UPDATE SET
                    title=excluded.title,
                    effective_at=excluded.effective_at,
                    source_sha256=excluded.source_sha256,
                    section_count=excluded.section_count,
                    chunk_count=excluded.chunk_count,
                    status=excluded.status,
                    activated_at=excluded.activated_at
                """,
                (
                    company_id,
                    version_id,
                    title,
                    effective_at,
                    source_sha256,
                    section_count,
                    chunk_count,
                    status,
                    _now(),
                    _now() if status == STATUS_ACTIVE else None,
                ),
            )
            self._conn.commit()

    def set_status(self, company_id: str, version_id: str, status: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE policy_versions SET status=? WHERE company_id=? AND version_id=?",
                (status, company_id, version_id),
            )
            self._conn.commit()

    def activate_version(self, company_id: str, version_id: str) -> str | None:
        """Atomically switch the active pointer; returns the superseded version.

        The partial unique index makes it impossible for two versions of the
        same company to be active, even under concurrent activation.
        """

        with self._lock:
            cur = self._conn.execute(
                "SELECT version_id FROM policy_versions "
                "WHERE company_id=? AND status='active'",
                (company_id,),
            )
            previous = cur.fetchone()
            previous_id = previous["version_id"] if previous else None
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                if previous_id and previous_id != version_id:
                    self._conn.execute(
                        "UPDATE policy_versions SET status='retired' "
                        "WHERE company_id=? AND version_id=? AND status='active'",
                        (company_id, previous_id),
                    )
                self._conn.execute(
                    "UPDATE policy_versions SET status='active', activated_at=? "
                    "WHERE company_id=? AND version_id=? AND status IN ('staged','retired')",
                    (_now(), company_id, version_id),
                )
                if self._conn.total_changes == 0:  # pragma: no cover - guarded by callers
                    raise sqlite3.IntegrityError("activation affected no rows")
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return previous_id if previous_id != version_id else None

    # --- chunks ------------------------------------------------------------

    def replace_chunks(self, company_id: str, version_id: str, chunks: list[ChunkRow]) -> None:
        """Idempotently store the chunk text for one version (single txn)."""

        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._conn.execute(
                    "DELETE FROM chunks WHERE company_id=? AND version_id=?",
                    (company_id, version_id),
                )
                self._conn.executemany(
                    "INSERT INTO chunks (chunk_id, company_id, version_id, section_id,"
                    " section_label, document_title, chunk_index, text)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            c.chunk_id,
                            c.company_id,
                            c.version_id,
                            c.section_id,
                            c.section_label,
                            c.document_title,
                            c.chunk_index,
                            c.text,
                        )
                        for c in chunks
                    ],
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def get_chunks(self, company_id: str, version_id: str) -> list[ChunkRow]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM chunks WHERE company_id=? AND version_id=? "
                "ORDER BY section_id, chunk_index",
                (company_id, version_id),
            ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def count_chunks(self, company_id: str, version_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE company_id=? AND version_id=?",
                (company_id, version_id),
            ).fetchone()
        return int(row["n"])


def _row_to_version(row: sqlite3.Row) -> VersionRow:
    return VersionRow(
        company_id=row["company_id"],
        version_id=row["version_id"],
        title=row["title"],
        effective_at=row["effective_at"],
        source_sha256=row["source_sha256"],
        section_count=int(row["section_count"]),
        chunk_count=int(row["chunk_count"]),
        status=row["status"],
        activated_at=row["activated_at"],
    )


def _row_to_chunk(row: sqlite3.Row) -> ChunkRow:
    return ChunkRow(
        chunk_id=row["chunk_id"],
        company_id=row["company_id"],
        version_id=row["version_id"],
        section_id=row["section_id"],
        section_label=row["section_label"],
        document_title=row["document_title"],
        chunk_index=int(row["chunk_index"]),
        text=row["text"],
    )
