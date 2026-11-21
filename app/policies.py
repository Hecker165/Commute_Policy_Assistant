"""Policy parsing, deterministic chunking, and atomic version ingestion.

Ingestion sequence (the safety-critical part):
1. decode + validate input (utf-8, size, headings) -> parse into sections
2. chunk deterministically, compute stable chunk ids
3. stage: write chunk text to SQLite and vectors to Chroma (new version is
   still INACTIVE, so retrieval cannot see it)
4. verify the staged vector count matches expectation
5. activate: flip the active-version pointer in ONE SQLite transaction
6. best-effort cleanup of the superseded version's vectors

If any step before (5) fails, the previous active version is untouched and
the staged version is marked 'failed' and its orphan vectors are removed.
Retrying the same content is idempotent (stable ids + upserts).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime

from .catalog import Catalog, ChunkRow, STATUS_ACTIVE, STATUS_FAILED, STATUS_STAGED
from .config import Settings
from .vectordb import VectorStore

VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")
H1_PATTERN = re.compile(r"^#\s+(.+?)\s*$")
H2_PATTERN = re.compile(r"^##\s+(\d+)[.\)]?\s*(.+?)\s*$")  # "## 3. Night Drops"
H2_UNNUMBERED = re.compile(r"^##\s+(.+?)\s*$")
ALLOWED_EXTENSIONS = (".md", ".txt")


class PolicyValidationError(Exception):
    """Input failed validation; message is safe to show to an admin."""


class IngestError(Exception):
    """Ingestion failed; the previously active version is unchanged."""


@dataclass(frozen=True)
class Section:
    id: str
    label: str
    title: str
    text: str


@dataclass(frozen=True)
class ParsedPolicy:
    title: str
    sections: list[Section]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            ("\n\n".join(s.text for s in self.sections) + self.title).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    company_id: str
    version_id: str
    section_id: str
    section_label: str
    document_title: str
    chunk_index: int
    text: str


@dataclass(frozen=True)
class IngestResult:
    company_id: str
    version_id: str
    document_title: str
    effective_at: str
    section_count: int
    chunk_count: int
    superseded_version: str | None
    idempotent: bool


# --- parsing ---------------------------------------------------------------


def parse_policy(content: str) -> ParsedPolicy:
    """Parse a Markdown policy into titled, numbered sections.

    Only '## <n>. Title' headings become sections (stable ids for citations);
    the first '# Title' is the document title. Anything before the first
    section (company metadata block) is ignored for retrieval purposes.
    """

    if not content or not content.strip():
        raise PolicyValidationError("document is empty")

    title: str | None = None
    sections: list[Section] = []
    current: Section | None = None
    lines_buffer: list[str] = []

    def flush() -> None:
        nonlocal current, lines_buffer
        if current is not None:
            text = "\n".join(lines_buffer).strip()
            if text:
                sections.append(
                    Section(id=current.id, label=current.label, title=current.title, text=text)
                )
        lines_buffer = []

    for line in content.splitlines():
        h1 = H1_PATTERN.match(line)
        h2 = H2_PATTERN.match(line)
        h2_any = H2_UNNUMBERED.match(line)
        if h1 and title is None and current is None:
            title = h1.group(1).strip()
            continue
        if h2:
            flush()
            num, sec_title = h2.group(1), h2.group(2)
            current = Section(id=num, label=f"{num}. {sec_title}", title=sec_title, text="")
            continue
        if h2_any:
            flush()
            sec_title = h2_any.group(1)
            slug = re.sub(r"[^a-z0-9]+", "-", sec_title.lower()).strip("-")[:24] or "section"
            current = Section(id=slug, label=sec_title, title=sec_title, text="")
            continue
        lines_buffer.append(line)
    flush()

    if title is None:
        raise PolicyValidationError("document must start with an '# Title' heading")
    if not sections:
        raise PolicyValidationError("document must contain at least one '## <n>. Title' section")
    return ParsedPolicy(title=title, sections=sections)


def chunk_policy(
    parsed: ParsedPolicy,
    *,
    company_id: str,
    version_id: str,
    max_chars: int,
    overlap_chars: int,
) -> list[Chunk]:
    """Split sections into fixed-size, sentence-aware overlapping chunks.

    Chunk ids are deterministic: '<company>:<version>:s<section>:<index>' so
    re-ingesting the same content is idempotent and citations stay stable.
    Each embedded text carries a contextual header (document + section) so
    the embedding sees the citation context, while metadata stays trusted.
    """

    chunks: list[Chunk] = []
    for section in parsed.sections:
        pieces = _split_text(section.text, max_chars=max_chars, overlap_chars=overlap_chars)
        for idx, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"{company_id}:{version_id}:s{section.id}:{idx}",
                    company_id=company_id,
                    version_id=version_id,
                    section_id=section.id,
                    section_label=section.label,
                    document_title=parsed.title,
                    chunk_index=idx,
                    text=piece,
                )
            )
    return chunks


def _split_text(text: str, *, max_chars: int, overlap_chars: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces: list[str] = []
    buf = ""
    for sentence in sentences:
        candidate = f"{buf} {sentence}".strip()
        if len(candidate) <= max_chars or not buf:
            # A single oversized sentence is hard-split; correctness beats purity.
            while len(sentence) > max_chars:
                pieces.append(sentence[:max_chars])
                sentence = sentence[max_chars - overlap_chars :]
            buf = candidate if len(candidate) <= max_chars else sentence
        else:
            pieces.append(buf)
            tail = buf[-overlap_chars:] if overlap_chars > 0 else ""
            buf = f"{tail} {sentence}".strip()
    if buf:
        pieces.append(buf)
    return [p for p in pieces if p.strip()]


# --- ingestion service ------------------------------------------------------


class IngestService:
    """Validates, stages, verifies and activates a policy version."""

    def __init__(
        self,
        catalog: Catalog,
        vectordb: VectorStore,
        settings: Settings,
        *,
        company_names: dict[str, str] | None = None,
    ) -> None:
        self._catalog = catalog
        self._vectordb = vectordb
        self._settings = settings
        self._company_names = company_names or {}

    def ingest(
        self,
        *,
        company_id: str,
        version_id: str,
        effective_at: str,
        content: bytes,
        source_name: str = "upload",
        verify_vectors: bool = True,
    ) -> IngestResult:
        self._validate_inputs(version_id, effective_at, content, source_name)

        text = self._decode(content)
        parsed = parse_policy(text)
        chunks = chunk_policy(
            parsed,
            company_id=company_id,
            version_id=version_id,
            max_chars=self._settings.chunk_max_chars,
            overlap_chars=self._settings.chunk_overlap_chars,
        )
        sha = parsed.sha256

        existing = self._catalog.get_version(company_id, version_id)
        if existing is not None:
            if existing.source_sha256 == sha and existing.status == STATUS_ACTIVE:
                # Deliberate re-ingestion of identical content: no-op, keep answerable.
                return IngestResult(
                    company_id=company_id,
                    version_id=version_id,
                    document_title=existing.title,
                    effective_at=existing.effective_at,
                    section_count=existing.section_count,
                    chunk_count=existing.chunk_count,
                    superseded_version=None,
                    idempotent=True,
                )
            if existing.source_sha256 != sha:
                raise PolicyValidationError(
                    f"version '{version_id}' already exists with different content; "
                    "use a new version id"
                )
            # Same content re-uploaded after a failure: restage cleanly.
            self._catalog.set_status(company_id, version_id, STATUS_STAGED)

        row = ChunkRow  # typing shorthand
        chunk_rows = [
            row(
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
        ]

        try:
            # (3) stage -- inactive until activation
            self._catalog.upsert_version(
                company_id=company_id,
                version_id=version_id,
                title=parsed.title,
                effective_at=effective_at,
                source_sha256=sha,
                section_count=len(parsed.sections),
                chunk_count=len(chunks),
                status=STATUS_STAGED,
            )
            self._catalog.replace_chunks(company_id, version_id, chunk_rows)
            self._vectordb.upsert_chunks(company_id=company_id, version_id=version_id, chunks=chunks)

            # (4) verify staged vectors before touching the pointer
            if verify_vectors:
                staged = self._vectordb.count(company_id, version_id)
                if staged != len(chunks):
                    raise IngestError(
                        f"vector verification failed: staged {staged}, expected {len(chunks)}"
                    )

            # (5) activate atomically; old version answers until this commits
            superseded = self._catalog.activate_version(company_id, version_id)
        except Exception as exc:
            self._catalog.set_status(company_id, version_id, STATUS_FAILED)
            try:  # best-effort orphan cleanup; never touch the active version
                self._vectordb.delete_version(company_id, version_id)
            except Exception:  # pragma: no cover
                pass
            if isinstance(exc, (PolicyValidationError, IngestError)):
                raise
            raise IngestError(f"ingestion failed before activation: {type(exc).__name__}") from exc

        # (6) the old version is dead once the new pointer is live
        if superseded:
            try:
                self._vectordb.delete_version(company_id, superseded)
            except Exception:  # pragma: no cover - retrieval filter already excludes it
                pass

        return IngestResult(
            company_id=company_id,
            version_id=version_id,
            document_title=parsed.title,
            effective_at=effective_at,
            section_count=len(parsed.sections),
            chunk_count=len(chunks),
            superseded_version=superseded,
            idempotent=False,
        )

    # helpers ----------------------------------------------------------------

    def _validate_inputs(
        self, version_id: str, effective_at: str, content: bytes, source_name: str
    ) -> None:
        if not VERSION_PATTERN.match(version_id or ""):
            raise PolicyValidationError(
                "version id must match [A-Za-z0-9][A-Za-z0-9._-]{0,31}"
            )
        if not _is_iso_date(effective_at or ""):
            raise PolicyValidationError("effective_at must be an ISO date (YYYY-MM-DD)")
        name = (source_name or "").lower()
        if name and not name.endswith(ALLOWED_EXTENSIONS):
            raise PolicyValidationError("only .md and .txt policy files are supported")
        if len(content) == 0:
            raise PolicyValidationError("uploaded document is empty")
        if len(content) > self._settings.max_upload_bytes:
            raise PolicyValidationError("uploaded document exceeds the size limit")

    def _decode(self, content: bytes) -> str:
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PolicyValidationError("document must be valid UTF-8") from exc


def _is_iso_date(value: str) -> bool:
    try:
        datetime.strptime(value, "%Y-%m-%d")  # noqa: DTZ007 - format check only
        return True
    except ValueError:
        return False
