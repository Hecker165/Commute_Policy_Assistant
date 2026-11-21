"""Tenant-safe retrieval.

Every query resolves the company's ACTIVE version from the SQLite catalog
and applies a mandatory company+version metadata filter in the vector store.
The caller (route handler) passes the company from the verified JWT; there
is no parameter that can widen the filter.

Two modes:
- semantic (baseline): Chroma cosine similarity, similarity floor applied.
- hybrid (the measured bonus): fuse semantic ranks with a lexical BM25-lite
  score over the same tenant/version-filtered chunk corpus (from SQLite)
  using reciprocal rank fusion.
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter
from dataclasses import dataclass, field

from .catalog import Catalog
from .config import Settings
from .vectordb import VectorStore, tokenize

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Evidence:
    """A retrieved chunk with trusted metadata only (never model-authored)."""

    chunk_id: str
    company_id: str
    version_id: str
    document_title: str
    section_id: str
    section_label: str
    text: str
    similarity: float
    lexical_rank: int | None = None


@dataclass(frozen=True)
class RetrievalStats:
    latency_ms: float
    mode: str
    k: int
    min_similarity: float
    candidates_considered: int = 0
    sections: list[str] = field(default_factory=list)


class Retriever:
    def __init__(self, catalog: Catalog, vectordb: VectorStore, settings: Settings) -> None:
        self._catalog = catalog
        self._vectordb = vectordb
        self._settings = settings

    def retrieve(self, question: str, company_id: str) -> tuple[list[Evidence], RetrievalStats]:
        started = time.perf_counter()
        active = self._catalog.get_active_version(company_id)
        if active is None:
            logger.warning("no active policy version for company")
            return [], self._stats(started, mode=self._settings.retrieval_mode, k=0)

        if self._settings.retrieval_mode == "hybrid":
            evidence = self._hybrid(question, company_id, active.version_id)
        else:
            evidence = self._semantic(question, company_id, active.version_id)

        stats = self._stats(
            started,
            mode=self._settings.retrieval_mode,
            k=len(evidence),
            candidates_considered=max(len(evidence), self._settings.retrieval_k),
            sections=[e.section_label for e in evidence],
        )
        return evidence, stats

    # --- baseline -------------------------------------------------------------

    def _semantic(self, question: str, company_id: str, version_id: str) -> list[Evidence]:
        hits = self._vectordb.query(
            question=question,
            company_id=company_id,
            version_id=version_id,
            k=self._settings.retrieval_k,
        )
        floor = self._settings.retrieval_min_similarity
        # Chunk text comes from the SQLite catalog (trusted), matched by id.
        text_by_id = {
            c.chunk_id: c.text
            for c in self._catalog.get_chunks(company_id, version_id)
        }
        evidence: list[Evidence] = []
        for h in hits:
            if h.similarity < floor:
                continue
            text = text_by_id.get(h.chunk_id)
            if text is None:
                # Vector exists but catalog row missing: treat as corrupt, skip.
                logger.error("vector chunk missing from catalog; skipping")
                continue
            evidence.append(
                Evidence(
                    chunk_id=h.chunk_id,
                    company_id=h.metadata["company_id"],
                    version_id=h.metadata["version_id"],
                    document_title=h.metadata["document_title"],
                    section_id=h.metadata["section_id"],
                    section_label=h.metadata["section_label"],
                    text=text,
                    similarity=h.similarity,
                )
            )
        return evidence[: self._settings.retrieval_k]

    # --- bonus: hybrid rank fusion ---------------------------------------------

    def _hybrid(self, question: str, company_id: str, version_id: str) -> list[Evidence]:
        # 1) semantic candidates (no floor yet; fusion decides membership)
        n = max(self._settings.retrieval_k, self._settings.lexical_candidates)
        semantic_hits = self._vectordb.query(
            question=question, company_id=company_id, version_id=version_id, k=n
        )
        semantic_order = [h.chunk_id for h in semantic_hits]

        # 2) lexical candidates over the same tenant+version corpus (SQLite)
        chunk_rows = self._catalog.get_chunks(company_id, version_id)
        lexical_order = _lexical_ranking(question, chunk_rows)[: self._settings.lexical_candidates]

        # 3) reciprocal rank fusion
        scores: dict[str, float] = {}
        for rank, chunk_id in enumerate(semantic_order):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self._settings.rrf_k + rank + 1)
        for rank, chunk_id in enumerate(lexical_order):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self._settings.rrf_k + rank + 1)
        fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

        sim_by_id = {h.chunk_id: h.similarity for h in semantic_hits}
        floor = self._settings.retrieval_min_similarity
        evidence: list[Evidence] = []
        for chunk_id, _score in fused[: self._settings.retrieval_k]:
            row = next((c for c in chunk_rows if c.chunk_id == chunk_id), None)
            if row is None:
                continue  # never cite a chunk absent from the trusted catalog
            sim = sim_by_id.get(chunk_id, 0.0)
            lexical_rank = lexical_order.index(chunk_id) if chunk_id in lexical_order else None
            # Keep an item if the semantic floor passes OR lexical evidence found it.
            if sim >= floor or lexical_rank is not None:
                evidence.append(
                    Evidence(
                        chunk_id=row.chunk_id,
                        company_id=row.company_id,
                        version_id=row.version_id,
                        document_title=row.document_title,
                        section_id=row.section_id,
                        section_label=row.section_label,
                        text=row.text,
                        similarity=sim,
                        lexical_rank=lexical_rank,
                    )
                )
        return evidence

    def _stats(
        self,
        started: float,
        *,
        mode: str,
        k: int,
        candidates_considered: int = 0,
        sections: list[str] | None = None,
    ) -> RetrievalStats:
        return RetrievalStats(
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            mode=mode,
            k=k,
            min_similarity=self._settings.retrieval_min_similarity,
            candidates_considered=candidates_considered,
            sections=sections or [],
        )


def _lexical_ranking(question: str, chunk_rows) -> list[str]:  # noqa: ANN001 - ChunkRow list
    """BM25-lite lexical scoring; returns chunk ids best-first, stable ties."""

    query_tokens = tokenize(question)
    if not query_tokens or not chunk_rows:
        return []

    doc_tokens = {c.chunk_id: tokenize(c.text) for c in chunk_rows}
    df: Counter[str] = Counter()
    for tokens in doc_tokens.values():
        df.update(set(tokens))
    n_docs = max(1, len(chunk_rows))
    avg_len = sum(len(t) for t in doc_tokens.values()) / n_docs

    scores: dict[str, float] = {}
    k1, b = 1.5, 0.75
    for chunk_id, tokens in doc_tokens.items():
        if not tokens:
            continue
        tf = Counter(tokens)
        score = 0.0
        for token in query_tokens:
            if token not in tf:
                continue
            idf = math.log(max((n_docs - df[token] + 0.5) / (df[token] + 0.5) + 1.0, 1e-9))
            score += idf * (tf[token] * (k1 + 1)) / (
                tf[token] + k1 * (1 - b + b * len(tokens) / (avg_len or 1))
            )
        if score > 0:
            scores[chunk_id] = score
    return [cid for cid, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]
