"""Chroma vector store wrapper.

The store is dumb and tenant-blind: every read and write goes through
methods that REQUIRE company_id + version_id metadata. There is no code
path that queries the collection without both filters, which is what makes
cross-company leakage structurally impossible (and testable).

Embedding providers:
- "default": Chroma's bundled ONNX MiniLM-L6-v2 (semantic; downloads ~80MB
  on first use into CHROMA_ONNX_MODEL_CACHE_DIR if set, else user cache).
- "hash": a deterministic hashed bag-of-ngrams vector (no download, no
  network) used by the offline test/CI path.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import chromadb
from chromadb.api import Collection
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from .config import Settings

if TYPE_CHECKING:  # circular-free typing only
    from .policies import Chunk

_COLLECTION = "policies"
_TOKEN = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the
    to was were will with you your our we they their this these those can could
    do does did if then than so such not no nor but while about into over under
    after before between during without within upon per via any all each other
    more most some only own same very just also may might must shall should
    would there here when where why how what which who whom whose""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens minus stopwords (shared by retrieval)."""

    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class _HashNgramEmbedding(EmbeddingFunction[Documents]):
    """Deterministic hashed bag-of-(1,2)-grams vector; L2-normalised.

    Not semantic — it behaves like a mini TF-IDF. Used for offline tests/CI
    so no model download or network is ever required.
    """

    def __init__(self, dim: int = 1024) -> None:
        self._dim = dim

    def __call__(self, input):  # noqa: ANN001 - chroma signature
        vectors = []
        for text in input:
            vec = [0.0] * self._dim
            tokens = tokenize(text)
            grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            for gram in grams:
                h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)  # noqa: S324
                vec[h % self._dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors

    @staticmethod
    def name() -> str:
        return "hash-ngram"

    def get_config(self) -> dict:
        return {"dim": self._dim}

    @staticmethod
    def build_from_config(config: dict) -> "_HashNgramEmbedding":
        return _HashNgramEmbedding(dim=config.get("dim", 1024))

    def default_space(self) -> str:
        return "cosine"


def build_embedding_function(provider: str, model_cache_dir: Path | None = None):  # noqa: ANN201
    if provider == "hash":
        return _HashNgramEmbedding()
    if provider == "default":
        from chromadb.utils import embedding_functions

        if model_cache_dir is not None:
            # Redirect Chroma's ONNX MiniLM download/cache next to the app data
            # (class attribute override) so nothing lands in the user's home dir.
            cls = embedding_functions.ONNXMiniLM_L6_V2
            cls.DOWNLOAD_PATH = model_cache_dir / "onnx_models" / cls.MODEL_NAME
        return embedding_functions.DefaultEmbeddingFunction()
    raise ValueError(f"unknown embedding provider: {provider}")


@dataclass(frozen=True)
class VectorHit:
    chunk_id: str
    similarity: float
    metadata: dict


class VectorStore:
    """Thin, lock-guarded Chroma wrapper; cosine space; tenant-filtered I/O."""

    def __init__(self, settings: Settings) -> None:
        path = Path(settings.chroma_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        client = chromadb.PersistentClient(
            path=str(path),
            settings=chromadb.Settings(anonymized_telemetry=False, is_persistent=True),
        )
        self._embedding = build_embedding_function(
            settings.embedding_provider, model_cache_dir=settings.model_cache_dir
        )
        self._collection: Collection = client.get_or_create_collection(
            name=_COLLECTION,
            embedding_function=self._embedding,
            metadata={"hnsw:space": "cosine"},
        )

    def healthcheck(self) -> bool:
        with self._lock:
            self._collection.count()
            return True

    def upsert_chunks(self, *, company_id: str, version_id: str, chunks: list[Chunk]) -> None:
        """Stage vectors for one version; ids are stable so retries are idempotent."""

        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        documents = [
            f"[{c.company_id} | {c.document_title} | Section {c.section_label}]\n{c.text}"
            for c in chunks
        ]
        metadatas = [
            {
                "company_id": c.company_id,
                "version_id": c.version_id,
                "section_id": c.section_id,
                "section_label": c.section_label,
                "document_title": c.document_title,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]
        with self._lock:
            self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def delete_version(self, company_id: str, version_id: str) -> None:
        """Remove all vectors for one (company, version) — used for cleanup."""

        with self._lock:
            self._collection.delete(where=self._where(company_id, version_id))

    def count(self, company_id: str, version_id: str) -> int:
        with self._lock:
            found = self._collection.get(where=self._where(company_id, version_id), include=[])
            return len(found.get("ids", []))

    def query(
        self, *, question: str, company_id: str, version_id: str, k: int
    ) -> list[VectorHit]:
        """Cosine-similarity search, hard-filtered to company + version."""

        where = self._where(company_id, version_id)
        with self._lock:
            result = self._collection.query(
                query_texts=[question],
                n_results=max(1, min(k, self._collection.count() or 1)),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        hits: list[VectorHit] = []
        for i, chunk_id in enumerate(result["ids"][0]):
            distance = result["distances"][0][i]
            similarity = 1.0 - float(distance)  # cosine distance -> similarity
            metadata = dict(result["metadatas"][0][i])
            hits.append(VectorHit(chunk_id=chunk_id, similarity=similarity, metadata=metadata))
        hits.sort(key=lambda h: h.similarity, reverse=True)
        return hits

    @staticmethod
    def _where(company_id: str, version_id: str) -> dict:
        # The single source of the tenant/version filter; never optional.
        return {
            "$and": [
                {"company_id": {"$eq": company_id}},
                {"version_id": {"$eq": version_id}},
            ]
        }
