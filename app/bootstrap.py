"""Builds the full service context (catalog, vector store, retriever, LLM,
answer + ingest services). Shared by the FastAPI app, the CLI, eval and the
benchmark so every path uses exactly the same wiring.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .answering import AnswerService
from .auth import ensure_jwt_secret
from .catalog import Catalog
from .config import Settings
from .llm import build_llm
from .llm.base import LLMProvider
from .policies import IngestService
from .retrieval import Retriever
from .vectordb import VectorStore


@dataclass
class AppContext:
    settings: Settings
    catalog: Catalog
    vectordb: VectorStore
    retriever: Retriever
    llm: LLMProvider
    answer_service: AnswerService
    ingest_service: IngestService
    jwt_secret: str

    def close(self) -> None:
        self.catalog.close()


def build_context(settings: Settings) -> AppContext:
    jwt_secret = ensure_jwt_secret(settings)
    catalog = Catalog(settings.catalog_path)
    vectordb = VectorStore(settings)
    retriever = Retriever(catalog, vectordb, settings)
    llm = build_llm(settings)
    answer_service = AnswerService(retriever, llm, settings)
    ingest_service = IngestService(catalog, vectordb, settings)
    return AppContext(
        settings=settings,
        catalog=catalog,
        vectordb=vectordb,
        retriever=retriever,
        llm=llm,
        answer_service=answer_service,
        ingest_service=ingest_service,
        jwt_secret=jwt_secret,
    )


def monotonic_deadline(settings: Settings) -> float:
    return time.monotonic() + settings.request_budget_s
