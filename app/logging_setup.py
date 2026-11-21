"""Logging configuration and the request-id context used across the app.

We log operational metadata only (request id, company, status, timings,
LLM call count). We never log question text, policy content, JWTs or secrets.
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
company_var: ContextVar[str] = ContextVar("company", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        record.request_id = request_id_var.get()
        record.company = company_var.get()
        return True


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] req=%(request_id)s company=%(company)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    # Chroma/HTTP libraries are noisy at INFO on startup.
    for noisy in ("chromadb", "httpx", "httpcore", "urllib3", "onnxruntime"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
