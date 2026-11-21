"""Seed the three fictional companies and their base policies at startup.

Seeding is idempotent: a policy whose exact content is already ACTIVE is
left untouched, so restarts never duplicate vectors or bump versions.
"""

from __future__ import annotations

import logging
import re

from .policies import IngestService

logger = logging.getLogger(__name__)

SEED_COMPANIES = {
    "acme": "Acme Technologies (fictional)",
    "bluebird": "Bluebird Transport (fictional)",
    "copper": "Copper Logistics (fictional)",
}

_FILE_PATTERN = re.compile(r"^(?P<company>[a-z0-9]+)-(?P<version>[\w.\-]+)\.md$")


def seed_policies(policies_dir, ingest: IngestService) -> list[str]:  # noqa: ANN001 - Path
    """Ingest every '<company>-<version>.md' in policies_dir; returns companies touched."""

    seeded: list[str] = []
    for path in sorted(policies_dir.glob("*.md")):
        match = _FILE_PATTERN.match(path.name)
        if not match:
            logger.warning("skipping unparsable policy filename: %s", path.name)
            continue
        company_id = match.group("company")
        version_id = match.group("version")
        content = path.read_bytes()
        result = ingest.ingest(
            company_id=company_id,
            version_id=version_id,
            effective_at="2026-01-01",
            content=content,
            source_name=path.name,
        )
        action = "already active" if result.idempotent else "activated"
        logger.info(
            "seed policy %s v%s: %s (%s sections, %s chunks)",
            company_id,
            version_id,
            action,
            result.section_count,
            result.chunk_count,
        )
        seeded.append(company_id)
    return seeded
