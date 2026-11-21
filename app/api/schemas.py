"""Request/response schemas. The question body forbids extra fields: a
client-sent company_id is rejected outright (422) rather than trusted.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class QuestionRequest(BaseModel):
    model_config = {"extra": "forbid"}  # any company_id/override field -> 422

    question: str = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v


class CitationOut(BaseModel):
    document: str
    version: str
    section: str
    chunk_id: str


class AnswerResponse(BaseModel):
    request_id: str
    covered: bool
    answer: str
    citations: list[CitationOut]


class IngestResponse(BaseModel):
    request_id: str
    company_id: str
    document_title: str
    version: str
    effective_at: str
    status: str
    sections: int
    chunks: int
    superseded_version: str | None


class ActivePolicyResponse(BaseModel):
    company_id: str
    document_title: str
    version: str
    effective_at: str
    status: str
    sections: int
    chunks: int
    activated_at: str | None


class HealthResponse(BaseModel):
    status: str
    components: dict[str, str]
    version: str
