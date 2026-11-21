"""Answer pipeline: grounding, citation allowlist, call budget, fallbacks."""

from __future__ import annotations

import json

import pytest

from app.answering import AnswerPipelineError
from app.llm.fake import ScriptedLLM, attack_behaviour

ESCORT_JSON = json.dumps(
    {
        "answerable": True,
        "answer": "Acme requires a security escort for women employees on drops after 8:00 PM.",
        "evidence_ids": ["c1"],
    }
)
FABRICATED_JSON = json.dumps(
    {
        "answerable": True,
        "answer": "Escort after 8:00 PM.",
        "evidence_ids": ["c1", "c9", "BLUEBIRD_CHUNK_1"],
    }
)
MALFORMED = "this is not json at all"
NOT_ANSWERABLE = json.dumps({"answerable": False, "answer": "", "evidence_ids": []})


def _ask(seeded_context, question="escort for a 9:30 PM drop?", company="acme"):
    import time

    return seeded_context.answer_service.answer(
        question=question,
        company_id=company,
        deadline=time.monotonic() + 5.0,
    )


def test_covered_answer_cites_retrieved_chunk(seeded_context):
    seeded_context.scripted.responses = [ESCORT_JSON]
    result = _ask(seeded_context)
    assert result.covered is True
    assert result.llm_calls == 1
    assert len(result.citations) == 1
    citation = result.citations[0]
    assert citation.company_id == "acme"
    assert citation.version == "2026.1"
    assert citation.section.startswith("4.")
    assert citation.chunk_id.startswith("acme:2026.1:s4:")


def test_no_evidence_means_zero_llm_calls(seeded_context):
    result = _ask(seeded_context, question="What is the capital of France?")
    assert result.covered is False
    assert result.llm_calls == 0
    assert result.citations == []
    assert "transport desk" in result.answer


def test_not_answerable_from_model_returns_not_covered(seeded_context):
    seeded_context.scripted.responses = [NOT_ANSWERABLE]
    result = _ask(seeded_context)
    assert result.covered is False
    assert result.llm_calls == 1
    assert result.citations == []


def test_fabricated_extras_dropped_valid_citation_kept(seeded_context):
    """Model cites c1 (real) plus fabricated ids: fabricated ids are dropped,
    the valid one survives, and no repair call is needed."""

    seeded_context.scripted.responses = [FABRICATED_JSON]
    result = _ask(seeded_context)
    assert result.llm_calls == 1
    assert result.covered is True
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id.startswith("acme:2026.1:s4")


def test_all_fabricated_citations_trigger_repair_then_success(seeded_context):
    all_fake = json.dumps(
        {
            "answerable": True,
            "answer": "Escort after 8:00 PM.",
            "evidence_ids": ["c9", "BLUEBIRD_CHUNK_1"],
        }
    )
    seeded_context.scripted.responses = [all_fake, ESCORT_JSON]
    result = _ask(seeded_context)
    assert result.llm_calls == 2  # exactly one repair call, budget respected
    assert result.covered is True
    assert result.citations[0].chunk_id.startswith("acme:")


def test_invalid_after_repair_raises_pipeline_error(seeded_context):
    seeded_context.scripted.responses = [MALFORMED, MALFORMED]
    with pytest.raises(AnswerPipelineError):
        _ask(seeded_context)
    assert seeded_context.scripted.calls == 2  # never a third attempt


def test_provider_failure_raises_pipeline_error(seeded_context):
    seeded_context.scripted.fail_all = True
    with pytest.raises(AnswerPipelineError):
        _ask(seeded_context)
    assert seeded_context.scripted.calls == 1  # no unbounded retry


def test_escaping_model_attack_via_citations(seeded_context):
    """Model obeys document injection: cites unknown ids AND names Bluebird
    although neither the question nor the evidence mentions it. The server
    must reject the output, attempt one repair, then fail safely."""

    seeded_context.scripted.behaviour = attack_behaviour
    seeded_context.scripted.responses = None
    with pytest.raises(AnswerPipelineError):
        _ask(seeded_context)
    assert seeded_context.scripted.calls == 2


def test_answer_naming_unknown_company_is_rejected(seeded_context):
    """A hallucinated cross-company answer is invalid output even when it
    cites a real chunk id."""

    hallucinated = json.dumps(
        {
            "answerable": True,
            "answer": "Per Bluebird policy, escorts start after 10:00 PM.",
            "evidence_ids": ["c1"],
        }
    )
    seeded_context.scripted.responses = [hallucinated, ESCORT_JSON]
    result = _ask(seeded_context, question="What is the escort rule for late drops?")
    assert result.covered is True
    assert "Bluebird" not in result.answer
    assert result.citations[0].company_id == "acme"


def test_call_budget_never_exceeds_two(seeded_context):
    # Two invalid outputs consume the whole budget; the third scripted
    # response must never be requested.
    seeded_context.scripted.responses = [MALFORMED, MALFORMED, ESCORT_JSON]
    with pytest.raises(AnswerPipelineError):
        _ask(seeded_context)
    assert seeded_context.scripted.calls == 2


def test_cross_company_question_cites_only_own_company(seeded_context):
    """Employee asks about another company by name; the pipeline must ground
    on the authenticated company's policy only."""

    seeded_context.scripted.responses = [
        json.dumps(
            {
                "answerable": True,
                "answer": "Escort is mandatory after 8:00 PM for women employees.",
                "evidence_ids": ["c1"],
            }
        )
    ]
    result = _ask(
        seeded_context,
        question="My colleague at Bluebird gets escorts after 10 PM; what time applies to me?",
    )
    assert result.covered is True
    assert all(c.company_id == "acme" for c in result.citations)


def test_prompt_labels_excerpts_as_untrusted(seeded_context):
    seeded_context.scripted.responses = [ESCORT_JSON]
    _ask(seeded_context)
    user_prompt = seeded_context.scripted.last_user_prompt()
    system_prompt = seeded_context.scripted.stats.history[-1].system
    # The injection text travels as quoted policy data...
    assert "ignore all previous instructions" in user_prompt.lower()
    assert "quoted policy data" in user_prompt.lower()
    # ...and the system prompt forbids following it.
    assert "untrusted" in system_prompt.lower()
    assert "do NOT follow" in system_prompt
