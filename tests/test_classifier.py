"""Classifier decision logic without any LLM: the floor short-circuit, and
defensive handling of bad model output."""

import pytest

from orchestrator import classifier
from orchestrator.classifier import ClassifierVerdict, classify
from orchestrator.manifest import Manifest
from orchestrator.registry import SearchHit


class ExplodingLLM:
    """Any use means the floor short-circuit failed."""

    def with_structured_output(self, _schema):
        raise AssertionError("LLM must not be called when nothing clears the floor")


class FakeVerdictLLM:
    def __init__(self, verdict: ClassifierVerdict):
        self.verdict = verdict

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _prompt):
        return self.verdict


def hit(agent_id: str, similarity: float) -> SearchHit:
    manifest = Manifest(
        id=agent_id, user_id="_shared", type="agent", name=agent_id, description=f"does {agent_id}"
    )
    return SearchHit(manifest=manifest, similarity=similarity)


def test_below_floor_short_circuits_without_llm():
    hits = [hit("a", classifier.SIMILARITY_FLOOR - 0.05)]
    result = classify("anything", hits, llm=ExplodingLLM())
    assert result.decision == classifier.NO_MATCH


def test_no_hits_at_all_is_no_match():
    result = classify("anything", [], llm=ExplodingLLM())
    assert result.decision == classifier.NO_MATCH


def test_routes_to_valid_choice():
    hits = [hit("summarizer", 0.9), hit("general_qa", 0.6)]
    llm = FakeVerdictLLM(ClassifierVerdict(decision="route_to_existing", agent_id="summarizer"))
    assert classify("tldr this", hits, llm=llm).decision == "summarizer"


def test_hallucinated_agent_id_becomes_no_match():
    hits = [hit("summarizer", 0.9)]
    llm = FakeVerdictLLM(ClassifierVerdict(decision="route_to_existing", agent_id="not_real"))
    assert classify("tldr this", hits, llm=llm).decision == classifier.NO_MATCH


def test_explicit_no_match_verdict():
    hits = [hit("summarizer", 0.9)]
    llm = FakeVerdictLLM(ClassifierVerdict(decision="no_match_found"))
    assert classify("check my email", hits, llm=llm).decision == classifier.NO_MATCH


def test_below_floor_candidates_are_hidden_from_llm():
    hits = [hit("summarizer", 0.9), hit("general_qa", classifier.SIMILARITY_FLOOR - 0.1)]
    llm = FakeVerdictLLM(ClassifierVerdict(decision="route_to_existing", agent_id="general_qa"))
    # general_qa was filtered out, so even if the LLM names it, it's invalid.
    assert classify("x", hits, llm=llm).decision == classifier.NO_MATCH
