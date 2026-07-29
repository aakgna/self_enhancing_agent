"""Phase 1 exit criteria, on paraphrased queries (never verbatim manifest text).

The registry's contract is candidate retrieval, not final routing: the correct
agent must always be in the top-k the classifier sees, and top-1 must be right
for a large majority. Near-tie top-1 misses on ambiguous short questions are
the classifier's job (Phase 2), not the embedder's.

Run explicitly:

    uv run pytest eval/test_retrieval.py -v

First run downloads the local embedding model (~100 MB, one-time).
"""

import pytest

from orchestrator import registry

PARAPHRASED = [
    ("tldr this article for me", "summarizer"),
    ("can you shorten these meeting notes to the key points", "summarizer"),
    ("boil this email thread down to a couple of sentences", "summarizer"),
    ("give me the gist of the following transcript", "summarizer"),
    ("what's the difference between TCP and UDP", "general_qa"),
    ("explain how photosynthesis works", "general_qa"),
    ("who wrote The Brothers Karamazov", "general_qa"),
    ("why is the sky blue", "general_qa"),
    ("recommend a good sci-fi novel", "general_qa"),
    ("condense this 5-page report into one paragraph", "summarizer"),
]

TOP1_ACCURACY_FLOOR = 0.8


@pytest.fixture(scope="module", autouse=True)
def setup_registry():
    registry.init_registry()
    registry.backfill()


@pytest.mark.parametrize("query,expected", PARAPHRASED)
def test_correct_agent_is_in_candidates(query, expected):
    """The classifier can only route to what retrieval surfaces."""
    hits = registry.search(query, user_id="some_user", top_k=3, type_filter="agent")
    ids = [h.manifest.id for h in hits]
    assert expected in ids, f"{query!r}: {expected} missing from candidates {ids}"


def test_top1_accuracy_floor():
    correct = 0
    misses = []
    for query, expected in PARAPHRASED:
        hits = registry.search(query, user_id="some_user", type_filter="agent")
        if hits and hits[0].manifest.id == expected:
            correct += 1
        else:
            misses.append((query, [(h.manifest.id, round(h.similarity, 3)) for h in hits[:2]]))
    accuracy = correct / len(PARAPHRASED)
    assert accuracy >= TOP1_ACCURACY_FLOOR, f"top-1 accuracy {accuracy:.0%}; misses: {misses}"


def test_per_user_scoping():
    """A user's search never sees another user's agents (only their own + _shared)."""
    from orchestrator.manifest import Manifest

    private = Manifest(
        id="private_tool",
        user_id="user_a",
        type="agent",
        name="Private",
        description="A private capability owned by user_a only.",
        example_requests=["do the private user_a thing"],
    )
    registry.upsert(private)
    try:
        hits_b = registry.search("do the private user_a thing", user_id="user_b")
        assert "private_tool" not in [h.manifest.id for h in hits_b]
        hits_a = registry.search("do the private user_a thing", user_id="user_a")
        assert hits_a[0].manifest.id == "private_tool"
    finally:
        registry.get_conn().execute(
            "DELETE FROM manifest_embeddings WHERE manifest_id = %s", (private.key,)
        )
