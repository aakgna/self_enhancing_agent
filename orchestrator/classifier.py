"""Capability-gap classifier: "an existing agent handles this" vs "nothing does".

Two stages, both cheap:
  1. Similarity floor — candidates below SIMILARITY_FLOOR are not even shown
     to the classifier; if nothing clears the floor, that's a no-match with
     zero LLM calls.
  2. A small fast model call (Haiku) over the surviving candidates. This is a
     routing decision, not a reasoning task.

Every no-match is logged to capability_gaps: it is the Agent Maker's backlog
and the regression eval's seed data.
"""

import json
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from db.conn import get_conn
from orchestrator.llm import fast_llm
from orchestrator.registry import SearchHit

# Tuned against eval/labeled_requests.json via eval/tune_threshold.py.
# The sweep showed the floor CANNOT separate covered from uncovered on its own:
# short-text cosine noise puts nearly everything above 0.45, and floors high
# enough to catch gaps (~0.7) start rejecting genuinely covered requests.
# So the floor is a conservative cost-saving pre-filter only — kept low so no
# covered request is ever hidden from the Haiku stage, which makes the call.
SIMILARITY_FLOOR = 0.45

NO_MATCH = "no_match_found"


class ClassifierVerdict(BaseModel):
    decision: Literal["route_to_existing", "no_match_found"] = Field(
        description="route_to_existing only if one candidate's stated scope genuinely covers the request"
    )
    agent_id: str | None = Field(
        default=None, description="Required when decision is route_to_existing"
    )


@dataclass
class Classification:
    decision: str  # an agent_id, or NO_MATCH
    candidates: list[SearchHit]


_PROMPT = """\
You are a routing classifier. Decide whether one of the candidate agents can \
genuinely handle the user's request, based strictly on each agent's stated scope.

Rules:
- Route to an agent only if the request falls squarely inside its description. \
Respect the "not for" exclusions in descriptions.
- Do not stretch a broad-sounding agent to cover a request that needs a \
different capability, external data, credentials, or a transformation it does \
not claim.
- If no candidate genuinely covers the request, answer no_match_found. \
Declining to route is a good outcome, not a failure: it triggers creation of a \
proper agent.

Candidates:
{candidates}

User request: {request}"""


def classify(request: str, hits: list[SearchHit], llm=None) -> Classification:
    viable = [h for h in hits if h.similarity >= SIMILARITY_FLOOR]
    if not viable:
        return Classification(decision=NO_MATCH, candidates=hits)

    catalog = "\n".join(
        f"- {h.manifest.id} (similarity {h.similarity:.2f}): {h.manifest.description}"
        for h in viable
    )
    chooser = (llm or fast_llm()).with_structured_output(ClassifierVerdict)
    verdict = chooser.invoke(_PROMPT.format(candidates=catalog, request=request))

    valid_ids = {h.manifest.id for h in viable}
    if verdict.decision == "route_to_existing" and verdict.agent_id in valid_ids:
        return Classification(decision=verdict.agent_id, candidates=viable)
    return Classification(decision=NO_MATCH, candidates=viable)


def log_gap(user_id: str, request: str, candidates: list[SearchHit]) -> None:
    get_conn().execute(
        "INSERT INTO capability_gaps (user_id, request, rejected_matches) VALUES (%s, %s, %s)",
        (
            user_id,
            request,
            json.dumps(
                [
                    {"agent_id": h.manifest.id, "similarity": round(h.similarity, 4)}
                    for h in candidates
                ]
            ),
        ),
    )
