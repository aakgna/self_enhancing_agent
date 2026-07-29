"""Phase 0 router: a cheap structured LLM choice among the discovered agents.

Phase 2 replaces the naive "show every agent" listing with registry similarity
search + the capability-gap classifier; the node shape stays the same.
"""

from pydantic import BaseModel, Field

from orchestrator.agent_loader import AgentHandle
from orchestrator.llm import fast_llm

FALLBACK_AGENT = "general_qa"


class RouteChoice(BaseModel):
    agent_id: str = Field(description="The id of the single best agent for this request")


def route(request: str, handles: dict[str, AgentHandle]) -> str:
    catalog = "\n".join(
        f"- {aid}: {h.manifest.get('description', '')}" for aid, h in handles.items()
    )
    chooser = fast_llm().with_structured_output(RouteChoice)
    choice = chooser.invoke(
        "Pick the single best agent for the user's request.\n\n"
        f"Available agents:\n{catalog}\n\n"
        f"User request: {request}"
    )
    if choice.agent_id in handles:
        return choice.agent_id
    return FALLBACK_AGENT if FALLBACK_AGENT in handles else next(iter(handles))
