"""The locked AgentState schema.

This is the single most important contract in the system: every dynamically
generated agent must conform to it, so it is defined explicitly here and must
not evolve implicitly.

THE WRITE RULE (enforced at runtime by the supervisor's execute node, and
restated in the generator's system prompt):

    An agent may READ the full state, but may only WRITE:
      - an append to "messages"
      - agent_scratchpad[its_own_AGENT_ID]

    No agent writes to another agent's scratchpad key. This one rule is what
    prevents dynamically added agents from corrupting each other's context.

"agent_scratchpad" deliberately has no reducer: all agent updates flow through
the supervisor's dispatcher, which validates the write rule and merges the
sub-key itself.
"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    active_agent: str | None
    agent_scratchpad: dict  # each agent may only write to its own sub-key
    routing_history: list[dict]  # for regression eval later
