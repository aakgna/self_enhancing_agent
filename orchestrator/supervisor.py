"""Supervisor graph: START -> route -> execute -> END.

The graph shape is fixed; agents are dispatched by name inside the execute
node. That is what makes newly generated agents immediately callable — they
are new directories the dispatcher can load, not new graph nodes.

The execute node is also the enforcement point for the State write rule and
the place every trace row gets its four-part version stamp.
"""

from langchain_core.messages import AIMessage, BaseMessage

from db import prompt_store
from orchestrator import agent_loader, classifier, registry
from orchestrator.llm import agent_llm
from orchestrator.state import AgentState
from orchestrator.tracing import write_trace

from langgraph.graph import END, START, StateGraph

ALLOWED_UPDATE_KEYS = {"messages", "agent_scratchpad"}


class ScratchpadViolation(RuntimeError):
    """An agent wrote outside its own scratchpad sub-key or state slice."""


def _latest_human_text(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if getattr(message, "type", None) == "human":
            return message.content if isinstance(message.content, str) else str(message.content)
    return ""


def route_node(state: AgentState) -> dict:
    """Registry similarity search + capability-gap classifier."""
    request = _latest_human_text(state)
    hits = registry.search(request, state["user_id"], top_k=5, type_filter="agent")
    result = classifier.classify(request, hits)

    if result.decision == classifier.NO_MATCH:
        classifier.log_gap(state["user_id"], request, result.candidates)
        entry = {"request": request, "routed_to": classifier.NO_MATCH}
        write_trace(
            user_id=state["user_id"],
            node="route",
            event="no_match_found",
            payload={
                **entry,
                "candidates": [
                    {"agent_id": h.manifest.id, "similarity": round(h.similarity, 4)}
                    for h in result.candidates
                ],
            },
        )
        return {"active_agent": None, "routing_history": [*state["routing_history"], entry]}

    agent_id = result.decision
    handles = agent_loader.discover(state["user_id"])
    entry = {"request": request, "routed_to": agent_id}
    write_trace(
        user_id=state["user_id"],
        node="route",
        event="routed",
        agent_id=agent_id,
        manifest_version=handles[agent_id].manifest_version,
        payload=entry,
    )
    return {
        "active_agent": agent_id,
        "routing_history": [*state["routing_history"], entry],
    }


def make_agent_node(state: AgentState) -> dict:
    """No existing agent covers the request: generate one, end to end."""
    from agent_maker import maker  # lazy: keeps the SDK out of ordinary requests

    request = _latest_human_text(state)
    candidates = registry.search(request, state["user_id"], top_k=5, type_filter="agent")
    agent_id = maker.make_agent(state["user_id"], request, candidates)

    if agent_id is None:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I couldn't build a working agent for that request — the "
                        "generated implementation never passed its own tests. The "
                        "attempt was logged; please try rephrasing or try again."
                    )
                )
            ]
        }

    entry = {"request": request, "routed_to": agent_id, "generated": True}
    return {
        "active_agent": agent_id,
        "routing_history": [*state["routing_history"], entry],
    }


def _after_route(state: AgentState) -> str:
    return "execute" if state["active_agent"] else "make_agent"


def _after_make(state: AgentState) -> str:
    return "execute" if state["active_agent"] else END


def validate_update(agent_id: str, update: dict) -> None:
    """Enforce the write rule from orchestrator/state.py on an agent's update."""
    extra_keys = set(update) - ALLOWED_UPDATE_KEYS
    if extra_keys:
        raise ScratchpadViolation(f"{agent_id} wrote disallowed state keys: {extra_keys}")
    foreign = set(update.get("agent_scratchpad", {})) - {agent_id}
    if foreign:
        raise ScratchpadViolation(f"{agent_id} wrote foreign scratchpad keys: {foreign}")
    for message in update.get("messages", []):
        if not isinstance(message, BaseMessage):
            raise ScratchpadViolation(f"{agent_id} appended a non-message to messages")


def execute_node(state: AgentState) -> dict:
    agent_id = state["active_agent"]
    handles = agent_loader.discover(state["user_id"])
    handle = handles[agent_id]
    module = agent_loader.load_module(handle)

    prompt, prompt_version = prompt_store.seed_prompt(
        agent_id, prompt_store.SHARED_USER, module.DEFAULT_PROMPT
    )

    update = module.run(state, prompt, agent_llm())
    validate_update(agent_id, update)
    scratch_update = update.get("agent_scratchpad", {})

    write_trace(
        user_id=state["user_id"],
        node="execute",
        event="agent_ran",
        agent_id=agent_id,
        agent_version=module.AGENT_VERSION,
        prompt_version=prompt_version,
        manifest_version=handle.manifest_version,
        payload={"n_messages": len(update.get("messages", []))},
    )

    # The dispatcher owns the scratchpad merge (see state.py docstring).
    return {
        "messages": update.get("messages", []),
        "agent_scratchpad": {**state["agent_scratchpad"], **scratch_update},
    }


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("route", route_node)
    graph.add_node("execute", execute_node)
    graph.add_node("make_agent", make_agent_node)
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", _after_route, ["execute", "make_agent"])
    graph.add_conditional_edges("make_agent", _after_make, ["execute", END])
    graph.add_edge("execute", END)
    return graph.compile()
