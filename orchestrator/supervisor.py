"""Supervisor graph: START -> route -> execute -> END.

The graph shape is fixed; agents are dispatched by name inside the execute
node. That is what makes newly generated agents immediately callable — they
are new directories the dispatcher can load, not new graph nodes.

The execute node is also the enforcement point for the State write rule and
the place every trace row gets its four-part version stamp.
"""

import inspect

from langchain_core.messages import AIMessage, BaseMessage

from db import prompt_store
from orchestrator import agent_loader, classifier, mcp_bindings, registry
from orchestrator.llm import agent_llm
from orchestrator.state import AgentState
from orchestrator.tracing import write_trace

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

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


def _credential_gate(state: AgentState, handle) -> list | None:
    """Phase 4: pause for explicit approval before any real binding; return
    live tools for a bound service, else None.

    interrupt() replays the node from the top on resume, so this runs before
    any non-idempotent work in execute_node.
    """
    manifest = handle.manifest
    if not manifest.get("requires_credentials"):
        return None
    service = manifest.get("mcp_service")
    if service not in mcp_bindings.KNOWN_SERVICES:
        return None

    user_id, request = state["user_id"], _latest_human_text(state)
    if not mcp_bindings.is_bound(user_id, service):
        service_entry = mcp_bindings.KNOWN_SERVICES[service]
        decision = interrupt(
            {
                "type": "credential_approval",
                "service": service,
                "scope": service_entry["scope"],
                "note": service_entry.get("real_service_note"),
                "agent_id": handle.agent_id,
                "reason": request,
            }
        )
        if not (isinstance(decision, dict) and decision.get("approved")):
            write_trace(
                user_id=user_id, node="execute", event="binding_declined",
                agent_id=handle.agent_id, payload={"service": service},
            )
            return None
        mcp_bindings.bind(user_id, service, request)
        write_trace(
            user_id=user_id, node="execute", event="capability_bound",
            agent_id=handle.agent_id, payload={"service": service},
        )
    return mcp_bindings.get_tools(user_id, service)


def execute_node(state: AgentState) -> dict:
    agent_id = state["active_agent"]
    handles = agent_loader.discover(state["user_id"])
    handle = handles[agent_id]

    tools = _credential_gate(state, handle)

    module = agent_loader.load_module(handle)
    # Prompt scope follows the agent's OWNING user (its manifest), not the
    # caller: "_shared" for hand-written agents, the generating user for
    # per-user generated ones. Using SHARED_USER unconditionally here would
    # collide two different users' same-named generated agents into one
    # global prompt — a real violation of the per-user isolation this system
    # is designed around.
    prompt_owner = handle.manifest.get("user_id", prompt_store.SHARED_USER)
    prompt, prompt_version = prompt_store.seed_prompt(
        agent_id, prompt_owner, module.DEFAULT_PROMPT
    )

    if "tools" in inspect.signature(module.run).parameters:
        update = module.run(state, prompt, agent_llm(), tools=tools)
    else:
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

    from eval import usage_logger  # Phase 5: every turn feeds the replay corpus

    usage_logger.log_turn(state["user_id"], _latest_human_text(state), agent_id)

    # The dispatcher owns the scratchpad merge (see state.py docstring).
    return {
        "messages": update.get("messages", []),
        "agent_scratchpad": {**state["agent_scratchpad"], **scratch_update},
    }


def build_graph(checkpointer=None):
    graph = StateGraph(AgentState)
    graph.add_node("route", route_node)
    graph.add_node("execute", execute_node)
    graph.add_node("make_agent", make_agent_node)
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", _after_route, ["execute", "make_agent"])
    graph.add_conditional_edges("make_agent", _after_make, ["execute", END])
    graph.add_edge("execute", END)
    return graph.compile(checkpointer=checkpointer)
