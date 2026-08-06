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
from orchestrator import agent_loader, classifier, mcp_bindings, registry, scope_gate
from orchestrator.llm import agent_llm
from orchestrator.state import AgentState
from orchestrator.tracing import write_trace

DECLINED_OUT_OF_SCOPE = "declined_out_of_scope"
SCOPE_CONFIG_ID = "_scope"

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


def _handle_capability_gap(state: AgentState, request: str, candidates: list) -> dict:
    """No existing agent matched (a fresh miss, or a dangling registry
    pointer that was just purged). Log the gap, then run the domain scope
    gate — only a request that clears it may proceed to generation. The
    scope gate runs HERE, not before the registry search: an existing,
    already-in-domain agent should never pay for a redundant scope judgment
    on every turn, and a match against it is not re-litigated. The trade-off
    this implies — an out-of-domain agent left sitting in the registry would
    become reachable again via ordinary classifier matching — is why any
    agent outside the declared domain must be purged from the registry, not
    just left unrouted (see :scope's docstring / README's domain scope gate
    section for which agents that applies to)."""
    classifier.log_gap(state["user_id"], request, candidates)
    no_match_entry = {"request": request, "routed_to": classifier.NO_MATCH}
    write_trace(
        user_id=state["user_id"],
        node="route",
        event="no_match_found",
        payload={
            **no_match_entry,
            "candidates": [
                {"agent_id": h.manifest.id, "similarity": round(h.similarity, 4)}
                for h in candidates
            ],
        },
    )
    history = [*state["routing_history"], no_match_entry]

    scope_prompt, _ = prompt_store.seed_prompt(
        SCOPE_CONFIG_ID, prompt_store.SHARED_USER, scope_gate.DEFAULT_SCOPE_PROMPT
    )
    verdict = scope_gate.check(request, scope_prompt)
    if not verdict.in_scope:
        decline_entry = {"request": request, "routed_to": DECLINED_OUT_OF_SCOPE}
        write_trace(
            user_id=state["user_id"],
            node="route",
            event=DECLINED_OUT_OF_SCOPE,
            payload={**decline_entry, "reason": verdict.reason},
        )
        return {
            "active_agent": None,
            "messages": [AIMessage(content=verdict.reason)],
            "routing_history": [*history, decline_entry],
        }
    return {"active_agent": None, "routing_history": history}


def route_node(state: AgentState) -> dict:
    """Registry similarity search + capability-gap classifier first; the
    domain scope gate only runs on a miss, immediately before generation
    would otherwise be attempted (see _handle_capability_gap)."""
    request = _latest_human_text(state)

    # 0. Stickiness: an agent mid multi-turn flow (e.g. checkin) owns the next
    # turn outright. Without this, a short reply like "yes" or a spoken DOB
    # would be reclassified from scratch every turn (classify() never sees
    # conversation history) and could be declined as out-of-scope or sent to
    # the planner to build an agent for "yes". Any multi-turn agent opts in by
    # setting agent_scratchpad[its_id]["awaiting_reply"] = True while its flow
    # is incomplete; single-shot agents never set it, so their behavior is
    # unchanged.
    active = state.get("active_agent")
    if active and state["agent_scratchpad"].get(active, {}).get("awaiting_reply"):
        entry = {"request": request, "routed_to": active}
        return {"active_agent": active, "routing_history": [*state["routing_history"], entry]}

    hits = registry.search(request, state["user_id"], top_k=5, type_filter="agent")
    result = classifier.classify(request, hits)

    if result.decision == classifier.NO_MATCH:
        return _handle_capability_gap(state, request, result.candidates)

    agent_id = result.decision
    handles = agent_loader.discover(state["user_id"])

    if agent_id not in handles:
        # The registry named an agent whose files don't exist — a dangling
        # pointer, not a real agent to route to (see registrar.purge_agent's
        # docstring for how this class of drift happens). Self-heal: purge
        # the stale entry and fall through to the same capability-gap path,
        # which is already wired to generate a fresh one.
        from agent_maker import registrar

        registrar.purge_agent(state["user_id"], agent_id, delete_files=False)
        write_trace(
            user_id=state["user_id"],
            node="route",
            event="orphaned_registry_entry_purged",
            agent_id=agent_id,
            payload={"request": request},
        )
        return _handle_capability_gap(state, request, result.candidates)

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
    if state["active_agent"]:
        return "execute"
    if state["routing_history"] and state["routing_history"][-1]["routed_to"] == DECLINED_OUT_OF_SCOPE:
        return END
    return "make_agent"


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

    if mcp_bindings.is_bound(user_id, service):
        try:
            return mcp_bindings.get_tools(user_id, service)
        except mcp_bindings.CredentialUnavailable as exc:
            # Self-heal rather than crash: a previously-approved binding's
            # stored credential is unusable (e.g. an old storage format from
            # before a schema change — this happened live). Forget it and
            # fall through to request a fresh approval below, exactly as if
            # it had never been bound.
            mcp_bindings.forget(user_id, service)
            write_trace(
                user_id=user_id, node="execute", event="credential_reset",
                agent_id=handle.agent_id, payload={"service": service, "reason": str(exc)},
            )

    if not mcp_bindings.is_bound(user_id, service):
        service_entry = mcp_bindings.KNOWN_SERVICES[service]
        # One round trip: the approval prompt already carries everything
        # needed to obtain the credential (instructions + field labels), so
        # the REPL can collect it and resume once, rather than a second
        # interrupt just to ask "now give me the value".
        decision = interrupt(
            {
                "type": "credential_approval",
                "service": service,
                "scope": service_entry["scope"],
                "note": service_entry.get("real_service_note"),
                "credential_type": service_entry["credential_type"],
                "fields": service_entry["credential_fields"],
                "acquisition_instructions": service_entry["acquisition_instructions"],
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
        mcp_bindings.bind(user_id, service, request, values=decision.get("values"))
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
    graph.add_conditional_edges("route", _after_route, ["execute", "make_agent", END])
    graph.add_conditional_edges("make_agent", _after_make, ["execute", END])
    graph.add_edge("execute", END)
    return graph.compile(checkpointer=checkpointer)
