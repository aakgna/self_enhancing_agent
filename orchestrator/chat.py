"""CLI REPL — the verification surface for every phase.

    uv run python -m orchestrator.chat --user <user_id>

Commands:
    :prompt <agent_id> <text>     set a new live prompt version (no restart)
    :rollback <agent_id> <n>      move the current-version pointer back
    :bad                          mark the previous answer as wrong (feeds eval)
    :quit
"""

import argparse
import getpass
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from db import prompt_store
from orchestrator.supervisor import build_graph


def _collect_credential_approval(payload: dict) -> dict:
    """Prints what's needed and why, then — for credential types that need a
    real value — collects it. Secret fields are read with getpass (hidden,
    never echoed, never sent anywhere but stored locally by mcp_bindings)."""
    print(
        f"\n[approval needed] agent '{payload['agent_id']}' wants to connect "
        f"to '{payload['service']}'\n"
        f"  scope:  {payload['scope']}\n"
        f"  reason: your request \"{payload['reason']}\""
    )
    if payload.get("note"):
        print(f"  note:   {payload['note']}")

    if payload["credential_type"] == "oauth2":
        print(
            "  This service uses OAuth and needs a one-time setup outside this "
            "app before it can be connected:"
        )
        for step in payload["acquisition_instructions"]:
            print(f"    - {step}")
        print("  (interactive OAuth sign-in is not implemented yet)")
        return {"approved": False}

    answer = input("  approve this connection? [y/N] ").strip().lower()
    if answer != "y":
        return {"approved": False}

    if not payload["fields"]:
        return {"approved": True}

    if payload["acquisition_instructions"]:
        print("  To get what's needed:")
        for step in payload["acquisition_instructions"]:
            print(f"    - {step}")

    values = {}
    for field in payload["fields"]:
        prompt_text = f"  {field['label']}: "
        values[field["name"]] = (
            getpass.getpass(prompt_text) if field["secret"] else input(prompt_text)
        )
    return {"approved": True, "values": values}


def main() -> None:
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set — copy .env.example to .env and fill it in.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True)
    args = parser.parse_args()

    prompt_store.init_db()
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": args.user}}

    state = {
        "messages": [],
        "user_id": args.user,
        "active_agent": None,
        "agent_scratchpad": {},
        "routing_history": [],
    }

    print(f"enhanced_agent REPL — user={args.user}  (:quit to exit)")
    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line == ":quit":
            break
        if line.startswith(":prompt "):
            _, agent_id, new_prompt = line.split(" ", 2)
            version = prompt_store.set_prompt(agent_id, prompt_store.SHARED_USER, new_prompt)
            print(f"[{agent_id} prompt -> v{version}]")
            continue
        if line.startswith(":rollback "):
            _, agent_id, target = line.split(" ", 2)
            prompt_store.rollback(agent_id, prompt_store.SHARED_USER, int(target))
            print(f"[{agent_id} prompt pointer -> v{target}]")
            continue
        if line == ":bad":
            from eval import usage_logger

            usage_logger.mark_last_corrected(args.user)
            print("[previous turn marked as corrected]")
            continue

        state["messages"] = state["messages"] + [HumanMessage(content=line)]
        result = graph.invoke(state, config)

        # Phase 4: credential-approval interrupts pause the graph mid-request.
        while "__interrupt__" in result:
            resume_payload = _collect_credential_approval(result["__interrupt__"][0].value)
            result = graph.invoke(Command(resume=resume_payload), config)

        state = {k: v for k, v in result.items() if k != "__interrupt__"}
        print(f"[{state['active_agent']}] {state['messages'][-1].content}")


if __name__ == "__main__":
    main()
