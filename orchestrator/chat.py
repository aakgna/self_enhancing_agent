"""CLI REPL — the verification surface for every phase.

    uv run python -m orchestrator.chat --user <user_id>

Commands:
    :prompt <agent_id> <text>     set a new live prompt version (no restart)
    :rollback <agent_id> <n>      move the current-version pointer back
    :quit
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from db import prompt_store
from orchestrator.supervisor import build_graph


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

        state["messages"] = state["messages"] + [HumanMessage(content=line)]
        result = graph.invoke(state, config)

        # Phase 4: credential-approval interrupts pause the graph mid-request.
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            print(
                f"\n[approval needed] agent '{payload['agent_id']}' wants to connect "
                f"to '{payload['service']}'\n"
                f"  scope:  {payload['scope']}\n"
                f"  reason: your request \"{payload['reason']}\""
            )
            answer = input("  approve this connection? [y/N] ").strip().lower()
            result = graph.invoke(Command(resume={"approved": answer == "y"}), config)

        state = {k: v for k, v in result.items() if k != "__interrupt__"}
        print(f"[{state['active_agent']}] {state['messages'][-1].content}")


if __name__ == "__main__":
    main()
