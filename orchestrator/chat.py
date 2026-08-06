"""CLI REPL — the verification surface for every phase.

    uv run python -m orchestrator.chat --user <user_id>
    uv run python -m orchestrator.chat --user <user_id> --furhat <ip>

Commands:
    :prompt <agent_id> <text>     set a new live prompt version (no restart)
    :rollback <agent_id> <n>      move the current-version pointer back
    :scope <text>                 set a new live domain scope config (no restart)
    :bad                          mark the previous answer as wrong (feeds eval)
    :diagnose                     ask an LLM whether this session shows a real
                                   agent bug (see eval/diagnose.py); if one is
                                   found, automatically attempts a sandboxed,
                                   test-gated fix to that agent's own files
                                   (see agent_maker/fixer.py) — a separate
                                   code path from self-extension's own
                                   additive-only, never-edit-existing-files
                                   generation. Restores the original files
                                   untouched if no fix passes its tests.
    :quit

FurHat mode (--furhat <ip> or FURHAT_IP env var) is optional and additive: if
no FurHat is configured, reachable, or it drops mid-session, this is exactly
today's plain-text CLI loop. See orchestrator/furhat_io.py's module docstring
— the FurHat path is unverified against real hardware; this CLI loop is the
fully verified default.
"""

import argparse
import asyncio
import getpass
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from db import prompt_store
from orchestrator.furhat_io import FurHatClient, FurHatUnavailable
from orchestrator.llm import render_text
from orchestrator.supervisor import SCOPE_CONFIG_ID, build_graph


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


def _connect_furhat(ip: str) -> FurHatClient | None:
    """Best-effort connect with a short timeout. Never raises — a FurHat
    that's absent or unreachable just means this session runs as plain CLI,
    exactly like not passing --furhat at all."""
    client = FurHatClient(ip)
    try:
        asyncio.run(client.connect())
    except FurHatUnavailable as exc:
        print(f"[furhat] not reachable ({exc}) — continuing in CLI mode")
        return None
    print(f"[furhat] connected to {ip} — speaking/listening via FurHat")
    return client


def _get_user_line(furhat: FurHatClient | None) -> tuple[str | None, FurHatClient | None]:
    """Returns (line, furhat) — furhat comes back None if the connection was
    just lost, so the caller permanently falls back to CLI for the rest of
    the session. line is None on EOF/interrupt (caller should stop).

    Voice is tried first when connected; a silent/timed-out listen falls
    through to typed input for that turn, which is also how an operator
    types a : command even in FurHat mode (they're operator tools, not
    conversation — see module docstring)."""
    if furhat is not None:
        try:
            heard = asyncio.run(furhat.listen())
        except FurHatUnavailable as exc:
            print(f"[furhat] connection lost ({exc}) — continuing in CLI mode")
            furhat = None
        else:
            if heard:
                print(f"you (furhat)> {heard}")
                return heard, furhat
    try:
        line = input("\nyou> ").strip()
    except (EOFError, KeyboardInterrupt):
        return None, furhat
    return line, furhat


def _deliver_reply(furhat: FurHatClient | None, label: str, text: str) -> FurHatClient | None:
    """Speaks the reply via FurHat when connected (dropping the connection
    and falling back to CLI for the rest of the session on failure), and
    always echoes it to the terminal as the transcript."""
    if furhat is not None:
        try:
            asyncio.run(furhat.speak(text))
        except FurHatUnavailable as exc:
            print(f"[furhat] connection lost ({exc}) — continuing in CLI mode")
            furhat = None
    print(f"{label} {text}")
    return furhat


def main() -> None:
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set — copy .env.example to .env and fill it in.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True)
    parser.add_argument("--furhat", default=os.environ.get("FURHAT_IP"), help="FurHat IP address (optional)")
    args = parser.parse_args()

    prompt_store.init_db()
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": args.user}}
    furhat = _connect_furhat(args.furhat) if args.furhat else None

    state = {
        "messages": [],
        "user_id": args.user,
        "active_agent": None,
        "agent_scratchpad": {},
        "routing_history": [],
    }

    print(f"enhanced_agent REPL — user={args.user}  (:quit to exit)")
    while True:
        line, furhat = _get_user_line(furhat)
        if line is None:
            break
        line = line.strip()
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
        if line.startswith(":scope "):
            new_scope = line.split(" ", 1)[1]
            version = prompt_store.set_prompt(SCOPE_CONFIG_ID, prompt_store.SHARED_USER, new_scope)
            print(f"[domain scope -> v{version}]")
            continue
        if line == ":bad":
            from eval import usage_logger

            usage_logger.mark_last_corrected(args.user)
            print("[previous turn marked as corrected]")
            continue
        if line == ":diagnose":
            from agent_maker import fixer
            from eval import diagnose

            if not state["messages"]:
                print("[diagnose] nothing to diagnose yet — have a conversation first.")
                continue
            print("[diagnose] analyzing this session's transcript...")
            transcript = diagnose.transcript_from_state(state)
            result = diagnose.diagnose(transcript, user_id=args.user)
            print(f"\n[diagnose]\n{diagnose.format_report(result)}\n")

            if result.bug_found and result.likely_agent_id:
                print(f"[diagnose] attempting to fix {result.likely_agent_id} automatically...")
                fixed = fixer.apply_fix(args.user, result.likely_agent_id, diagnose.format_report(result))
                if fixed:
                    print(
                        f"[diagnose] {result.likely_agent_id} was fixed and its tests "
                        "pass — the running session still has the old prompt/scratchpad "
                        "state in memory, so start a fresh request to see the fix."
                    )
                else:
                    print(
                        f"[diagnose] could not produce a passing fix for "
                        f"{result.likely_agent_id}; its original files were restored "
                        "unchanged. Try `uv run python -m eval.diagnose` on a saved "
                        "transcript, or fix it by hand."
                    )
            continue

        state["messages"] = state["messages"] + [HumanMessage(content=line)]
        result = graph.invoke(state, config)

        # Phase 4: credential-approval interrupts pause the graph mid-request.
        while "__interrupt__" in result:
            resume_payload = _collect_credential_approval(result["__interrupt__"][0].value)
            result = graph.invoke(Command(resume=resume_payload), config)

        state = {k: v for k, v in result.items() if k != "__interrupt__"}
        reply_text = render_text(state["messages"][-1].content)
        furhat = _deliver_reply(furhat, f"[{state['active_agent']}]", reply_text)


if __name__ == "__main__":
    main()
