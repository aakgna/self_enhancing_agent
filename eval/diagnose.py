"""Diagnose a conversation transcript for a genuine agent bug — dev-facing,
manual, diagnosis only. Never edits code: agent_maker/generator.py's
additive-only rule (never touch an existing agent's files) is a property of
the *automated generation* pipeline, and this tool is a deliberately
separate, human-in-the-loop path — it produces a bug report (root cause,
affected agent/file, suggested fix) for a person (or a following dev
session) to act on, the same way this project's own maintainer diagnosed
and fixed agents/me/appointment_booking's slot-substitution bug by hand
before this tool existed.

Two entry points:
  - diagnose(transcript, ...) — importable, used by orchestrator/chat.py's
    `:diagnose` REPL command (diagnoses the live session so far).
  - `uv run python -m eval.diagnose [file]` — reads a pasted transcript from
    a file argument or stdin, for exactly the workflow of handing over a
    terminal transcript for review.

Two-pass: an initial read from the transcript alone identifies the likely
culprit agent; if one is identified and its source is available, a second
pass re-diagnoses with that agent's actual agent.py attached, so the root
cause and suggested fix can cite real code rather than guessing from
symptoms alone.
"""

import sys
from pathlib import Path

from pydantic import BaseModel, Field

from orchestrator.llm import agent_llm


class Diagnosis(BaseModel):
    bug_found: bool = Field(description="True only for a genuine implementation bug, not a missing capability or the user changing their mind")
    summary: str = Field(default="", description="One sentence: what went wrong, if anything")
    likely_agent_id: str | None = Field(default=None, description="The agent whose code most likely caused this, if identifiable from its [agent_id] labels in the transcript")
    root_cause: str = Field(default="", description="The specific mechanism in the code that produced the wrong behavior, not just a description of the symptom")
    evidence: str = Field(default="", description="Direct quote(s) from the transcript that demonstrate the bug")
    suggested_fix: str = Field(default="", description="A concrete, actionable suggestion for what to change")
    confidence: str = Field(default="low", description="low | medium | high")


_INITIAL_PROMPT = """\
You are reviewing a transcript of a conversation between a user and a \
multi-agent system, to determine whether any agent behaved incorrectly (a \
genuine implementation bug), not just made an imperfect but reasonable \
choice.

Look specifically for:
  - Contradictions: the agent states or implies two different facts about \
the same thing (e.g. offers something for one date, then discusses a \
different date as if it were the same request).
  - Requests being silently ignored: a user's stated preference is not \
honored even though the agent's reply implies it was taken into account.
  - Dead ends: the same question or state repeats without progress despite \
the user trying to move the conversation forward.
  - Output that looks synthetic, hardcoded, or disconnected from what the \
user actually asked for.

Do NOT flag: a capability the system doesn't have (a clear decline is \
working as intended), the user simply changing their mind, or a stylistic \
phrasing choice you'd have made differently.

Known agents in this system, if it helps you attribute a turn to one: \
{agent_ids}

Transcript:
{transcript}

If you find a genuine bug, cite the exact turn(s) that demonstrate it as \
evidence, name the single agent whose code most likely caused it (from its \
bracketed [agent_id] label), and explain your best guess at the root cause. \
If nothing looks like a real bug, say so plainly — do not manufacture a \
finding to have something to report.
"""

_WITH_SOURCE_PROMPT = """\
{initial_prompt}

You previously suspected `{suspect_id}` was responsible. Here is that \
agent's actual implementation — use it to confirm or correct your \
diagnosis, and to make root_cause and suggested_fix specific to the real \
code (cite the actual function/logic responsible) rather than guessing \
from symptoms alone. If the source shows this agent is NOT actually at \
fault, say so and reconsider.

agents/{suspect_id}/agent.py:
```python
{source}
```
"""


def _run(prompt: str, llm) -> Diagnosis:
    return llm.with_structured_output(Diagnosis, method="json_schema").invoke(prompt)


def _read_agent_source(user_id: str, agent_id: str) -> str | None:
    from orchestrator.agent_loader import discover

    handle = discover(user_id).get(agent_id)
    if handle is None:
        return None
    agent_py = handle.directory / "agent.py"
    return agent_py.read_text() if agent_py.is_file() else None


def diagnose(
    transcript: str,
    user_id: str | None = None,
    agent_ids: list[str] | None = None,
    llm=None,
) -> Diagnosis:
    """user_id enables the second, source-grounded pass (via
    orchestrator.agent_loader.discover) — omit it (as the standalone CLI
    does, not knowing which user's agents to look at) to get transcript-only
    diagnosis. agent_ids, if given, are just named in the prompt as known
    agents; the model still attributes blame from the transcript's own
    [agent_id] labels."""
    llm = llm or agent_llm()
    ids_text = ", ".join(agent_ids) if agent_ids else "(infer from the transcript's own [agent_id] labels)"
    initial_prompt = _INITIAL_PROMPT.format(agent_ids=ids_text, transcript=transcript.strip())

    initial = _run(initial_prompt, llm)
    if not (initial.bug_found and initial.likely_agent_id and user_id):
        return initial

    source = _read_agent_source(user_id, initial.likely_agent_id)
    if source is None:
        return initial

    return _run(
        _WITH_SOURCE_PROMPT.format(initial_prompt=initial_prompt, suspect_id=initial.likely_agent_id, source=source),
        llm,
    )


def format_report(diagnosis: Diagnosis) -> str:
    if not diagnosis.bug_found:
        return f"No bug found. {diagnosis.summary}".strip()
    lines = [
        f"BUG FOUND ({diagnosis.confidence} confidence): {diagnosis.summary}",
        f"Agent: {diagnosis.likely_agent_id or 'unknown'}",
        f"Root cause: {diagnosis.root_cause}",
        f"Evidence: {diagnosis.evidence}",
        f"Suggested fix: {diagnosis.suggested_fix}",
    ]
    return "\n".join(lines)


def transcript_from_state(state: dict) -> str:
    """Reconstructs the exact `you> ...` / `[agent_id] ...` terminal format
    from a live AgentState, so `:diagnose` in the REPL sees precisely what
    the user saw — routing_history has one {request, routed_to} entry per
    human turn, in the same order as the Human/AI message pairs, since both
    are appended together by the same route -> execute cycle."""
    humans = [m for m in state["messages"] if getattr(m, "type", None) == "human"]
    ais = [m for m in state["messages"] if getattr(m, "type", None) == "ai"]
    routing = state.get("routing_history", [])

    lines = []
    for i, human in enumerate(humans):
        lines.append(f"you> {human.content}")
        if i < len(ais):
            label = routing[i]["routed_to"] if i < len(routing) and routing[i].get("routed_to") else "?"
            from orchestrator.llm import render_text

            lines.append(f"[{label}] {render_text(ais[i].content)}")
    return "\n".join(lines)


if __name__ == "__main__":
    text = Path(sys.argv[1]).read_text() if len(sys.argv) > 1 else sys.stdin.read()
    print(format_report(diagnose(text)))
