# enhanced_agent — self-extending multi-agent system

A LangGraph supervisor that routes user requests to per-user agents — and when
no agent covers a request, **plans, generates, sandbox-tests, and registers a
new one automatically**.

Locked design decisions: LangGraph/Python · per-user agents · create-only
lifecycle · additive-only generation (the generator never edits existing
files) · prompts versioned in Postgres · async, silent, non-blocking
regression testing · codegen via an agentic coding harness (Claude Agent SDK),
not a hand-rolled file-writing loop.

## Setup

```sh
brew install uv                     # if missing
createdb enhanced_agent             # local Homebrew PostgreSQL 15
cp .env.example .env                # then paste your ANTHROPIC_API_KEY
uv sync
```

## Run

```sh
uv run python -m orchestrator.chat --user <your_id>
```

REPL commands: `:prompt <agent_id> <text>` (live prompt edit, no restart),
`:rollback <agent_id> <n>`, `:quit`.

## Tests

```sh
uv run pytest                                   # repo-level unit tests
uv run pytest agents/_shared/general_qa         # agent suites run per-directory
uv run pytest agents/_shared/summarizer         # (same way the sandbox runs them)
```

## Layout

```
agents/_shared/<id>/        hand-written agents: agent.py, manifest.json, test_agent.py
agents/<user_id>/<id>/      generated per-user agents (same contract)
tools/shared/, tools/<user_id>/   tools
orchestrator/               state (the locked schema), supervisor, router, loader, tracing
agent_maker/                planner, generator, sandbox runner, registrar (Phase 3)
db/                         schema.sql, prompt_store
eval/                       labeled routing set, regression runner, usage logging
```

## The one rule that keeps generated agents safe

An agent may read the full state but may only append to `messages` and write
`agent_scratchpad[its_own_agent_id]`. The supervisor enforces this at runtime
(`orchestrator/supervisor.py::validate_update`) and the generator restates it
in its system prompt.

## Observability

Every node writes a `traces` row stamped with `agent_id`, `agent_version`,
`prompt_version`, `manifest_version` — set up in Phase 0 so every later phase
inherits it.

## Production notes

The generation sandbox uses macOS `sandbox-exec` locally (network denied,
env scrubbed, writes confined to the new agent's directory). In production,
swap `agent_maker/sandbox_runner.py` for Docker or E2B — the interface is a
single `run_tests(agent_dir)` function.
