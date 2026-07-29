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

## Credentialed capabilities

`orchestrator/mcp_bindings.py::KNOWN_SERVICES` is a generic credential
framework, not one hardcoded email integration. Each entry declares a
`credential_type` (`none` / `api_key` / `app_password` / `oauth2`), the
fields it needs, and the steps to obtain them — the approval flow in
`chat.py` renders whichever of those apply generically, so adding a new
credentialed service never touches the REPL or the graph.

- **`mock_email`** — `credential_type: none`. Zero-friction stdio MCP demo
  with canned data; the system auto-issues a token, nothing to type. This is
  the only service offered to the Agent Maker's automatic generation
  (`planner_eligible: True`) — see below for why.
- **`gmail_imap`** — `credential_type: app_password`. Real, working Gmail
  access over IMAP using a 16-character
  [App Password](https://myaccount.google.com/apppasswords) (requires
  2-Step Verification on the account) — not full OAuth, so it needs no
  registered Google Cloud client. `planner_eligible: False` on purpose: it's
  fully wired (`tools/shared/gmail_imap/tool.py`, tested offline against a
  faked `imaplib`) but deliberately kept out of the automatic catalog, so a
  casual "check my email" doesn't start asking for a real credential. Flip
  the flag in `mcp_bindings.py` to make it auto-offered, or reference
  `"mcp_service": "gmail_imap"` directly in a hand-written agent's manifest.
  When approved, the App Password is typed into a hidden `getpass()` prompt
  **in your own terminal** — it is never sent through the chat/model, and
  the local credential store (`local_cache/credentials.json`) is
  chmod'd `0600`/`0700` (owner-only). That store is a local-dev convenience,
  not a production secrets manager — swap it for an OS keychain or a real
  vault before this leaves your machine.
- **`oauth2`** (type, no concrete entry yet) — the framework supports it
  (approval renders `acquisition_instructions` for the external client
  registration a user must do first), but the actual browser-redirect
  consent flow isn't implemented: shipping untested OAuth token-exchange
  code is worse than clearly declining, since auth bugs fail silently in
  exactly the way you don't want. Add a real Google Cloud OAuth client and
  `google-auth-oauthlib`'s `InstalledAppFlow` to build this out.

## Live verification (needs ANTHROPIC_API_KEY in .env)

Everything mechanical is covered by the offline suite (55 tests). Three
checks exercise the real models:

```sh
uv run python -m eval.tune_threshold --live   # Phase 2 exit: end-to-end routing accuracy
uv run python -m orchestrator.chat --user me  # Phase 0 exit: ask a question, then paste text to summarize
# Phase 3 exit, inside the REPL: "convert this text to pig latin: hello world"
#   -> plans, generates, sandbox-tests, registers, and answers; then check
#      `git status` (only agents/me/... created) and re-ask to see it route
#      directly. "check my email for anything urgent" exercises the Phase 4
#      approval gate against the mock email service (credential_type: none —
#      approve with 'y', nothing to type). Real Gmail isn't reachable this
#      way (gmail_imap is opt-in, see above).
```

## Production notes

The generation sandbox uses macOS `sandbox-exec` locally (network denied,
env scrubbed, writes confined to the new agent's directory). In production,
swap `agent_maker/sandbox_runner.py` for Docker or E2B — the interface is a
single `run_tests(agent_dir)` function.
