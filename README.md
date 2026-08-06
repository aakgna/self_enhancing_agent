# uv run python -m orchestrator.chat --user me

# enhanced_agent — self-extending multi-agent system

A LangGraph supervisor that routes user requests to per-user agents — and when
no agent covers a request, **plans, generates, sandbox-tests, and registers a
new one automatically**.

Locked design decisions: LangGraph/Python · per-user agents · create-only
lifecycle · additive-only generation (the generator never edits existing
files) · prompts versioned in Postgres · async, silent, non-blocking
regression testing · codegen via an agentic coding harness (Claude Agent SDK),
not a hand-rolled file-writing loop · a versioned, editable domain scope gate
that decides whether an _uncovered_ request may be built into a new agent at
all.

## LLM fallback

`orchestrator/llm.py`'s `agent_llm()`/`fast_llm()` wrap every LLM call
(routing, the scope gate, every hand-written agent's replies, the planner's
spec generation) in `FallbackChatModel`, which walks an **ordered provider
chain — Amazon Bedrock, then the direct Anthropic API, then OpenAI's
`gpt-4o-mini`** — instead of calling Anthropic directly. Bedrock goes first
because it's the provider actually reachable when the direct Anthropic API
is degraded for this account (exhausted credits, an outage, a rate limit),
while still serving the same Claude models. Each step only engages if that
provider's own credentials are set in `.env` (`AWS_ACCESS_KEY_ID`/
`AWS_SECRET_ACCESS_KEY` + `BEDROCK_MODEL_ID`, region via `AWS_REGION`
(default `us-east-1`); `ANTHROPIC_API_KEY`; or `OPENAI_API_KEY`); a
configured provider erroring on its own turn (`botocore`'s
`ClientError`/`BotoCoreError`, `anthropic.AnthropicError`, or
`openai.OpenAIError`) moves on to the next configured step. If nothing in
the chain is configured, or every configured step fails, the most recent
real failure is re-raised as-is rather than a synthetic "not configured"
error. For structured output (`.with_structured_output(...)`), only the
direct Anthropic leg gets the caller's requested `method` (e.g.
`method="json_schema"`, used by the planner) — Bedrock and OpenAI always get
`method="function_calling"` regardless of chain position, since both of
their native/default structured-output modes force
`additionalProperties: false` recursively, which silently empties
`AgentSpec`'s intentionally freeform `input_schema`/`output_schema: dict`
fields (a crash on OpenAI, silent data loss on Bedrock) — confirmed by
reading `langchain_aws.ChatBedrockConverse`'s own
`_set_additional_properties_false`, not assumed safe just because it's
still Claude.

**Code generation has its own, separate Bedrock-first fallback** —
`agent_maker/generator.py` drives the Claude Agent SDK directly (a
subprocess, not a LangChain chat model), so it can't reuse the chain above,
and there's no OpenAI leg (the harness only talks to Claude models).
`agent_maker/maker.py::_generate_with_retries` prefers Bedrock first
whenever configured (via the CLI's own `CLAUDE_CODE_USE_BEDROCK=1` switch)
and falls back to the direct Anthropic API if Bedrock's own model call fails
upstream. That upstream failure can't be caught as a Python exception — the
CLI subprocess absorbs it and reports it structurally via a `ResultMessage`'s
`is_error`/`api_error_status` fields instead, which looks identical to a
genuine code/test bug unless checked explicitly
(`generator.py::_is_upstream_api_failure`); without that check, a whole
sandbox-retry budget would be burned on a request that could never have
produced any files. If Bedrock isn't configured, generation just uses the
direct Anthropic API as before, with no fallback beyond it.

## Domain scope gate

The system is scoped to a declared domain (by default: a clinic front desk —
same-day check-in, appointment scheduling, and facility/treatment/billing
FAQ). `orchestrator/supervisor.py::route_node` checks the registry and
classifier **first**: an existing, already-in-domain agent match is routed to
directly, with no scope judgment on that turn — an agent's presence in the
registry already implies it was approved for this domain, either by being
hand-written to fit it or by having passed the gate at generation time.
`orchestrator/scope_gate.py` only runs on a miss (`route_node::
_handle_capability_gap`), immediately before generation would otherwise be
attempted: if the request doesn't fit the domain, decline outright — nothing
logged as a capability gap beyond the initial miss, no agent built. If it
does fit, fall through to the planner exactly as before.

This means an agent still sitting in the registry but outside the domain
would be directly reachable via ordinary classifier matching (the scope gate
never gets a turn once something already matches). The pre-existing
hand-written demo agents (`calculus_solver`, `pig_latin_converter`,
`insurance_coverage_checker`, `gmail_recent_reader`) were outside the
front-desk domain, so they've been fully removed — both the registry row
(`agent_maker/registrar.py::purge_agent`) and the files. `registry.backfill()`
now skips dot-prefixed agent directories, matching `agent_loader._scan()`'s
existing quarantine convention — this matters because `purge_agent(...,
delete_files=False)` only removes the registry row, and without the skip,
the next `backfill()` call (routine — several scripts call it) would
silently re-register anything still sitting in a normal (non-dot) directory,
undoing the purge. This bit a real session.

The domain definition (a core function statement plus concrete in-scope/
out-of-scope examples — boundary examples measurably help, not just one
free-text sentence) is versioned Postgres text, reusing `db/prompt_store.py`
under the reserved id `_scope`, and is a live LLM judgment call on every
miss — not a keyword allowlist. It generalizes to phrasings never listed in
the examples (that's the whole point of grounding it in a persona/mission
statement — "operate as a clinic front-desk receptionist" — rather than an
enumerated list). Edit it live with `:scope <text>` in the REPL, same as
`:prompt`.

An agent mid multi-turn flow is never re-routed or scope-checked turn to
turn: it sets `agent_scratchpad[its_id]["awaiting_reply"] = True` while its
flow is incomplete, and `route_node` gives it the next turn outright.
`checkin` is the worked example, and — critically — it's also now one of the
generator's three worked examples (`agent_maker/prompts/
generator_system_prompt.md`), with an explicit "Multi-turn agents" section
teaching this convention. Without that, a _generated_ multi-turn agent (e.g.
a scheduler collecting a name, then a date, then confirming) had no model of
this pattern at all: a follow-up reply looked like a brand-new, unrelated
request to `route_node`, which could misroute it or — worse — trigger
generation of an entirely new, duplicate agent for what was really just the
answer to the first agent's own question. This was a real, live bug, not a
hypothetical.

## Agents

- **`faq`** (`agents/_shared/faq/`) — answers general clinic questions (hours,
  parking, wayfinding, billing, insurance accepted) from a static knowledge
  base ported from a companion FurHat receptionist project. Local-only: if a
  question isn't in the knowledge base, it says so instead of guessing.
- **`checkin`** (`agents/_shared/checkin/`) — walks a patient through same-day
  check-in: identity by name + date of birth against a local sample EHR
  (`tools/shared/ehr_csv/`), today's appointment, consent, and intake forms.
  Ported from the same FurHat project's check-in specialist, conversational
  flow only — no camera/ID-card OCR, no insurance-card scan, no tablet
  dashboard (those are FurHat-hardware-specific and out of place in a
  CLI/voice-text agent). Also the generator's worked example for multi-turn
  agents (see "Domain scope gate" above).
- `general_qa`, `summarizer` — general-purpose demo agents. Outside the
  current domain scope config, but kept (unlike the four removed above)
  because the generator reads their files directly as worked examples on
  every generation (`agent_maker/generator.py::build_system_prompt`) — do
  not delete these two.
- `agents/me/` (per-user generated agents) starts empty. Self-extension
  recreates it on demand — nothing to set up. (Two earlier generated
  agents, `appointment_lookup` and `appointment_scheduler`, were purged: both
  predated the EHR-store grounding below and had invented their own
  incompatible fake data instead of using it — `appointment_scheduler` was
  additionally fully broken in production, since its `mcp_service: null`
  meant `_credential_gate` always handed it `tools=None`. A fresh request
  will regenerate equivalent agents grounded in the real store.)

## The local hospital "DB"

`tools/shared/ehr_csv/` — plain CSV files under `data/` (patients,
appointments, insurance, providers, slots), fronted by `store.py` — **is**
the hospital's system of record for this project, not sample data to
demo against. `store.py` is read-through, never cached: every function
re-reads its CSV from disk on every call, so it always reflects whatever
was most recently written, no matter which agent (or which of its own
functions) wrote it. `book_slot()`/`cancel_appointment()` persist
immediately; `cancel_appointment()` flips a status field rather than
deleting the row, so the visit stays in the record.

Every agent the self-extension system generates is grounded in this store
via a permanent, always-included section of `agent_maker/generator.py`'s
system prompt (`_ehr_store_section()`, introspected from the real
`store.py` functions and real CSV headers/sample rows at generation time,
so it can't drift) — the same never-invent-fake-data principle the
credentialed-tools grounding already applies to real MCP services. If none
of `store.py`'s existing functions fit a new capability, a generated agent
may implement its own new Create/Read/Update function (never Delete — retire
a record by flipping a status field, same as `cancel_appointment`) as a
plain module inside its own directory, operating on the same CSV files
using the exact read-fresh/write-back pattern `store.py` itself uses. It may
never edit `store.py` directly — additive-only generation means a generated
agent can only create files inside its own directory.

Two things had to be fixed for any of this to actually work, not just read
correctly in the prompt:

- The generation sandbox (`agent_maker/sandbox_runner.py`) had no
  `PYTHONPATH` back to the repo root, so `import tools.shared...` failed
  with `ModuleNotFoundError` — confirmed live, and exactly why earlier
  generated agents invented self-contained fake data instead: it was the
  only thing that could pass their own sandbox tests.
- The sandbox denies writes outside the new agent's own directory (by
  design), which would otherwise make `book_slot()`/`cancel_appointment()`
  fail every sandboxed test that actually exercises them. Rather than
  weakening that containment, `sandbox_runner.py` now copies the real EHR
  CSVs into a fresh throwaway temp directory before every run and points
  `store.py` at it via the `EHR_CSV_DATA_DIR` env var (unset everywhere
  else) — a generated agent's tests can book/cancel for real, against
  realistic data, without ever touching (or permanently corrupting, on
  every sandbox run and later regression re-run) the real shared CSVs.

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
uv run python -m orchestrator.chat --user <your_id> --furhat <furhat_ip>
```

REPL commands: `:prompt <agent_id> <text>` (live prompt edit, no restart),
`:rollback <agent_id> <n>`, `:scope <text>` (live domain scope edit, no
restart), `:quit`.

## FurHat mode (optional)

`--furhat <ip>` (or `FURHAT_IP` in `.env`) attempts to connect to a FurHat
robot over its Realtime WebSocket API (`orchestrator/furhat_io.py`) and, once
connected, speaks agent replies and listens for the patient's spoken replies
instead of `print()`/`input()`. It is **additive and always falls back to
plain CLI text**: no FurHat configured, unreachable at startup, or a
connection drop mid-session all degrade to exactly today's CLI loop rather
than crashing the session. Typed `:` operator commands (`:quit`, `:prompt`,
`:scope`, `:rollback`, `:bad`) work in FurHat mode too — a silent/timed-out
listen falls through to the terminal for that turn.

**Not verified against real hardware** — no FurHat or Virtual FurHat was
available during development, so this ships to the same honesty standard the
credential framework already applies to its untested `oauth2` type (see
below): built against the documented protocol and offline-tested against a
faked WebSocket transport (`tests/test_furhat_client.py`), but the connected
path itself hasn't been exercised against a real device. The CLI fallback is
the fully verified default.

## Tests

```sh
uv run pytest                                   # repo-level unit tests
uv run pytest agents/_shared/general_qa         # agent suites run per-directory
uv run pytest agents/_shared/summarizer         # (same way the sandbox runs them)
uv run pytest agents/_shared/faq
uv run pytest agents/_shared/checkin
```

## Layout

```
agents/_shared/<id>/        hand-written agents: agent.py, manifest.json, test_agent.py
agents/<user_id>/<id>/      generated per-user agents (same contract)
tools/shared/, tools/<user_id>/   tools (e.g. ehr_csv, gmail_imap, mock_email_mcp)
orchestrator/               state (the locked schema), supervisor, scope_gate, classifier,
                             loader, tracing, furhat_io (optional voice I/O)
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

Note: both services below are email-related, and email is outside the
default front-desk domain scope (see "Domain scope gate" above). `mock_email`
is still `planner_eligible` in the catalog itself, but `gmail_recent_reader`
— the one hand-written agent that actually uses it — has been purged from
the registry, so a live REPL session declines "check my email" after a
genuine registry miss rather than routing to it. The framework and
`gmail_recent_reader`'s files stay fully wired and offline-tested
(`tests/test_credential_gate.py` stubs both the classifier and the scope
gate directly so it still exercises this path); `registry.backfill()` would
make it reachable again if you ever want it back in this domain.

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

Everything mechanical is covered by the offline suite. A few checks exercise
the real models:

```sh
uv run python -m eval.tune_threshold --live   # Phase 2 exit: end-to-end routing accuracy
uv run python -m orchestrator.chat --user me
# "what are your office hours" -> routes to faq, answers from the ported
#   knowledge base.
# "hi, I'm here for my appointment, Sarah Johnson, DOB 1985-03-12" -> routes
#   to checkin, walks the full flow (identity, appointment, consent, forms)
#   to completion across several turns without losing the thread — this
#   proves the awaiting_reply stickiness fix works live, not just offline.
# "check my email for anything urgent" and "convert this to pig latin" ->
#   both miss the registry (their old agents were purged, see "Domain scope
#   gate" above) and then get declined by the scope gate before generation
#   is attempted. This IS the working behavior, not a bug.
# "can I schedule an appointment" -> genuinely in-domain (scheduling is a
#   declared purpose) but not covered by faq or checkin: exercises Phase 3
#   self-extension end to end (plans, generates, sandbox-tests, registers,
#   answers). Check `git status`
#   (only agents/me/... created) and re-ask to see it route directly.
# :scope <new text> -> edit the domain boundary live, no restart.
```

## Production notes

The generation sandbox uses macOS `sandbox-exec` locally (network denied,
env scrubbed, writes confined to the new agent's directory, plus a
throwaway per-run copy of the EHR CSVs — see "The local hospital DB" above).
In production, swap `agent_maker/sandbox_runner.py` for Docker or E2B — the
interface is a single `run_tests(agent_dir)` function; keep the same
temp-copy-and-env-var pattern for `EHR_CSV_DATA_DIR` regardless of sandbox
backend, since that isolation is about protecting the shared CSV "DB" from
test pollution, not about the sandbox mechanism itself.

`tools/shared/ehr_csv/store.py`'s CSVs stand in for a real hospital database
for this project (see "The local hospital DB" above) — before this leaves a
demo, swap the plain-file read/write in `store.py` for a real DB client,
keeping the same function signatures so nothing else in the system (the
generator's grounding, `checkin`, or any generated agent) needs to change.
