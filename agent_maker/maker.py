"""The Agent Maker pipeline: plan -> generate -> sandbox-test -> register.

Self-correction: sandbox failures are fed back to the same generation session
(context preserved), capped at MAX_RETRIES, then surfaced as a failure rather
than looping forever. Failed generations are quarantined into a dot-directory
the loader can never discover, and logged to generation_failures.

Provider order mirrors orchestrator/llm.py: Amazon Bedrock first when
configured, direct Anthropic API otherwise/as fallback — no OpenAI leg here,
since the coding harness (Claude Agent SDK) only talks to Claude models, not
a LangChain chat model. If the *harness's own* model call fails upstream
(exhausted credits, an outage, a rate limit — see
generator.py::_is_upstream_api_failure) rather than a genuine code/test bug,
the whole attempt is retried once through the other configured leg instead
of burning the sandbox-retry budget on a request that could never have
produced any files in the first place. This is a different mechanism from
orchestrator/llm.py's fallback chain (this harness runs the Claude Agent SDK
as its own subprocess, not a LangChain chat model), so it needs its own,
separate ordering.
"""

import asyncio
import shutil
import time
from pathlib import Path

from agent_maker import contract_check, planner, registrar, sandbox_runner
from agent_maker.generator import GenerationSession
from orchestrator.agent_loader import AGENTS_ROOT
from orchestrator.llm import bedrock_configured
from orchestrator.registry import SearchHit

MAX_RETRIES = 2


def make_agent(
    user_id: str,
    request: str,
    candidates: list[SearchHit],
    progress=print,
) -> str | None:
    """Returns the new agent_id on success, None on failure."""
    progress("[agent_maker] planning a new agent for this request...")
    spec = planner.plan(request, candidates, user_id)
    progress(f"[agent_maker] spec: {spec.proposed_agent_name} — {spec.purpose[:90]}")

    agent_dir = AGENTS_ROOT / user_id / spec.proposed_agent_name
    agent_dir.mkdir(parents=True, exist_ok=False)

    try:
        passed, output = asyncio.run(
            _generate_with_retries(spec, agent_dir, request, progress)
        )
    except Exception as exc:  # harness/connection failures are generation failures
        registrar.log_failure(user_id, spec, request, f"harness error: {exc}", None)
        _quarantine(agent_dir)
        progress(f"[agent_maker] generation failed: {exc}")
        return None

    if not passed:
        registrar.log_failure(user_id, spec, request, "tests failed after retries", output)
        _quarantine(agent_dir)
        progress("[agent_maker] tests never passed — attempt logged, nothing registered.")
        return None

    try:
        manifest = registrar.register(spec, agent_dir, user_id, request)
    except Exception as exc:
        registrar.log_failure(user_id, spec, request, f"registration failed: {exc}", output)
        _quarantine(agent_dir)
        progress(f"[agent_maker] registration failed: {exc}")
        return None

    progress(f"[agent_maker] registered new agent: {manifest.id}")

    # Phase 5: async, silent, non-blocking — the agent is already live; this
    # never delays registration or availability.
    from eval import regression_runner

    regression_runner.on_agent_registered(user_id, manifest.id)
    return manifest.id


async def _generate_with_retries(
    spec, agent_dir: Path, request: str, progress
) -> tuple[bool, str]:
    prefer_bedrock = bedrock_configured()
    passed, output, upstream_failed = await _attempt_generation(
        spec, agent_dir, request, progress, use_bedrock=prefer_bedrock
    )
    if not passed and upstream_failed and prefer_bedrock:
        # The first attempt was Bedrock (preferred whenever configured) and
        # its own model call failed upstream — fall back to the direct
        # Anthropic API, the only other leg this harness can run on. If
        # Bedrock wasn't configured to begin with, the first attempt was
        # already Anthropic and there's nothing left to retry through.
        progress(
            "[agent_maker] Bedrock's own model call failed upstream before any "
            "code was generated — retrying via the direct Anthropic API..."
        )
        _clear_dir(agent_dir)
        passed, output, _ = await _attempt_generation(
            spec, agent_dir, request, progress, use_bedrock=False
        )
    return passed, output


async def _attempt_generation(
    spec, agent_dir: Path, request: str, progress, use_bedrock: bool
) -> tuple[bool, str, bool]:
    """Returns (passed, output, upstream_api_failed)."""
    check_contract = contract_check.applies(spec.mcp_service)
    provider = "Amazon Bedrock" if use_bedrock else "the direct Anthropic API"

    async with GenerationSession(spec, agent_dir, request, progress, use_bedrock=use_bedrock) as session:
        progress(f"[agent_maker] generating via {provider} (tests first, then implementation)...")
        await session.generate()
        output = ""
        for attempt in range(MAX_RETRIES + 1):
            if session.upstream_api_failed():
                return False, "the coding harness's model call failed upstream", True

            progress(f"[agent_maker] sandbox test run {attempt + 1}/{MAX_RETRIES + 1}...")
            passed, output = sandbox_runner.run_tests(agent_dir)

            if passed and check_contract:
                progress("[agent_maker] tests passed — checking the tools contract...")
                passed, contract_output = contract_check.run_contract_check(
                    agent_dir, spec.mcp_service, request
                )
                if not passed:
                    output = (
                        "The generated tests passed, but run() does not match the "
                        "real tools contract:\n\n" + contract_output
                    )

            if passed:
                progress("[agent_maker] tests passed in sandbox.")
                return True, output, False
            if attempt < MAX_RETRIES:
                progress("[agent_maker] checks failed — feeding output back for a fix...")
                await session.refine(output)
        return False, output, False


def _clear_dir(agent_dir: Path) -> None:
    """Wipe any files a failed attempt left behind before retrying through a
    different provider, so a partial write from the failed attempt can't
    masquerade as (or corrupt) the retry's output."""
    for child in agent_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _quarantine(agent_dir: Path) -> None:
    """Move a failed generation where the loader can never discover it."""
    if not agent_dir.exists():
        return
    graveyard = agent_dir.parent / ".failed"
    graveyard.mkdir(exist_ok=True)
    shutil.move(str(agent_dir), graveyard / f"{agent_dir.name}-{int(time.time())}")
