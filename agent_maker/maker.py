"""The Agent Maker pipeline: plan -> generate -> sandbox-test -> register.

Self-correction: sandbox failures are fed back to the same generation session
(context preserved), capped at MAX_RETRIES, then surfaced as a failure rather
than looping forever. Failed generations are quarantined into a dot-directory
the loader can never discover, and logged to generation_failures.
"""

import asyncio
import shutil
import time
from pathlib import Path

from agent_maker import contract_check, planner, registrar, sandbox_runner
from agent_maker.generator import GenerationSession
from orchestrator.agent_loader import AGENTS_ROOT
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
    check_contract = contract_check.applies(spec.mcp_service)

    async with GenerationSession(spec, agent_dir, request, progress) as session:
        progress("[agent_maker] generating (tests first, then implementation)...")
        await session.generate()
        output = ""
        for attempt in range(MAX_RETRIES + 1):
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
                return True, output
            if attempt < MAX_RETRIES:
                progress("[agent_maker] checks failed — feeding output back for a fix...")
                await session.refine(output)
        return False, output


def _quarantine(agent_dir: Path) -> None:
    """Move a failed generation where the loader can never discover it."""
    if not agent_dir.exists():
        return
    graveyard = agent_dir.parent / ".failed"
    graveyard.mkdir(exist_ok=True)
    shutil.move(str(agent_dir), graveyard / f"{agent_dir.name}-{int(time.time())}")
