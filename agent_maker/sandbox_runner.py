"""Run a generated agent's tests (and system-authored checks) inside a macOS
sandbox.

Guarantees, regardless of what the generated code tries:
  - no network access
  - writes confined to the agent's own directory (+ OS temp)
  - fully scrubbed environment — no API keys, no credentials, no dotfiles

Production swap: replace _run() with a Docker/E2B implementation; the
interface is just (agent_dir, argv) -> (passed, output).
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
EHR_DATA_DIR = REPO_ROOT / "tools" / "shared" / "ehr_csv" / "data"

_PROFILE_TEMPLATE = """\
(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write*
  (subpath "{agent_dir}")
  (subpath "/private/var/folders")
  (subpath "/private/tmp")
  (literal "/dev/null")
  (literal "/dev/urandom")
  (literal "/dev/random"))
"""

TIMEOUT_SECONDS = 120


def _run(agent_dir: Path, argv: list[str]) -> tuple[bool, str]:
    agent_dir = agent_dir.resolve()
    with tempfile.NamedTemporaryFile("w", suffix=".sb", delete=False) as profile:
        profile.write(_PROFILE_TEMPLATE.format(agent_dir=agent_dir))
        profile_path = profile.name

    # A throwaway copy of the real EHR CSVs, so a generated agent's own
    # test_agent.py can call book_slot()/cancel_appointment() (or its own
    # new Create/Update function — see generator.py's EHR-store grounding)
    # for real, against realistic data, without permanently corrupting the
    # shared "hospital DB" on every sandbox run — and every later regression
    # re-run of that same test (see eval/regression_runner.py). Lives under
    # the OS tempdir, already covered by the sandbox's existing
    # /private/var/folders write-allow rule below, so no profile change
    # needed; store.py picks it up via the EHR_CSV_DATA_DIR env var.
    ehr_data_copy = Path(tempfile.mkdtemp(prefix="ehr_csv_sandbox_")).resolve()
    shutil.copytree(EHR_DATA_DIR, ehr_data_copy, dirs_exist_ok=True)

    try:
        proc = subprocess.run(
            ["sandbox-exec", "-f", profile_path, *argv],
            cwd=agent_dir,
            env={
                # Deliberately minimal: no ANTHROPIC_API_KEY, no user env.
                "PATH": "/usr/bin:/bin",
                "HOME": str(agent_dir),
                "TMPDIR": tempfile.gettempdir(),
                # Without this, `import tools.shared...` (e.g. the shared EHR
                # store) fails with ModuleNotFoundError, since cwd is the
                # agent's own directory, not the repo root — confirmed live:
                # this is exactly why earlier generated agents invented
                # self-contained fake data instead of using the real shared
                # store, since importing it made their own sandbox tests fail.
                # Read-only access to the rest of the repo was never denied by
                # the sandbox profile below (only writes/network are), so this
                # doesn't weaken containment — it only fixes import resolution.
                "PYTHONPATH": str(REPO_ROOT),
                "EHR_CSV_DATA_DIR": str(ehr_data_copy),
            },
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {TIMEOUT_SECONDS}s"
    finally:
        Path(profile_path).unlink(missing_ok=True)
        shutil.rmtree(ehr_data_copy, ignore_errors=True)

    output = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, output[-4000:]


def run_tests(agent_dir: Path) -> tuple[bool, str]:
    return _run(
        agent_dir,
        [str(PYTHON), "-m", "pytest", "test_agent.py", "-q", "--no-header", "-p", "no:cacheprovider"],
    )


def run_script(agent_dir: Path, script: str) -> tuple[bool, str]:
    """Run an arbitrary system-authored Python script with cwd=agent_dir,
    under the same containment as run_tests. Used for checks the generator
    doesn't author itself (see agent_maker/contract_check.py)."""
    return _run(agent_dir, [str(PYTHON), "-c", script])
