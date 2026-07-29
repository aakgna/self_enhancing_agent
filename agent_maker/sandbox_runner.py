"""Run a generated agent's tests inside a macOS sandbox.

Guarantees, regardless of what the generated code tries:
  - no network access
  - writes confined to the agent's own directory (+ OS temp)
  - fully scrubbed environment — no API keys, no credentials, no dotfiles

Production swap: replace run_tests() with a Docker/E2B implementation; the
interface is just (agent_dir) -> (passed, output).
"""

import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

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


def run_tests(agent_dir: Path) -> tuple[bool, str]:
    agent_dir = agent_dir.resolve()
    with tempfile.NamedTemporaryFile("w", suffix=".sb", delete=False) as profile:
        profile.write(_PROFILE_TEMPLATE.format(agent_dir=agent_dir))
        profile_path = profile.name

    try:
        proc = subprocess.run(
            [
                "sandbox-exec",
                "-f",
                profile_path,
                str(PYTHON),
                "-m",
                "pytest",
                "test_agent.py",
                "-q",
                "--no-header",
                "-p",
                "no:cacheprovider",
            ],
            cwd=agent_dir,
            env={
                # Deliberately minimal: no ANTHROPIC_API_KEY, no user env.
                "PATH": "/usr/bin:/bin",
                "HOME": str(agent_dir),
                "TMPDIR": tempfile.gettempdir(),
            },
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return False, f"tests timed out after {TIMEOUT_SECONDS}s"
    finally:
        Path(profile_path).unlink(missing_ok=True)

    output = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, output[-4000:]
