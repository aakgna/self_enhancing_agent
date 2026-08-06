"""Dynamic agent discovery and loading.

Agents live at agents/_shared/<agent_id>/ (hand-written, all users) and
agents/<user_id>/<agent_id>/ (generated, per-user). Each directory contains
agent.py + manifest.json (+ test_agent.py).

Loading is importlib-based and uncached-on-miss, so an agent registered by the
Agent Maker mid-session is immediately callable without a restart. A per-user
agent with the same id shadows a shared one.
"""

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

AGENTS_ROOT = Path(__file__).resolve().parent.parent / "agents"
SHARED_DIR = "_shared"


@dataclass
class AgentHandle:
    agent_id: str
    directory: Path
    manifest: dict

    @property
    def manifest_version(self) -> int:
        return int(self.manifest.get("version", 0))


def _scan(directory: Path) -> dict[str, AgentHandle]:
    handles: dict[str, AgentHandle] = {}
    if not directory.is_dir():
        return handles
    for sub in sorted(directory.iterdir()):
        if sub.name.startswith("."):  # e.g. .failed quarantine — never loadable
            continue
        agent_py = sub / "agent.py"
        manifest_json = sub / "manifest.json"
        if agent_py.is_file() and manifest_json.is_file():
            manifest = json.loads(manifest_json.read_text())
            handles[sub.name] = AgentHandle(sub.name, sub, manifest)
    return handles


def discover(user_id: str) -> dict[str, AgentHandle]:
    """All agents callable by this user; per-user shadows shared."""
    handles = _scan(AGENTS_ROOT / SHARED_DIR)
    handles.update(_scan(AGENTS_ROOT / user_id))
    return handles


def load_module(handle: AgentHandle) -> ModuleType:
    module_name = f"agents.{handle.directory.parent.name}.{handle.agent_id}"
    spec = importlib.util.spec_from_file_location(module_name, handle.directory / "agent.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    # A generated agent may import a sibling helper module it wrote alongside
    # agent.py (e.g. `from queue_status import ...`). That import succeeds
    # during the sandboxed test run because pytest puts the test file's own
    # directory on sys.path, but crashed here at first real production load
    # with a bare ModuleNotFoundError — this mirrors that same behavior by
    # adding the agent's own directory for the duration of the load.
    # Narrow known gap: two different agents both writing a same-named helper
    # (e.g. both writing "utils.py") would collide via the shared sys.modules
    # cache — not addressed here.
    agent_dir = str(handle.directory)
    sys.path.insert(0, agent_dir)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(agent_dir)
    return module
