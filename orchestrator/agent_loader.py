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
    spec.loader.exec_module(module)
    return module
