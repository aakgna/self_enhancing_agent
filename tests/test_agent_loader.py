import shutil
import uuid

from orchestrator import agent_loader
from orchestrator.agent_loader import AGENTS_ROOT


def test_discovers_shared_agents():
    handles = agent_loader.discover("any_user")
    assert {"general_qa", "summarizer"} <= set(handles)
    assert handles["general_qa"].manifest_version == 1


def test_load_module_exposes_contract():
    handles = agent_loader.discover("any_user")
    module = agent_loader.load_module(handles["summarizer"])
    assert module.AGENT_ID == "summarizer"
    assert isinstance(module.AGENT_VERSION, int)
    assert callable(module.run)
    assert isinstance(module.DEFAULT_PROMPT, str)


def test_load_module_resolves_a_sibling_helper_module():
    """Regression test for a live crash: a generated agent that imports a
    helper module it wrote alongside agent.py (e.g. `from queue_status
    import ...`) passes its sandboxed test run — pytest puts the test file's
    own directory on sys.path — but crashed with a bare ModuleNotFoundError
    at real production load, because load_module never did the same. It must
    now resolve the sibling import instead of crashing."""
    user_id = f"_loader_sibling_{uuid.uuid4().hex[:8]}"
    agent_dir = AGENTS_ROOT / user_id / "with_helper"
    agent_dir.mkdir(parents=True)
    (agent_dir / "helper_module.py").write_text("VALUE = 42\n")
    (agent_dir / "agent.py").write_text(
        "from helper_module import VALUE\n"
        "AGENT_ID = 'with_helper'\n"
        "AGENT_VERSION = 1\n"
        "DEFAULT_PROMPT = 'x'\n"
        "def run(state, system_prompt, llm):\n"
        "    return {'messages': [], 'agent_scratchpad': {}}\n"
    )
    (agent_dir / "manifest.json").write_text(
        '{"id": "with_helper", "user_id": "%s", "type": "agent", "name": "x", '
        '"description": "test fixture", "example_requests": ["hi"], "version": 1}' % user_id
    )
    try:
        handles = agent_loader.discover(user_id)
        module = agent_loader.load_module(handles["with_helper"])
        assert module.VALUE == 42
    finally:
        shutil.rmtree(AGENTS_ROOT / user_id)
