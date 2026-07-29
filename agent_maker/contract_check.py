"""System-authored contract check for credentialed agents.

The generator's own test_agent.py is self-authored: if it misunderstands the
`tools` interface, its tests can pass while being self-consistently wrong —
this happened in practice (a generated agent called `tools.some_method(...)`
against a `list`, and its own fake used the same made-up shape). This check
is not written by the generator; it builds fake tools from the same catalog
the generator was given (orchestrator.mcp_bindings.KNOWN_SERVICES) and calls
the real run() against them, so a contract mismatch fails registration
instead of surfacing mid-conversation after a user approves a real
credential.

Runs in the same sandbox as the generator's own tests: no network, no
credentials, writes confined to the agent directory.
"""

from pathlib import Path

from agent_maker import sandbox_runner
from orchestrator.mcp_bindings import KNOWN_SERVICES

_CHECK_SCRIPT = """\
import sys, os
sys.path.insert(0, os.getcwd())

from langchain_core.messages import AIMessage, HumanMessage
import agent as agent_module

class FakeLLM:
    def invoke(self, messages):
        return AIMessage(content="stub")

class FakeTool:
    def __init__(self, name):
        self.name = name

    def invoke(self, args):
        return "fake result"

def make_state():
    return {{
        "messages": [HumanMessage(content={trigger!r})],
        "user_id": "contract_check",
        "active_agent": agent_module.AGENT_ID,
        "agent_scratchpad": {{}},
        "routing_history": [],
    }}

update_none = agent_module.run(make_state(), agent_module.DEFAULT_PROMPT, FakeLLM(), tools=None)
assert isinstance(update_none, dict), f"run(tools=None) must return a dict, got {{type(update_none)}}"

fake_tools = [FakeTool(n) for n in {tool_names!r}]
update_bound = agent_module.run(make_state(), agent_module.DEFAULT_PROMPT, FakeLLM(), tools=fake_tools)
assert isinstance(update_bound, dict), f"run(tools=<bound>) must return a dict, got {{type(update_bound)}}"

print("CONTRACT_CHECK_OK")
"""


def applies(mcp_service: str | None) -> bool:
    return mcp_service in KNOWN_SERVICES


def run_contract_check(agent_dir: Path, mcp_service: str, triggering_request: str) -> tuple[bool, str]:
    tool_names = [t["name"] for t in KNOWN_SERVICES[mcp_service]["tools"]]
    script = _CHECK_SCRIPT.format(tool_names=tool_names, trigger=triggering_request)
    passed, output = sandbox_runner.run_script(agent_dir, script)
    return (passed and "CONTRACT_CHECK_OK" in output), output
