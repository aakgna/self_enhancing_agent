from orchestrator import agent_loader


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
