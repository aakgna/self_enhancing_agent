"""render_text: display-layer normalization for AIMessage.content, which is
normally a plain string but becomes a list of content blocks (thinking +
text) whenever agent_llm()'s adaptive thinking decides to think for a given
call. Regression test for a live bug: chat.py printed/spoke the raw content
list (including the base64 thinking signature) whenever this triggered.

Also: FallbackChatModel / _StructuredFallback, the ordered Bedrock ->
Anthropic -> OpenAI provider chain. Chain-walking behavior is tested offline
against directly-constructed fake provider tuples (label, builder,
error_types, is_configured) rather than _agent_providers()/_fast_providers()
themselves, so these tests never touch a real API and don't care what the
real provider list looks like. A separate, small set of tests at the bottom
checks that _agent_providers()/_fast_providers() themselves produce the
right order and read the right env vars — real .env values (this repo
requires ANTHROPIC_API_KEY) would otherwise leak into any test that relies
on a provider being "unconfigured", since load_dotenv() runs at import time,
so every such test explicitly delenvs what it needs to."""

import anthropic
import openai
import pytest
from botocore.exceptions import ClientError

import orchestrator.llm as llm_module
from orchestrator.llm import FallbackChatModel, render_text


def _clear_fallback_env(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)


def test_plain_string_passes_through():
    assert render_text("hello") == "hello"


def test_extracts_text_block_and_drops_thinking_block():
    content = [
        {"type": "thinking", "thinking": "internal reasoning here", "signature": "abc123"},
        {"type": "text", "text": "Welcome! Could you share your name and date of birth?"},
    ]
    assert render_text(content) == "Welcome! Could you share your name and date of birth?"


def test_joins_multiple_text_blocks():
    content = [{"type": "text", "text": "Hello, "}, {"type": "text", "text": "world."}]
    assert render_text(content) == "Hello, world."


def test_no_text_block_returns_empty_string():
    content = [{"type": "thinking", "thinking": "...", "signature": "abc"}]
    assert render_text(content) == ""


def test_non_string_non_list_falls_back_to_str():
    assert render_text(42) == "42"


class _RaisingModel:
    def __init__(self, exc):
        self._exc = exc

    def invoke(self, *a, **k):
        raise self._exc

    def with_structured_output(self, schema, **kwargs):
        return self  # same raising behavior for the structured path


class _FakeModel:
    """Records calls; returns a fixed value."""

    def __init__(self, value="reply"):
        self.value = value
        self.invoke_calls = 0
        self.structured_schema = None
        self.structured_kwargs = None

    def invoke(self, *a, **k):
        self.invoke_calls += 1
        return self.value

    def with_structured_output(self, schema, **kwargs):
        self.structured_schema = schema
        self.structured_kwargs = kwargs
        return self


def _provider(label, model, error_types=(Exception,), configured=True):
    """A directly-constructed provider tuple, bypassing
    _agent_providers()/_fast_providers() entirely so chain-walking behavior
    can be tested without touching real env vars or provider libraries."""
    return (label, lambda max_tokens: model, error_types, lambda: configured)


# ---- FallbackChatModel.invoke(): chain walking ----


def test_first_configured_provider_is_used_on_success():
    first = _FakeModel("bedrock reply")
    model = FallbackChatModel([_provider("Bedrock", first)], max_tokens=1024)
    assert model.invoke("hi") == "bedrock reply"
    assert first.invoke_calls == 1


def test_unconfigured_provider_is_skipped():
    skipped = _FakeModel("skipped")
    used = _FakeModel("used")
    providers = [
        _provider("Bedrock", skipped, configured=False),
        _provider("Anthropic", used),
    ]
    model = FallbackChatModel(providers, max_tokens=1024)
    assert model.invoke("hi") == "used"
    assert skipped.invoke_calls == 0
    assert used.invoke_calls == 1


def test_falls_through_on_that_providers_own_error():
    """Regression shape for the live case: Bedrock (first, preferred) fails
    with its own error type, so the chain moves on to Anthropic."""
    failing_bedrock = _RaisingModel(ClientError({"Error": {"Code": "ThrottlingException"}}, "Converse"))
    rescuing_anthropic = _FakeModel("rescued")
    providers = [
        _provider("Bedrock", failing_bedrock, error_types=(ClientError,)),
        _provider(llm_module._ANTHROPIC, rescuing_anthropic),
    ]
    model = FallbackChatModel(providers, max_tokens=1024)
    assert model.invoke("hi") == "rescued"


def test_non_matching_exception_is_never_caught():
    failing = _RaisingModel(ValueError("a real bug, not a provider failure"))
    rescuing = _FakeModel("rescued")
    providers = [
        _provider(llm_module._ANTHROPIC, failing, error_types=(anthropic.AnthropicError,)),
        _provider("OpenAI", rescuing),
    ]
    model = FallbackChatModel(providers, max_tokens=1024)
    with pytest.raises(ValueError):
        model.invoke("hi")
    assert rescuing.invoke_calls == 0


def test_exhausted_chain_reraises_the_most_recent_real_failure():
    first_exc = ClientError({"Error": {"Code": "ThrottlingException"}}, "Converse")
    second_exc = openai.OpenAIError("second failure")
    providers = [
        _provider("Bedrock", _RaisingModel(first_exc), error_types=(ClientError,)),
        _provider("OpenAI", _RaisingModel(second_exc), error_types=(openai.OpenAIError,)),
    ]
    model = FallbackChatModel(providers, max_tokens=1024)
    with pytest.raises(openai.OpenAIError) as excinfo:
        model.invoke("hi")
    assert excinfo.value is second_exc


def test_nothing_configured_raises_a_clear_runtime_error():
    providers = [_provider("Bedrock", _FakeModel(), configured=False)]
    model = FallbackChatModel(providers, max_tokens=1024)
    with pytest.raises(RuntimeError, match="No LLM provider is configured"):
        model.invoke("hi")


# ---- with_structured_output(): only the Anthropic leg gets the caller's method ----


def test_structured_output_uses_callers_method_on_anthropic():
    anthropic_model = _FakeModel("anthropic structured")
    model = FallbackChatModel([_provider(llm_module._ANTHROPIC, anthropic_model)], max_tokens=8192)

    class Schema:
        pass

    result = model.with_structured_output(Schema, method="json_schema").invoke("extract this")

    assert result == "anthropic structured"
    assert anthropic_model.structured_kwargs == {"method": "json_schema"}


def test_structured_output_forces_function_calling_on_bedrock_and_openai():
    """Regression test: both Bedrock's and OpenAI's own native/default
    structured-output modes enforce `additionalProperties: false`
    recursively, which silently empties AgentSpec's freeform
    input_schema/output_schema: dict fields (Bedrock, confirmed by reading
    langchain_aws's _set_additional_properties_false) or crashes outright
    (OpenAI, missing a required key). Whatever method the caller asked for
    must never reach them, regardless of chain position.
    """

    class Schema:
        pass

    bedrock_model = _FakeModel("bedrock structured")
    FallbackChatModel([_provider("Bedrock", bedrock_model)], max_tokens=8192).with_structured_output(
        Schema, method="json_schema"
    ).invoke("extract this")
    assert bedrock_model.structured_kwargs == {"method": "function_calling"}

    openai_model = _FakeModel("openai structured")
    FallbackChatModel([_provider("OpenAI", openai_model)], max_tokens=8192).with_structured_output(
        Schema, method="json_schema"
    ).invoke("extract this")
    assert openai_model.structured_kwargs == {"method": "function_calling"}


def test_structured_output_falls_through_the_chain():
    failing_bedrock = _RaisingModel(ClientError({"Error": {"Code": "ThrottlingException"}}, "Converse"))
    rescuing_anthropic = _FakeModel("rescued")
    providers = [
        _provider("Bedrock", failing_bedrock, error_types=(ClientError,)),
        _provider(llm_module._ANTHROPIC, rescuing_anthropic),
    ]
    model = FallbackChatModel(providers, max_tokens=8192)

    class Schema:
        pass

    result = model.with_structured_output(Schema, method="json_schema").invoke("extract this")

    assert result == "rescued"
    # Even though the caller asked for json_schema, Anthropic wasn't first
    # in *chain position* here — but it's still the Anthropic *provider*, so
    # it still gets the caller's method (the rule is keyed on provider
    # identity, not position).
    assert rescuing_anthropic.structured_kwargs == {"method": "json_schema"}


# ---- _agent_providers()/_fast_providers(): real order and configured-checks ----


def test_agent_providers_order_is_bedrock_anthropic_openai(monkeypatch):
    _clear_fallback_env(monkeypatch)
    labels = [label for label, *_ in llm_module._agent_providers()]
    assert labels == ["Bedrock", llm_module._ANTHROPIC, "OpenAI"]


def test_fast_providers_order_is_bedrock_anthropic_openai(monkeypatch):
    _clear_fallback_env(monkeypatch)
    labels = [label for label, *_ in llm_module._fast_providers()]
    assert labels == ["Bedrock", llm_module._ANTHROPIC, "OpenAI"]


def test_providers_configured_checks_read_the_right_env_vars(monkeypatch):
    _clear_fallback_env(monkeypatch)
    configured = {label: is_configured for label, _, _, is_configured in llm_module._agent_providers()}

    assert configured["Bedrock"]() is False
    assert configured[llm_module._ANTHROPIC]() is False
    assert configured["OpenAI"]() is False

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-test")
    monkeypatch.setenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    assert configured["Bedrock"]() is True
    assert configured[llm_module._ANTHROPIC]() is True
    assert configured["OpenAI"]() is True


def test_bedrock_configured_requires_both_credentials_and_model_id(monkeypatch):
    _clear_fallback_env(monkeypatch)
    assert llm_module.bedrock_configured() is False

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-test")
    assert llm_module.bedrock_configured() is False  # no BEDROCK_MODEL_ID yet

    monkeypatch.setenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
    assert llm_module.bedrock_configured() is True
