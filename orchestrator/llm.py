"""LLM factories. Agents get Opus with adaptive thinking; routing/classification
gets Haiku (cheap, fast — it's a routing decision, not a reasoning task).

Every call walks an ordered provider chain: Amazon Bedrock, then the direct
Anthropic API, then OpenAI (gpt-4o-mini) — Bedrock first because it's the
provider actually reachable when the direct Anthropic API is degraded for
this account (exhausted credits, an outage, a rate limit), while still
serving the same Claude models. Each step only engages if that provider's
own credentials are configured (AWS_ACCESS_KEY_ID + BEDROCK_MODEL_ID,
ANTHROPIC_API_KEY, or OPENAI_API_KEY); a configured provider erroring on its
own turn moves on to the next configured one. If nothing is configured, or
every configured step fails, the most recent real failure is re-raised
as-is, so it stays legible instead of being replaced by a confusing "not
configured" error for someone who never asked for a fallback.
"""

import os
from functools import lru_cache

import anthropic
import openai
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

AGENT_MODEL = "claude-opus-4-8"
FAST_MODEL = "claude-haiku-4-5"
OPENAI_FALLBACK_MODEL = "gpt-4o-mini"

_ANTHROPIC = "Anthropic"


def _bedrock_region() -> str:
    return os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"


def _build_bedrock(max_tokens: int):
    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(model=os.environ["BEDROCK_MODEL_ID"], region_name=_bedrock_region(), max_tokens=max_tokens)


def _bedrock_configured() -> bool:
    return bool(os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("BEDROCK_MODEL_ID"))


def bedrock_configured() -> bool:
    """Public check reused by agent_maker/generator.py's own, separate
    Bedrock-first fallback for code generation (the Claude Agent SDK
    subprocess path, not this module's chain) — same "is Bedrock actually
    usable" question, kept in one place so the two callers can't drift
    apart."""
    return _bedrock_configured()


def _anthropic_configured() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _build_openai_fallback(max_tokens: int):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=OPENAI_FALLBACK_MODEL, max_tokens=max_tokens)


def _openai_configured() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _agent_providers():
    """Bedrock -> Anthropic -> OpenAI for agent_llm() (Opus tier). Built
    fresh on every call — rather than once as a module-level constant — so
    tests can monkeypatch the module-level builder/configured-check names
    and have this pick up the patched versions; a constant built at import
    time would freeze in the original function objects and silently ignore
    the patch. Adaptive thinking is an Anthropic-specific feature (Bedrock's
    Converse API support for it isn't verified here), so only the direct
    Anthropic leg requests it — the Bedrock leg still serves the same Opus
    model, just without that extra behavior."""
    return [
        ("Bedrock", _build_bedrock, (BotoCoreError, ClientError), _bedrock_configured),
        (
            _ANTHROPIC,
            lambda max_tokens: ChatAnthropic(model=AGENT_MODEL, max_tokens=max_tokens, thinking={"type": "adaptive"}),
            (anthropic.AnthropicError,),
            _anthropic_configured,
        ),
        ("OpenAI", _build_openai_fallback, (openai.OpenAIError,), _openai_configured),
    ]


def _fast_providers():
    """Bedrock -> Anthropic -> OpenAI for fast_llm() (Haiku tier)."""
    return [
        ("Bedrock", _build_bedrock, (BotoCoreError, ClientError), _bedrock_configured),
        (
            _ANTHROPIC,
            lambda max_tokens: ChatAnthropic(model=FAST_MODEL, max_tokens=max_tokens),
            (anthropic.AnthropicError,),
            _anthropic_configured,
        ),
        ("OpenAI", _build_openai_fallback, (openai.OpenAIError,), _openai_configured),
    ]


def _invoke_chain(providers, max_tokens: int, invoke_fn, *args, **kwargs):
    """invoke_fn(label, model, *args, **kwargs) -> result. Walks providers in
    order, skipping any unconfigured one and moving to the next on that
    provider's own error. Returns the first success; re-raises the most
    recent real failure (never a synthetic "not configured" error) if the
    chain is exhausted. Raises RuntimeError only in the unreachable-in-
    practice case where nothing at all is configured (this repo's setup
    requires ANTHROPIC_API_KEY), so there's never a real failure to
    re-raise."""
    last_exc: Exception | None = None
    for label, build, error_types, is_configured in providers:
        if not is_configured():
            continue
        try:
            model = build(max_tokens)
            return invoke_fn(label, model, *args, **kwargs)
        except error_types as exc:
            print(f"[llm] {type(exc).__name__} from {label} — trying the next configured provider: {exc}")
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(
        "No LLM provider is configured — set ANTHROPIC_API_KEY, OPENAI_API_KEY, "
        "or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY + BEDROCK_MODEL_ID."
    )


class _StructuredFallback:
    """The .with_structured_output(...) counterpart to FallbackChatModel.
    Provider-specific method kwargs apply only to the direct Anthropic leg
    (whatever the caller passed to with_structured_output, e.g.
    method="json_schema" — see planner.py's plan()); Bedrock and OpenAI
    always get method="function_calling" instead, regardless of chain
    position. Both of their default/native structured-output modes enforce
    `additionalProperties: false` recursively (langchain-openai's default,
    and confirmed by reading langchain_aws.ChatBedrockConverse's own
    _set_additional_properties_false), which silently collapses AgentSpec's
    intentionally freeform input_schema/output_schema: dict fields down to
    an empty object — a live crash for OpenAI (a missing key it requires),
    and silent data loss for Bedrock (no error at all, just always {}).
    function_calling tolerates permissive schemas and works fine for the
    simpler flat ones (ClassifierVerdict, ScopeVerdict, IdentityFields) too,
    so it's the safe default for every provider except the one this was
    actually verified against."""

    def __init__(self, providers, max_tokens: int, schema, anthropic_kwargs: dict):
        self._providers = providers
        self._max_tokens = max_tokens
        self._schema = schema
        self._anthropic_kwargs = anthropic_kwargs

    def invoke(self, *args, **kwargs):
        def _call(label, model, *a, **k):
            method_kwargs = self._anthropic_kwargs if label == _ANTHROPIC else {"method": "function_calling"}
            return model.with_structured_output(self._schema, **method_kwargs).invoke(*a, **k)

        return _invoke_chain(self._providers, self._max_tokens, _call, *args, **kwargs)


class FallbackChatModel:
    """Wraps an ordered provider chain behind the same invoke()/
    with_structured_output() surface every call site in this codebase
    already uses. Tries each configured provider in order (see
    _agent_providers/_fast_providers), moving on when that provider's own
    error type is raised. Non-provider exceptions (a bug in this code, a
    malformed message) are never caught here — only genuine provider-level
    failures fall through to the next step."""

    def __init__(self, providers, max_tokens: int):
        self._providers = providers
        self._max_tokens = max_tokens

    def invoke(self, *args, **kwargs):
        return _invoke_chain(
            self._providers, self._max_tokens, lambda label, model, *a, **k: model.invoke(*a, **k), *args, **kwargs
        )

    def with_structured_output(self, schema, **kwargs):
        return _StructuredFallback(self._providers, self._max_tokens, schema, kwargs)


@lru_cache(maxsize=1)
def agent_llm() -> FallbackChatModel:
    return FallbackChatModel(_agent_providers(), max_tokens=8192)


@lru_cache(maxsize=1)
def fast_llm() -> FallbackChatModel:
    return FallbackChatModel(_fast_providers(), max_tokens=1024)


def render_text(content: str | list) -> str:
    """AIMessage.content from agent_llm() is normally a plain string, but
    becomes a list of content blocks (a "thinking" block plus a "text"
    block) whenever adaptive thinking decides to actually think for a given
    call — that decision is per-request and content-dependent, so this must
    be handled at every display boundary rather than assumed away. Only text
    blocks are shown; thinking is internal reasoning, never meant for the
    user. Display-only: never mutates the stored message, since Anthropic's
    own guidance is to replay thinking blocks back verbatim in later turns,
    not strip them. (Bedrock/OpenAI fallback replies never produce
    list-shaped content — this only ever triggers on the direct Anthropic
    path, the only leg where adaptive thinking is requested.)"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)
