"""Planner: turns a capability gap into a structured agent spec.

Tool dedup happens HERE, before generation is ever triggered: every proposed
new tool is checked against the registry, and anything close enough to an
existing tool is swapped in instead. This is the defense against near-identical
tools piling up per user.
"""

import json
import re

from pydantic import BaseModel, Field, field_validator

from orchestrator import registry
from orchestrator.llm import agent_llm
from orchestrator.registry import SearchHit
from orchestrator.tracing import write_trace

# High bar on purpose: false merges are worse than an occasional duplicate,
# and short-text similarity is noisy below ~0.8 (see classifier tuning notes).
TOOL_DEDUP_THRESHOLD = 0.80

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,40}$")


class NewToolSpec(BaseModel):
    name: str
    purpose: str


class AgentSpec(BaseModel):
    proposed_agent_name: str = Field(
        description="snake_case identifier, 3-40 chars, e.g. pig_latin_converter"
    )
    purpose: str = Field(description="one paragraph: what the agent does and for whom")
    input_schema: dict = Field(default_factory=dict)
    output_schema: dict = Field(default_factory=dict)
    existing_tools_needed: list[str] = Field(default_factory=list)
    new_tools_needed: list[NewToolSpec] = Field(default_factory=list)
    requires_credentials: bool = False
    mcp_service: str | None = Field(
        default=None,
        description="If credentials are required: the matching service name from the catalog, else null",
    )
    example_requests: list[str] = Field(
        default_factory=list,
        description="5-8 diverse user phrasings this agent should handle",
    )

    @field_validator("input_schema", "output_schema", mode="before")
    @classmethod
    def _coerce_json_string_schemas(cls, value):
        # Defense in depth: structured-output tool calling has, in practice,
        # occasionally serialized a nested dict field as a JSON string
        # instead of a proper object (seen live with forced tool calling +
        # thinking — see plan()'s method="json_schema" for the actual fix).
        # Tolerate it here rather than crash on a model formatting quirk.
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {}
        return value


_PROMPT = """\
A user request had no matching agent in a per-user multi-agent system. Design \
a spec for a new, narrowly scoped agent to fill exactly this gap.

User request: {request}

Agents that exist but were judged not to cover it:
{rejected}

Rules:
- Scope the agent to the *capability* behind the request (e.g. "convert text \
to pig latin"), not to this single utterance — but no broader than that.
- Deterministic transformations should be described as plain logic the agent \
implements in code; only genuinely language-heavy work should lean on an LLM.
- requires_credentials is true only if the capability needs an external \
authenticated service (email, calendar, paid APIs). Local computation never \
requires credentials.
- Credentialed services available for binding (set mcp_service to the name if \
one fits, else leave it null):
{services}
- Propose new tools only for reusable, self-contained pieces of logic; most \
small agents need none.
- example_requests must be diverse phrasings a real user would type."""


def _sanitize_name(name: str, user_id: str) -> str:
    candidate = re.sub(r"[^a-z0-9_]", "_", name.lower()).strip("_")[:40]
    if not _NAME_RE.match(candidate or ""):
        candidate = "generated_agent"
    from orchestrator.agent_loader import discover

    existing = set(discover(user_id))
    base, n = candidate, 2
    while candidate in existing:
        candidate = f"{base}_{n}"
        n += 1
    return candidate


def dedup_tools(spec: AgentSpec, user_id: str) -> AgentSpec:
    """Swap proposed new tools for close-enough existing ones; log either way."""
    kept: list[NewToolSpec] = []
    for tool in spec.new_tools_needed:
        hits = registry.search(f"{tool.name}: {tool.purpose}", user_id, type_filter="tool")
        best = hits[0] if hits else None
        if best and best.similarity >= TOOL_DEDUP_THRESHOLD:
            spec.existing_tools_needed.append(best.manifest.id)
            decision = {"tool": tool.name, "reused": best.manifest.id,
                        "similarity": round(best.similarity, 4)}
        else:
            kept.append(tool)
            decision = {
                "tool": tool.name,
                "reused": None,
                "best_similarity": round(best.similarity, 4) if best else None,
            }
        write_trace(user_id=user_id, node="planner", event="tool_dedup", payload=decision)
    spec.new_tools_needed = kept
    return spec


def plan(request: str, rejected: list[SearchHit], user_id: str) -> AgentSpec:
    from orchestrator.mcp_bindings import planner_catalog

    # Only services marked planner_eligible are offered here — e.g. gmail_imap
    # is fully wired but deliberately excluded so "check my email" doesn't
    # silently start asking for a real Gmail app password. See
    # mcp_bindings.py's module docstring.
    catalog = planner_catalog()
    rejected_text = "\n".join(
        f"- {h.manifest.id}: {h.manifest.description}" for h in rejected
    ) or "- (none were close)"
    services_text = "\n".join(
        f"  - {name}: {entry['scope']}" for name, entry in catalog.items()
    ) or "  - (none available)"
    spec = (
        agent_llm()
        # method="json_schema" uses native structured output (output_config.format)
        # instead of the default forced-tool-calling emulation, which LangChain
        # itself warns is not guaranteed to hold together with thinking enabled —
        # confirmed live: forced tool calling returned output_schema as a JSON
        # string instead of a nested object, crashing pydantic validation.
        .with_structured_output(AgentSpec, method="json_schema")
        .invoke(_PROMPT.format(request=request, rejected=rejected_text, services=services_text))
    )
    if spec.mcp_service not in catalog:
        spec.mcp_service = None
    spec.proposed_agent_name = _sanitize_name(spec.proposed_agent_name, user_id)
    if request not in spec.example_requests:
        spec.example_requests.insert(0, request)
    spec = dedup_tools(spec, user_id)
    write_trace(
        user_id=user_id,
        node="planner",
        event="spec_produced",
        agent_id=spec.proposed_agent_name,
        payload=spec.model_dump(mode="json"),
    )
    return spec
