"""The shared manifest shape for agents and tools.

The description field is what gets embedded for retrieval — its quality
directly determines routing quality and dedup quality. Write it like
documentation (what it does, when to use it, when NOT to use it), never like
a label.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class Manifest(BaseModel):
    id: str
    user_id: str
    type: str = Field(pattern="^(agent|tool)$")
    name: str
    description: str
    # Representative user utterances this capability should win retrieval for.
    # Requests resemble other requests far more than they resemble capability
    # descriptions, so these dominate retrieval quality. For generated agents,
    # the request that triggered generation always goes in here.
    example_requests: list[str] = Field(default_factory=list)
    input_schema: dict = Field(default_factory=dict)
    output_schema: dict = Field(default_factory=dict)
    tools_used: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    version: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    flagged_for_review: bool = False
    requires_credentials: bool = False
    # Which entry in mcp_bindings.KNOWN_SERVICES this agent needs, if any.
    # Binding only ever happens behind the Phase 4 human-approval gate.
    mcp_service: str | None = None

    @property
    def key(self) -> str:
        """Registry primary key — unique per (user, type, id)."""
        return f"{self.user_id}/{self.type}/{self.id}"
