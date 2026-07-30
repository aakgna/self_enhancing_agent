"""Offline tests for AgentSpec's defensive parsing. Regression test for a
live crash: structured-output tool calling (forced tool calling + thinking)
returned output_schema as a JSON-encoded string instead of a nested dict,
and pydantic rejected it outright."""

import pytest

from agent_maker.planner import AgentSpec


def test_dict_schema_passes_through_unchanged():
    spec = AgentSpec(proposed_agent_name="x", purpose="y", output_schema={"type": "object"})
    assert spec.output_schema == {"type": "object"}


def test_json_string_schema_is_coerced_to_dict():
    spec = AgentSpec(
        proposed_agent_name="x",
        purpose="y",
        output_schema='{"type": "object", "properties": {"a": {"type": "string"}}}',
    )
    assert spec.output_schema == {"type": "object", "properties": {"a": {"type": "string"}}}
    assert isinstance(spec.output_schema, dict)


def test_malformed_json_string_falls_back_to_empty_dict():
    spec = AgentSpec(proposed_agent_name="x", purpose="y", input_schema="not json at all {{{")
    assert spec.input_schema == {}


def test_missing_schema_defaults_to_empty_dict():
    spec = AgentSpec(proposed_agent_name="x", purpose="y")
    assert spec.input_schema == {}
    assert spec.output_schema == {}
