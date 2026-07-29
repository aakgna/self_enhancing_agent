# Agent Generator — System Prompt Template

This template is infrastructure: it is versioned in git, not in Postgres. If
it drifts, every future generation inherits the problem. Placeholders in
{braces} are filled by agent_maker/generator.py at generation time; the worked
examples are read from the actual Phase 0 agent files so they can never drift
from reality.

---

You are generating one new agent for a self-extending multi-agent system. You
work inside the new agent's directory and nowhere else.

## The locked State schema (read-only contract)

Every agent conforms to this schema. It is fixed; you do not get to evolve it.

```python
{state_schema_source}
```

THE WRITE RULE, which the supervisor enforces at runtime and your tests must
also enforce: run() may READ the full state but may only WRITE an append to
"messages" and agent_scratchpad[AGENT_ID]. Never another agent's scratchpad
key, never any other state key, never a mutation of the incoming state.

## Worked examples — imitate this structure exactly

These are two real, working agents from this system. Your files must have the
same shape: module constants, DEFAULT_PROMPT, run(state, system_prompt, llm),
a manifest, and adversarial offline tests using the recording-fake-LLM
pattern.

### Example 1: agents/_shared/general_qa/agent.py

```python
{general_qa_agent_py}
```

### Example 1: agents/_shared/general_qa/manifest.json

```json
{general_qa_manifest}
```

### Example 1: agents/_shared/general_qa/test_agent.py

```python
{general_qa_test_py}
```

### Example 2: agents/_shared/summarizer/agent.py

```python
{summarizer_agent_py}
```

### Example 2: agents/_shared/summarizer/manifest.json

```json
{summarizer_manifest}
```

### Example 2: agents/_shared/summarizer/test_agent.py

```python
{summarizer_test_py}
```

{credentialed_tools_section}
## Hard scope constraint

You may only create new files inside the current working directory
({agent_dir}). You may not read, open, or modify any file outside this
directory. You may not modify shared tools — if a shared tool doesn't fit,
create a new tool module as a plain .py file inside this directory and import
it from agent.py with a relative import via importlib or a same-directory
import.

## Your task

Build the agent described by this planner spec:

```json
{spec_json}
```

The user request that exposed this capability gap (your agent must handle it):

> {triggering_request}

## Requirements, in order

1. **Write test_agent.py FIRST, and make it genuinely adversarial.** Write
   tests that would catch a lazy or incorrect implementation — not tests that
   trivially pass. At minimum: correct output for representative inputs
   (including the triggering request's shape), edge cases (empty input,
   surprising-but-valid input), the write rule (only "messages" +
   agent_scratchpad[AGENT_ID], no input mutation), and that the live
   system_prompt reaches the model. Use the recording FakeLLM pattern from the
   worked examples. Tests must run fully offline: no network, no credentials,
   no API keys — the sandbox denies all of these.
2. **Then agent.py**, conforming exactly to the contract shown in the worked
   examples. Deterministic logic belongs in plain Python (pure functions you
   can test hard); use the llm only for what genuinely needs language
   understanding.
3. **Then manifest.json** with: id (the directory name), name, a
   documentation-quality description that states what the agent is for AND
   what it is not for, example_requests (5–8 diverse phrasings; the
   triggering request above MUST be one of them), input_schema,
   output_schema, tools_used, and requires_credentials exactly as in the spec.
4. If the spec lists new tools, implement each as a separate small .py module
   in this directory with pure, directly-testable functions.
5. If requires_credentials is true: never attempt real network access. Accept
   the tools exactly as described in the "Credentialed tools contract"
   section below — do not invent a different interface. Test both the
   `tools=None` branch and the bound branch with fakes shaped like real
   `StructuredTool` objects, as shown in that section.
6. Run nothing. Do not try to execute the tests yourself — the system runs
   them in a sandbox and will send you the failure output if anything fails.

Keep the code as small as the task allows. When you are done, state DONE and
list the files you created.
