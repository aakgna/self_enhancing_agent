"""General-purpose Q&A agent.

Every agent module follows this exact contract:

  - AGENT_ID / AGENT_VERSION module constants.
  - DEFAULT_PROMPT: the seed system prompt. The live prompt is versioned in
    Postgres and passed in by the supervisor — never read this constant at
    runtime.
  - run(state, system_prompt, llm) -> partial state update.

The write rule: run() may only append to "messages" and write
agent_scratchpad[AGENT_ID]. It must never mutate the incoming state, touch
another agent's scratchpad key, or return any other state key.
"""

from langchain_core.messages import SystemMessage

AGENT_ID = "general_qa"
AGENT_VERSION = 1

DEFAULT_PROMPT = (
    "You are a concise, accurate general-purpose assistant. Answer the user's "
    "question directly. If you do not know the answer, say so plainly instead "
    "of guessing."
)


def run(state: dict, system_prompt: str, llm) -> dict:
    # Full conversation context, with the live system prompt in front.
    messages = [SystemMessage(content=system_prompt), *state["messages"]]
    reply = llm.invoke(messages)

    own_scratch = state["agent_scratchpad"].get(AGENT_ID, {})
    turns_handled = own_scratch.get("turns_handled", 0) + 1

    return {
        "messages": [reply],
        "agent_scratchpad": {AGENT_ID: {"turns_handled": turns_handled}},
    }
