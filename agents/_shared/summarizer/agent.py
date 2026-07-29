"""Summarizer agent: condenses text the user supplies.

Follows the standard agent contract (see general_qa/agent.py for the full
statement of the contract and the write rule). This agent shows the other
common shape: instead of passing the whole conversation through, it extracts
the material to work on and builds an explicit instruction.
"""

from langchain_core.messages import HumanMessage, SystemMessage

AGENT_ID = "summarizer"
AGENT_VERSION = 1

DEFAULT_PROMPT = (
    "You are a summarization specialist. Produce a faithful, compact summary "
    "of the text you are given: lead with the main point, keep essential "
    "facts and numbers, and drop filler. Never add information that is not "
    "in the source text."
)


def _latest_human_text(state: dict) -> str:
    for message in reversed(state["messages"]):
        if getattr(message, "type", None) == "human":
            return message.content if isinstance(message.content, str) else str(message.content)
    return ""


def run(state: dict, system_prompt: str, llm) -> dict:
    source_text = _latest_human_text(state)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Summarize the following:\n\n{source_text}"),
    ]
    reply = llm.invoke(messages)

    summary = reply.content if isinstance(reply.content, str) else str(reply.content)
    return {
        "messages": [reply],
        "agent_scratchpad": {
            AGENT_ID: {
                "source_chars": len(source_text),
                "summary_chars": len(summary),
            }
        },
    }
