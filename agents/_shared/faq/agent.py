"""FAQ agent: answers general clinic/front-desk questions from a static
knowledge base.

Follows the standard agent contract (see general_qa/agent.py for the full
statement of the contract and the write rule). Same shape as general_qa (one
llm.invoke() over the conversation) — there's no credential and no external
call, so no tool-calling loop is needed. The one addition is that the
knowledge base is loaded once at import time and appended to the live,
versioned system_prompt at call time, mirroring how the ported source project
(FurHat's app/knowledge/faq.py + data/hospital_faqs.json) keeps persona/
behavior instructions separate from FAQ content. Local-only: if a question
isn't covered by the knowledge base, the agent says so plainly and suggests
asking staff, instead of guessing or reaching out to the web.
"""

import json
from pathlib import Path

from langchain_core.messages import SystemMessage

AGENT_ID = "faq"
AGENT_VERSION = 1

DEFAULT_PROMPT = (
    "You are the front-desk FAQ specialist at a clinic. Answer the patient's "
    "question using ONLY the knowledge base provided below — never guess or "
    "invent facility details. If the knowledge base does not cover the "
    "question, say so plainly and suggest asking a staff member instead of "
    "making something up. Answer conversationally in 1-3 sentences, with no "
    "bullet points or markdown."
)

_DATA_PATH = Path(__file__).parent / "data" / "hospital_faqs.json"
_ENTRIES = json.loads(_DATA_PATH.read_text())


def _corpus() -> str:
    return "\n".join(f"[{i}] Q: {e['question']}\nA: {e['answer']}" for i, e in enumerate(_ENTRIES, 1))


def run(state: dict, system_prompt: str, llm) -> dict:
    full_prompt = f"{system_prompt}\n\nKNOWLEDGE BASE:\n{_corpus()}"
    messages = [SystemMessage(content=full_prompt), *state["messages"]]
    reply = llm.invoke(messages)

    own_scratch = state["agent_scratchpad"].get(AGENT_ID, {})
    turns_handled = own_scratch.get("turns_handled", 0) + 1

    return {
        "messages": [reply],
        "agent_scratchpad": {AGENT_ID: {"turns_handled": turns_handled}},
    }
