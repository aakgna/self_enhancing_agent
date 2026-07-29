"""LLM factories. Agents get Opus with adaptive thinking; routing/classification
gets Haiku (cheap, fast — it's a routing decision, not a reasoning task)."""

from functools import lru_cache

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

AGENT_MODEL = "claude-opus-4-8"
FAST_MODEL = "claude-haiku-4-5"


@lru_cache(maxsize=1)
def agent_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=AGENT_MODEL,
        max_tokens=8192,
        thinking={"type": "adaptive"},
    )


@lru_cache(maxsize=1)
def fast_llm() -> ChatAnthropic:
    return ChatAnthropic(model=FAST_MODEL, max_tokens=1024)
