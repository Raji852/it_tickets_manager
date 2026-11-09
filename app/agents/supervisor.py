from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.config import OPENAI_API_KEY, OPENAI_MODEL
from app.graph.state import HelpdeskState

VALID_INTENTS = {"network", "access", "software", "hardware", "general"}


def supervisor_node(state: HelpdeskState) -> dict:
    """Classify intent and initialize routing metadata."""
    query = state.get("user_query") or state.get("messages", [{}])[-1].get("content", "")

    llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)
    prompt = (
        "Classify this IT helpdesk issue into exactly one label: "
        "network, access, software, hardware, or general.\n"
        "Reply with only the label.\n\n"
        f"Issue: {query}"
    )
    raw = llm.invoke(prompt).content.strip().lower()
    intent = raw if raw in VALID_INTENTS else "general"

    return {
        "user_query": query,
        "intent": intent,
        "current_agent": "supervisor",
        "agent_trace": [f"supervisor:intent={intent}"],
        "tool_results": state.get("tool_results") or {},
        "needs_approval": False,
        "approved": None,
    }
