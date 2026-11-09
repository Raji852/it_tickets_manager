from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.config import OPENAI_API_KEY, OPENAI_MODEL
from app.graph.state import HelpdeskState

NON_IT_RESPONSE = (
    "Couldn't provide the response for the query. Please ask IT related questions"
)

VALID_IT_INTENTS = {"network", "access", "software", "hardware", "general_it"}


def intent_analyzer_node(state: HelpdeskState) -> dict:
    """
    Understand the user question.

    Non-IT / generic questions stop here with a fixed response.
    IT questions continue with a category label.
    """
    query = state.get("user_query") or state.get("messages", [{}])[-1].get("content", "")

    llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)
    prompt = f"""You are an IT helpdesk intent analyzer.
Decide if the user message is an IT support question
(network, VPN, laptop, software, email, access/IAM, password, printer, hardware, etc.).

If it is NOT IT-related (weather, sports, jokes, recipes, general trivia, etc.),
reply with exactly:
NOT_IT

If it IS IT-related, reply with exactly one label:
network
access
software
hardware
general_it

User message: {query}
"""
    raw = (llm.invoke(prompt).content or "").strip().lower().replace("`", "")
    first_token = raw.split()[0] if raw else "not_it"

    if first_token == "not_it" or raw.startswith("not_it"):
        return {
            "user_query": query,
            "is_it_related": False,
            "intent": "not_it",
            "current_agent": "intent_analyzer",
            "agent_trace": ["intent_analyzer:not_it"],
            "tool_results": state.get("tool_results") or {},
            "needs_approval": False,
            "approved": None,
            "final_resolution": NON_IT_RESPONSE,
            "action_mode": "reject_non_it",
        }

    intent = first_token if first_token in VALID_IT_INTENTS else "general_it"
    return {
        "user_query": query,
        "is_it_related": True,
        "intent": intent,
        "current_agent": "intent_analyzer",
        "agent_trace": [f"intent_analyzer:it intent={intent}"],
        "tool_results": state.get("tool_results") or {},
        "needs_approval": False,
        "approved": None,
        "action_mode": "",
    }
