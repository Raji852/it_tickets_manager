from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.config import OPENAI_API_KEY, OPENAI_MODEL
from app.graph.state import HelpdeskState
from app.tools.registry import get_tool_registry


def _decide_action_mode(state: HelpdeskState, llm: ChatOpenAI) -> str:
    """Choose resolve (self-service steps) vs create_jira (formal ticket)."""
    query = (state.get("user_query") or "").lower()
    explicit = any(
        phrase in query
        for phrase in (
            "create a ticket",
            "create ticket",
            "raise a ticket",
            "raise ticket",
            "open a jira",
            "open jira",
            "create jira",
            "file a ticket",
            "log a ticket",
        )
    )
    if explicit:
        return "create_jira"

    prompt = f"""You route IT helpdesk work.
Choose exactly one mode:
resolve — user can follow self-service steps now
create_jira — needs a formal ticket (hardware replacement, depot, major incident,
              privileged provisioning that must be tracked, or no safe self-service path)

Reply with only: resolve OR create_jira

User issue: {state.get("user_query")}
Intent: {state.get("intent")}
Reasoning notes:
{state.get("reasoning_notes", "")}
"""
    raw = (llm.invoke(prompt).content or "").strip().lower()
    if "create_jira" in raw or raw.startswith("create"):
        return "create_jira"
    return "resolve"


def action_node(state: HelpdeskState) -> dict:
    """
    Either draft a user-facing resolution, or signal that a Jira ticket is needed.
    """
    llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)
    registry = get_tool_registry()
    mode = _decide_action_mode(state, llm)

    if mode == "create_jira":
        return {
            "action_mode": "create_jira",
            "current_agent": "action",
            "agent_trace": ["action:route=create_jira"],
        }

    _ = registry.get("propose_resolution")
    knowledge = (state.get("tool_results") or {}).get("knowledge") or []
    evidence = "\n".join(
        f"- [{h.get('ticket_id')}] {h.get('resolution_summary')}" for h in knowledge
    )

    prompt = f"""You are an IT helpdesk action agent.
Write a clear resolution the helpdesk can send to the user.
Use the reasoning notes and similar past resolutions.
Do not invent company systems that are not implied by the evidence.
Keep it under 180 words. Use numbered steps when useful.

User issue: {state.get("user_query")}
Reasoning notes:
{state.get("reasoning_notes", "")}

Similar resolutions:
{evidence or "(none)"}
"""
    candidate = llm.invoke(prompt).content.strip()

    tool_results = dict(state.get("tool_results") or {})
    tool_results["action"] = {
        "tool": "propose_resolution",
        "candidate_resolution": candidate,
    }

    return {
        "action_mode": "resolve",
        "candidate_resolution": candidate,
        "tool_results": tool_results,
        "current_agent": "action",
        "agent_trace": ["action:proposed_resolution"],
    }
