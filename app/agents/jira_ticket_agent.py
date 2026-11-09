from __future__ import annotations

import json
from datetime import datetime, timezone

from langchain_openai import ChatOpenAI

from app.config import OPENAI_API_KEY, OPENAI_MODEL
from app.graph.state import HelpdeskState
from app.jira_ticket import save_jira_ticket_template
from app.tools.registry import get_tool_registry


def jira_ticket_node(state: HelpdeskState) -> dict:
    """Create a Jira ticket template instead of a self-service action resolution."""
    llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)
    _ = get_tool_registry().get("create_jira_ticket")

    knowledge = (state.get("tool_results") or {}).get("knowledge") or []
    evidence = "\n".join(
        f"- [{h.get('ticket_id')}] {h.get('resolution_summary')}" for h in knowledge
    )
    access = (state.get("tool_results") or {}).get("access") or {}

    prompt = f"""You create IT helpdesk Jira ticket templates.
Return ONLY valid JSON with these keys:
project_key, issue_type, priority, summary, description, component,
labels (array of strings), reporter, assignee, acceptance_criteria.

Rules:
- project_key must be "IT"
- issue_type one of: Incident, Service Request, Problem
- priority one of: Highest, High, Medium, Low
- summary <= 100 chars
- description should include the user issue and context from evidence
- reporter: "end_user"
- assignee: "unassigned"
- labels should include the intent category when known

User issue: {state.get("user_query")}
Intent: {state.get("intent")}
Access/SLA: {json.dumps(access)}
Reasoning notes:
{state.get("reasoning_notes", "")}
Similar past resolutions:
{evidence or "(none)"}
"""
    raw = (llm.invoke(prompt).content or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()

    try:
        ticket = json.loads(raw)
    except json.JSONDecodeError:
        ticket = {
            "project_key": "IT",
            "issue_type": "Incident",
            "priority": "Medium",
            "summary": (state.get("user_query") or "IT support request")[:100],
            "description": state.get("user_query") or "",
            "component": state.get("intent") or "general_it",
            "labels": [state.get("intent") or "general_it"],
            "reporter": "end_user",
            "assignee": "unassigned",
            "acceptance_criteria": "Issue investigated and resolved or workaround provided.",
        }

    ticket["session_id"] = state.get("session_id") or ""
    ticket["generated_at"] = datetime.now(timezone.utc).isoformat()
    ticket.setdefault("project_key", "IT")

    path = save_jira_ticket_template(ticket)

    final = (
        "A Jira ticket template has been created for this request "
        "(action deferred to ticket workflow).\n\n"
        f"Summary : {ticket.get('summary')}\n"
        f"Type    : {ticket.get('issue_type')}\n"
        f"Priority: {ticket.get('priority')}\n"
        f"Saved   : {path}"
    )

    tool_results = dict(state.get("tool_results") or {})
    tool_results["jira"] = {
        "tool": "create_jira_ticket",
        "path": str(path),
        "ticket": ticket,
    }

    return {
        "action_mode": "create_jira",
        "jira_ticket_path": str(path),
        "candidate_resolution": final,
        "final_resolution": final,
        "grounded": True,
        "needs_approval": False,
        "tool_results": tool_results,
        "current_agent": "jira_ticket",
        "agent_trace": [f"jira_ticket:saved={path.name}"],
    }
