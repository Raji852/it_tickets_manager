from __future__ import annotations

from app.graph.state import HelpdeskState


def access_check_node(state: HelpdeskState) -> dict:
    """
    Mock account/SLA check.

    Prefer metadata from the top knowledge hit when available;
    otherwise fall back to safe defaults.
    """
    knowledge = (state.get("tool_results") or {}).get("knowledge") or []
    top = knowledge[0] if knowledge else {}

    access = {
        "user_status": top.get("user_status") or "active",
        "sla": top.get("sla") or "standard",
        "intent": state.get("intent", "general"),
        "source_ticket": top.get("ticket_id"),
        "account_ok": (top.get("user_status") or "active") == "active",
    }

    tool_results = dict(state.get("tool_results") or {})
    tool_results["access"] = access

    return {
        "tool_results": tool_results,
        "current_agent": "access_check",
        "agent_trace": [
            f"access_check:status={access['user_status']},sla={access['sla']}"
        ],
    }
