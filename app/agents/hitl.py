from __future__ import annotations

from langgraph.types import interrupt

from app.graph.state import HelpdeskState


def hitl_node(state: HelpdeskState) -> dict:
    """Pause for human approval when policy requires it."""
    if not state.get("needs_approval"):
        return {
            "final_resolution": state.get("candidate_resolution", ""),
            "current_agent": "hitl",
            "agent_trace": ["hitl:skipped"],
        }

    decision = interrupt(
        {
            "type": "approval_required",
            "reason": state.get("approval_reason", "Policy requires approval"),
            "proposal": state.get("candidate_resolution", ""),
            "intent": state.get("intent"),
        }
    )

    approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)

    if approved:
        return {
            "approved": True,
            "final_resolution": state.get("candidate_resolution", ""),
            "current_agent": "hitl",
            "agent_trace": ["hitl:approved"],
        }

    return {
        "approved": False,
        "final_resolution": (
            "Request rejected by Admin. "
            "No privileged change was applied. Please check with your IT Team."
        ),
        "current_agent": "hitl",
        "agent_trace": ["hitl:rejected"],
    }
