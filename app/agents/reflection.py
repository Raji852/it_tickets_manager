from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.config import OPENAI_API_KEY, OPENAI_MODEL
from app.graph.state import HelpdeskState
from app.policy.engine import get_policy_engine
from app.tools.registry import get_tool_registry


def reflection_node(state: HelpdeskState) -> dict:
    """
    Groundedness check + policy gate.

    Sets needs_approval when policy or privileged tool rules fire.
    """
    llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)
    knowledge = (state.get("tool_results") or {}).get("knowledge") or []
    candidate = state.get("candidate_resolution", "")

    evidence_blob = "\n".join(
        h.get("resolution_summary", "") for h in knowledge
    ) or "(no evidence)"

    prompt = f"""Judge if the candidate resolution is reasonably grounded in the evidence.
Reply with exactly YES or NO, then a short reason.

Evidence:
{evidence_blob}

Candidate:
{candidate}
"""
    verdict = llm.invoke(prompt).content.strip()
    grounded = verdict.upper().startswith("YES")

    policy = get_policy_engine().evaluate(
        query=state.get("user_query", ""),
        intent=state.get("intent", ""),
    )
    registry = get_tool_registry()

    # Privileged tool is considered when access-like privileged intents appear.
    privileged_tool = registry.approval_required("privileged_access_change")
    intent = (state.get("intent") or "").lower()
    query_l = (state.get("user_query") or "").lower()
    looks_privileged = intent == "access" and any(
        k in query_l
        for k in ("mfa", "admin", "production", "prod ", "privileged", "reset")
    )

    needs_approval = bool(policy.needs_approval or (privileged_tool and looks_privileged))
    reason = policy.reason
    if needs_approval and not reason:
        reason = "Privileged access-related change requires human approval."

    updates: dict = {
        "grounded": grounded,
        "needs_approval": needs_approval,
        "approval_reason": reason,
        "current_agent": "reflection",
        "agent_trace": [
            f"reflection:grounded={grounded},needs_approval={needs_approval}"
        ],
    }

    if not needs_approval:
        updates["final_resolution"] = candidate
        updates["agent_trace"] = updates["agent_trace"] + ["reflection:auto_finalized"]

    return updates
