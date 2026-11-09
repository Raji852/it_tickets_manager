from __future__ import annotations

import uuid
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.agents.access import access_check_node
from app.agents.action import action_node
from app.agents.hitl import hitl_node
from app.agents.intent_analyzer import intent_analyzer_node
from app.agents.jira_ticket_agent import jira_ticket_node
from app.agents.knowledge import knowledge_search_node
from app.agents.reasoning import reasoning_node
from app.agents.reflection import reflection_node
from app.graph.state import HelpdeskState


def _route_after_intent(state: HelpdeskState) -> str:
    if state.get("is_it_related") is False:
        return "end"
    return "knowledge_search"


def _route_after_action(state: HelpdeskState) -> str:
    if state.get("action_mode") == "create_jira":
        return "jira_ticket"
    return "reflection"


def _route_after_reflection(state: HelpdeskState) -> str:
    if state.get("needs_approval"):
        return "hitl"
    return "end"


def build_graph():
    """
    intent_analyzer
      ├─ (not IT) → END
      └─ knowledge_search → access_check → reasoning → action
            ├─ create_jira → jira_ticket → END
            └─ resolve → reflection → (hitl?) → END
    """
    graph = StateGraph(HelpdeskState)

    graph.add_node("intent_analyzer", intent_analyzer_node)
    graph.add_node("knowledge_search", knowledge_search_node)
    graph.add_node("access_check", access_check_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("action", action_node)
    graph.add_node("jira_ticket", jira_ticket_node)
    graph.add_node("reflection", reflection_node)
    graph.add_node("hitl", hitl_node)

    graph.add_edge(START, "intent_analyzer")
    graph.add_conditional_edges(
        "intent_analyzer",
        _route_after_intent,
        {"knowledge_search": "knowledge_search", "end": END},
    )
    graph.add_edge("knowledge_search", "access_check")
    graph.add_edge("access_check", "reasoning")
    graph.add_edge("reasoning", "action")
    graph.add_conditional_edges(
        "action",
        _route_after_action,
        {"jira_ticket": "jira_ticket", "reflection": "reflection"},
    )
    graph.add_edge("jira_ticket", END)
    graph.add_conditional_edges(
        "reflection",
        _route_after_reflection,
        {"hitl": "hitl", "end": END},
    )
    graph.add_edge("hitl", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def run_ticket(
    user_query: str,
    *,
    approval_callback=None,
    graph=None,
) -> dict[str, Any]:
    """Execute one helpdesk ticket through the graph."""
    app = graph or build_graph()
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    initial: HelpdeskState = {
        "messages": [{"role": "user", "content": user_query}],
        "user_query": user_query,
        "session_id": session_id,
        "agent_trace": [],
        "tool_results": {},
    }

    result = app.invoke(initial, config=config)

    while True:
        snapshot = app.get_state(config)
        interrupted = bool(snapshot.tasks) and any(
            getattr(t, "interrupts", None) for t in snapshot.tasks
        )
        if not interrupted:
            break

        payload = None
        for task in snapshot.tasks:
            if task.interrupts:
                payload = task.interrupts[0].value
                break

        if approval_callback is None:
            approved = False
        else:
            approved = bool(approval_callback(payload or {}))

        result = app.invoke(
            Command(resume={"approved": approved}),
            config=config,
        )

    return {
        "session_id": session_id,
        "intent": result.get("intent"),
        "is_it_related": result.get("is_it_related"),
        "action_mode": result.get("action_mode"),
        "agent_trace": result.get("agent_trace", []),
        "tool_results": result.get("tool_results", {}),
        "reasoning_notes": result.get("reasoning_notes"),
        "needs_approval": result.get("needs_approval", False),
        "approval_reason": result.get("approval_reason"),
        "approved": result.get("approved"),
        "grounded": result.get("grounded"),
        "final_resolution": result.get("final_resolution"),
        "candidate_resolution": result.get("candidate_resolution"),
        "jira_ticket_path": result.get("jira_ticket_path"),
    }
