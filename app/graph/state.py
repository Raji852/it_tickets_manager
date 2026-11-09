from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict


class HelpdeskState(TypedDict, total=False):
    messages: list[dict[str, str]]
    session_id: str
    user_query: str
    intent: str
    is_it_related: bool
    current_agent: str
    agent_trace: Annotated[list[str], operator.add]
    tool_results: dict[str, Any]
    reasoning_notes: str
    action_mode: str  # reject_non_it | resolve | create_jira
    candidate_resolution: str
    needs_approval: bool
    approval_reason: str
    approved: Optional[bool]
    final_resolution: str
    grounded: bool
    jira_ticket_path: str
