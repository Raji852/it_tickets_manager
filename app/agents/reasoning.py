from __future__ import annotations

import json

from langchain_openai import ChatOpenAI

from app.config import OPENAI_API_KEY, OPENAI_MODEL
from app.graph.state import HelpdeskState


def reasoning_node(state: HelpdeskState) -> dict:
    """Draft root-cause analysis and next steps from tool evidence."""
    llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)

    knowledge = (state.get("tool_results") or {}).get("knowledge") or []
    access = (state.get("tool_results") or {}).get("access") or {}

    evidence = [
        {
            "ticket_id": h.get("ticket_id"),
            "category": h.get("category"),
            "resolution_summary": h.get("resolution_summary"),
        }
        for h in knowledge
    ]

    prompt = f"""You are an IT helpdesk reasoning agent.
Analyze the user issue using ONLY the provided evidence.
Return concise bullet points covering: likely cause, recommended steps, risks.

User issue: {state.get("user_query")}
Intent: {state.get("intent")}
Access/SLA: {json.dumps(access)}
Similar past resolutions: {json.dumps(evidence, indent=2)}
"""
    notes = llm.invoke(prompt).content.strip()

    return {
        "reasoning_notes": notes,
        "current_agent": "reasoning",
        "agent_trace": ["reasoning:drafted_notes"],
    }
