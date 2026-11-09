from __future__ import annotations

from app.graph.state import HelpdeskState
from app.rag.milvus_store import MilvusKnowledgeStore


def knowledge_search_node(state: HelpdeskState) -> dict:
    """Always-on RAG retrieval from local Milvus."""
    store = MilvusKnowledgeStore()
    hits = store.search(state["user_query"])

    tool_results = dict(state.get("tool_results") or {})
    tool_results["knowledge"] = hits

    preview = ", ".join(h.get("ticket_id", "?") for h in hits) or "none"
    return {
        "tool_results": tool_results,
        "current_agent": "knowledge_search",
        "agent_trace": [f"knowledge_search:hits={preview}"],
    }
