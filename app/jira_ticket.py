from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import ROOT_DIR

JIRA_TICKETS_DIR = ROOT_DIR / "jira_tickets"


def _safe_slug(text: str, max_len: int = 40) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", (text or "ticket").strip())
    cleaned = cleaned.strip("_") or "ticket"
    return cleaned[:max_len]


def save_jira_ticket_template(ticket: dict[str, Any]) -> Path:
    """Persist a Jira ticket template (JSON + readable text) under jira_tickets/."""
    JIRA_TICKETS_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    key = ticket.get("project_key", "IT")
    summary_slug = _safe_slug(str(ticket.get("summary", "issue")))
    base = f"{key}_{stamp}_{summary_slug}"

    json_path = JIRA_TICKETS_DIR / f"{base}.json"
    txt_path = JIRA_TICKETS_DIR / f"{base}.txt"

    json_path.write_text(json.dumps(ticket, indent=2), encoding="utf-8")

    txt = "\n".join(
        [
            "JIRA TICKET TEMPLATE",
            "=" * 64,
            f"Project     : {ticket.get('project_key', '')}",
            f"Issue Type  : {ticket.get('issue_type', '')}",
            f"Priority    : {ticket.get('priority', '')}",
            f"Summary     : {ticket.get('summary', '')}",
            f"Component   : {ticket.get('component', '')}",
            f"Labels      : {', '.join(ticket.get('labels') or [])}",
            f"Reporter    : {ticket.get('reporter', '')}",
            f"Assignee    : {ticket.get('assignee', '')}",
            "-" * 64,
            "Description:",
            str(ticket.get("description", "")),
            "-" * 64,
            "Acceptance Criteria:",
            str(ticket.get("acceptance_criteria", "")),
            "=" * 64,
            f"Generated   : {ticket.get('generated_at', '')}",
            f"Session ID  : {ticket.get('session_id', '')}",
        ]
    )
    txt_path.write_text(txt, encoding="utf-8")
    return json_path
