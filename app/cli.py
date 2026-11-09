from __future__ import annotations

import json
from pathlib import Path

import click

from app.config import REGISTRIES_DIR, ROOT_DIR, require_api_key
from app.graph.workflow import run_ticket
from app.pdf_response import save_final_resolution_pdf
from app.rag.milvus_store import MilvusKnowledgeStore
from app.tools.registry import get_tool_registry


def _print_banner() -> None:
    click.echo("=" * 64)
    click.echo(" IT Helpdesk Multi-Agent Ticket Manager (CLI)")
    click.echo(" LangGraph + OpenAI gpt-4o-mini + Milvus Lite")
    click.echo("=" * 64)


def _approval_prompt(payload: dict) -> bool:
    click.echo("")
    click.secho("HUMAN APPROVAL REQUIRED", fg="yellow", bold=True)
    click.echo(f"Reason : {payload.get('reason', 'n/a')}")
    click.echo(f"Intent : {payload.get('intent', 'n/a')}")
    click.echo("-" * 64)
    click.echo(payload.get("proposal", ""))
    click.echo("-" * 64)
    return click.confirm("Approve this privileged resolution?", default=False)


def _print_result(result: dict, verbose: bool = False, user_query: str = "") -> None:
    click.echo("")
    click.secho("AGENT TRACE", fg="cyan", bold=True)
    for step in result.get("agent_trace") or []:
        click.echo(f"  • {step}")

    click.echo("")
    click.echo(f"Intent     : {result.get('intent')}")
    click.echo(f"IT related : {result.get('is_it_related')}")
    click.echo(f"Action mode: {result.get('action_mode')}")
    click.echo(f"Grounded   : {result.get('grounded')}")
    click.echo(f"HITL used  : {result.get('needs_approval')}")
    if result.get("needs_approval"):
        click.echo(f"HITL result: {result.get('approved')}")
        click.echo(f"HITL reason: {result.get('approval_reason')}")
    if result.get("jira_ticket_path"):
        click.secho(f"Jira file  : {result.get('jira_ticket_path')}", fg="yellow")

    if verbose:
        knowledge = (result.get("tool_results") or {}).get("knowledge") or []
        click.echo("")
        click.secho("KNOWLEDGE HITS", fg="cyan", bold=True)
        for hit in knowledge:
            click.echo(
                f"  [{hit.get('ticket_id')}] score={hit.get('score'):.4f} "
                f"({hit.get('category')})"
            )
            click.echo(f"    {hit.get('resolution_summary')}")

        click.echo("")
        click.secho("REASONING NOTES", fg="cyan", bold=True)
        click.echo(result.get("reasoning_notes") or "")

    final_resolution = result.get("final_resolution") or "(empty)"

    click.echo("")
    click.secho("FINAL RESOLUTION", fg="green", bold=True)
    click.echo(final_resolution)
    click.echo("")

    pdf_path = save_final_resolution_pdf(
        final_resolution=final_resolution,
        user_query=user_query or "",
        session_id=result.get("session_id") or "",
        intent=result.get("intent") or "",
        approved=result.get("approved"),
        needs_approval=bool(result.get("needs_approval")),
    )
    click.secho(f"PDF saved   : {pdf_path}", fg="cyan")
    click.echo("")


@click.group()
def main() -> None:
    """IT Helpdesk multi-agent CLI."""


@main.command("ask")
@click.argument("query", required=False)
@click.option("--verbose", "-v", is_flag=True, help="Show knowledge hits and reasoning.")
def ask(query: str | None, verbose: bool) -> None:
    """Submit one IT helpdesk query through the agent graph."""
    require_api_key()
    _print_banner()

    store = MilvusKnowledgeStore()
    if store.count() == 0:
        raise SystemExit(
            "Knowledge base is empty. Run first:\n"
            "  python -m scripts.ingest"
        )

    if not query:
        query = click.prompt("Enter IT helpdesk issue")

    click.echo("")
    click.echo(f"Query: {query}")
    click.echo("Running agents...")

    result = run_ticket(query, approval_callback=_approval_prompt)
    _print_result(result, verbose=verbose, user_query=query)


@main.command("interactive")
@click.option("--verbose", "-v", is_flag=True)
def interactive(verbose: bool) -> None:
    """Repeatedly accept CLI queries until you type exit."""
    require_api_key()
    _print_banner()

    store = MilvusKnowledgeStore()
    if store.count() == 0:
        raise SystemExit("Knowledge base is empty. Run: python -m scripts.ingest")

    click.echo("Type your IT issue (or 'exit' to quit).")
    while True:
        query = click.prompt("\nhelpdesk>").strip()
        if query.lower() in {"exit", "quit", "q"}:
            click.echo("Bye.")
            break
        if not query:
            continue
        result = run_ticket(query, approval_callback=_approval_prompt)
        _print_result(result, verbose=verbose, user_query=query)


@main.command("agents")
def agents() -> None:
    """List agents from the AI registry."""
    path = REGISTRIES_DIR / "agents.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for agent in data.get("agents", []):
        caps = ", ".join(agent.get("capabilities", []))
        click.echo(f"- {agent['name']}: {agent.get('description')} [{caps}]")


@main.command("tools")
def tools() -> None:
    """List tools from the tool registry."""
    for tool in get_tool_registry().list_tools():
        flag = "APPROVAL" if tool.get("approval_required") else "auto"
        click.echo(
            f"- {tool['name']} v{tool.get('version')} [{flag}]: {tool.get('description')}"
        )


@main.command("status")
def status() -> None:
    """Show local Milvus knowledge-base status."""
    require_api_key()
    store = MilvusKnowledgeStore()
    click.echo(f"Project root : {ROOT_DIR}")
    click.echo(f"Milvus URI   : {store.uri}")
    click.echo(f"Collection   : {store.collection}")
    click.echo(f"Documents    : {store.count()}")


if __name__ == "__main__":
    main()
