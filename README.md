# IT Tickets Manager

Local multi-agent IT helpdesk CLI using **LangGraph**, **OpenAI gpt-4o-mini**, and **Milvus Lite**.

Scope: user input + agent orchestration (no OpenTelemetry).

## Setup

```powershell
cd d:\it_tickets_manager
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
copy .env.example .env
```

Edit `.env` and set your real `OPENAI_API_KEY`.

## Ingest knowledge base (once)

```powershell
python -m scripts.ingest
```

## Run CLI

```powershell
# Single query
python -m app ask "I cannot connect to the VPN after laptop rebuild. Error 809." -v

# Interactive mode
python -m app interactive -v

# HITL demo (privileged request)
python -m app ask "Please reset MFA for my account. I lost my phone." -v

# Helpers
python -m app agents
python -m app tools
python -m app status
```
