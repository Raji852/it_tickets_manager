from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)

# Do NOT name this MILVUS_URI — pymilvus reads that env var as a server URL
# (http/https) at import time and will crash on a local .db path.
_default_db = ROOT_DIR / "data" / "milvus_it_helpdesk.db"
MILVUS_DB_PATH = str(
    Path(os.getenv("MILVUS_DB_PATH", str(_default_db))).expanduser().resolve()
)
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "it_resolutions")

# Clear a mistaken local-file MILVUS_URI so pymilvus import does not fail.
_milvus_uri_env = os.environ.get("MILVUS_URI", "")
if _milvus_uri_env and not _milvus_uri_env.startswith(("http://", "https://")):
    os.environ.pop("MILVUS_URI", None)

DATA_DIR = ROOT_DIR / "data"
REGISTRIES_DIR = ROOT_DIR / "registries"
MESSAGE_TO_USER_DIR = ROOT_DIR / "message_to_user"
JIRA_TICKETS_DIR = ROOT_DIR / "jira_tickets"
TICKETS_CSV = DATA_DIR / "golden_it_support_tickets.csv"

TOP_K = 3


def require_api_key() -> None:
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-your"):
        raise SystemExit(
            "OPENAI_API_KEY is missing. Copy .env.example to .env and set your key."
        )
