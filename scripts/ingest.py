"""Seed local Milvus Lite with golden IT resolution summaries."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python -m scripts.ingest` from project root without install.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import TICKETS_CSV, require_api_key
from app.rag.milvus_store import MilvusKnowledgeStore


def main() -> None:
    require_api_key()
    store = MilvusKnowledgeStore()
    print(f"Ingesting: {TICKETS_CSV}")
    print(f"Milvus DB: {store.uri}")
    count = store.ingest_csv(TICKETS_CSV, drop=True)
    print(f"Upserted {count} resolutions into collection '{store.collection}'.")
    print(f"Row count now: {store.count()}")


if __name__ == "__main__":
    main()
