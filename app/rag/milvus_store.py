from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

# Import config before pymilvus so a local-file MILVUS_URI is cleared
# (pymilvus treats MILVUS_URI as an http server URL at import time).
from app.config import (
    MILVUS_COLLECTION,
    MILVUS_DB_PATH,
    TICKETS_CSV,
    TOP_K,
)
from pymilvus import DataType, MilvusClient

from app.rag.embeddings import get_embeddings


class MilvusKnowledgeStore:
    """Local Milvus Lite store for resolution summaries."""

    def __init__(
        self,
        uri: str | None = None,
        collection: str | None = None,
    ) -> None:
        self.uri = str(Path(uri or MILVUS_DB_PATH).expanduser().resolve())
        self.collection = collection or MILVUS_COLLECTION
        Path(self.uri).parent.mkdir(parents=True, exist_ok=True)
        self.client = MilvusClient(uri=self.uri)
        self.embeddings = get_embeddings()
        self.dim = 1536  # text-embedding-3-small

    def ensure_collection(self, drop: bool = False) -> None:
        if drop and self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)

        if self.client.has_collection(self.collection):
            return

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=32)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field("ticket_id", DataType.VARCHAR, max_length=32)
        schema.add_field("category", DataType.VARCHAR, max_length=64)
        schema.add_field("priority", DataType.VARCHAR, max_length=32)
        schema.add_field("initial_message", DataType.VARCHAR, max_length=2048)
        schema.add_field("resolution_summary", DataType.VARCHAR, max_length=4096)
        schema.add_field("user_status", DataType.VARCHAR, max_length=32)
        schema.add_field("sla", DataType.VARCHAR, max_length=32)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        self.client.create_collection(
            collection_name=self.collection,
            schema=schema,
            index_params=index_params,
        )

    def ingest_csv(self, csv_path: Path | None = None, drop: bool = True) -> int:
        path = csv_path or TICKETS_CSV
        df = pd.read_csv(path)
        self.ensure_collection(drop=drop)

        texts = [
            f"Issue: {row.initial_message}\nResolution: {row.resolution_summary}"
            for row in df.itertuples(index=False)
        ]
        vectors = self.embeddings.embed_documents(texts)

        rows: list[dict[str, Any]] = []
        for i, row in enumerate(df.itertuples(index=False)):
            rows.append(
                {
                    "id": str(row.ticket_id),
                    "vector": vectors[i],
                    "ticket_id": str(row.ticket_id),
                    "category": str(row.category),
                    "priority": str(row.priority),
                    "initial_message": str(row.initial_message)[:2000],
                    "resolution_summary": str(row.resolution_summary)[:4000],
                    "user_status": str(row.user_status),
                    "sla": str(row.sla),
                }
            )

        self.client.upsert(collection_name=self.collection, data=rows)
        self.client.load_collection(self.collection)
        return len(rows)

    def _ensure_loaded(self) -> None:
        """Milvus requires the collection to be loaded before search/query."""
        self.client.load_collection(self.collection)

    def search(self, query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
        if not self.client.has_collection(self.collection):
            raise RuntimeError(
                "Milvus collection is empty. Run: python -m scripts.ingest"
            )

        self._ensure_loaded()
        vector = self.embeddings.embed_query(query)
        results = self.client.search(
            collection_name=self.collection,
            data=[vector],
            limit=top_k,
            output_fields=[
                "ticket_id",
                "category",
                "priority",
                "initial_message",
                "resolution_summary",
                "user_status",
                "sla",
            ],
        )

        hits: list[dict[str, Any]] = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            hits.append(
                {
                    "ticket_id": entity.get("ticket_id"),
                    "category": entity.get("category"),
                    "priority": entity.get("priority"),
                    "initial_message": entity.get("initial_message"),
                    "resolution_summary": entity.get("resolution_summary"),
                    "user_status": entity.get("user_status"),
                    "sla": entity.get("sla"),
                    "score": hit.get("distance"),
                }
            )
        return hits

    def count(self) -> int:
        if not self.client.has_collection(self.collection):
            return 0
        stats = self.client.get_collection_stats(self.collection)
        return int(stats.get("row_count", 0))
