from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from app.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )
