from __future__ import annotations
from typing import Iterable
from app.config import Settings
from app.llm.openai_client import make_client

def embed_texts(texts: list[str], *, settings: Settings) -> list[list[float]]:
    client = make_client(settings)
    # OpenAI embeddings endpoint
    resp = client.embeddings.create(
        model=settings.openai_embed_model,
        input=texts,
    )
    return [d.embedding for d in resp.data]

def embed_query(query: str, *, settings: Settings) -> list[float]:
    return embed_texts([query], settings=settings)[0]
