from __future__ import annotations
from typing import Any, Iterable
import os
import chromadb
from chromadb.config import Settings as ChromaSettings

COLLECTION_NAME = "candidate_resume_chunks"

def get_client(persist_dir: str) -> chromadb.ClientAPI:
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir, settings=ChromaSettings(anonymized_telemetry=False))

def get_collection(client: chromadb.ClientAPI):
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

def upsert_chunks(
    *,
    persist_dir: str,
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    client = get_client(persist_dir)
    col = get_collection(client)
    col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

def query_chunks(
    *,
    persist_dir: str,
    query_embedding: list[float],
    n_results: int,
    where: dict[str, Any] | None = None,
) -> dict[str, Any]:
    client = get_client(persist_dir)
    col = get_collection(client)
    return col.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
