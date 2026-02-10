from __future__ import annotations
from typing import Any
from app.rag.vector_store import query_chunks

def retrieve(
    *,
    chroma_dir: str,
    query_embedding: list[float],
    top_k: int = 40,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    res = query_chunks(
        persist_dir=chroma_dir,
        query_embedding=query_embedding,
        n_results=top_k,
        where=where,
    )
    # Chroma returns lists nested by query
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    out = []
    for i in range(len(ids)):
        out.append({
            "chunk_id": ids[i],
            "text": docs[i],
            "metadata": metas[i],
            "distance": float(dists[i]),
        })
    return out
