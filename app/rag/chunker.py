from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str

def chunk_text(text: str, *, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    # Simple character-based chunking with overlap.
    t = (text or "").strip()
    if not t:
        return []
    chunks: list[str] = []
    start = 0
    n = len(t)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = t[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks
