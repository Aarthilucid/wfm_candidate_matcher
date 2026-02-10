from __future__ import annotations
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_embed_model: str = "text-embedding-3-large"
    openai_reason_model: str = "gpt-5.2"

    chroma_dir: str = "storage/chroma"
    tfidf_path: str = "storage/tfidf.joblib"

    max_chunk_chars: int = 600
    chunk_overlap_chars: int = 120

def get_settings() -> Settings:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required. Set it in your environment or .env file.")
    return Settings(
        openai_api_key=key,
        openai_embed_model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large"),
        openai_reason_model=os.getenv("OPENAI_REASON_MODEL", "gpt-5.2"),
        chroma_dir=os.getenv("CHROMA_DIR", "storage/chroma"),
        tfidf_path=os.getenv("TFIDF_PATH", "storage/tfidf.joblib"),
    )
