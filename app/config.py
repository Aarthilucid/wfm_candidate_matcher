from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or not v.strip():
        return default
    try:
        return int(v.strip())
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # Provider (for now we use OpenAI. Later we can add Gemini with same interface.)
    openai_api_key: str

    # Models
    openai_embed_model: str = "text-embedding-3-large"
    openai_reason_model: str = "gpt-5.2"

    # Storage
    chroma_dir: str = "storage/chroma"
    tfidf_path: str = "storage/tfidf.joblib"

    # Chunking
    max_chunk_chars: int = 600
    chunk_overlap_chars: int = 120

    # SaaS knobs: speed/cost
    enable_llm_scoring: bool = True
    max_llm_candidates: int = 10          # how many candidates get LLM scoring per request
    retrieve_k_cap: int = 200             # cap vector retrieval per request
    evidence_per_candidate: int = 6       # how many snippets per candidate to send to LLM

    # Caching (seconds)
    cache_ttl_seconds: int = 300
    cache_max_items: int = 2048


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
        max_chunk_chars=_env_int("MAX_CHUNK_CHARS", 600),
        chunk_overlap_chars=_env_int("CHUNK_OVERLAP_CHARS", 120),
        enable_llm_scoring=_env_bool("ENABLE_LLM_SCORING", True),
        max_llm_candidates=_env_int("MAX_LLM_CANDIDATES", 10),
        retrieve_k_cap=_env_int("RETRIEVE_K_CAP", 200),
        evidence_per_candidate=_env_int("EVIDENCE_PER_CANDIDATE", 6),
        cache_ttl_seconds=_env_int("CACHE_TTL_SECONDS", 300),
        cache_max_items=_env_int("CACHE_MAX_ITEMS", 2048),
    )
