from __future__ import annotations

from openai import OpenAI
from app.config import Settings


def make_client(settings: Settings) -> OpenAI:
    # OpenAI python SDK supports timeout + max_retries
    return OpenAI(
        api_key=settings.openai_api_key,
        timeout=getattr(settings, "openai_timeout_seconds", 30),
        max_retries=getattr(settings, "openai_max_retries", 2),
    )
