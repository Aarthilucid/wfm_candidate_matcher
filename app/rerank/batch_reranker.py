from __future__ import annotations

import json
from typing import Any

from app.config import Settings
from app.llm.openai_client import make_client

SYSTEM = """You are a strict hiring assistant.
You must ONLY use the provided evidence snippets for each candidate.

Rules:
- Do NOT infer missing info. If not in evidence, mark missing.
- Any must-have hit MUST be supported by citations (chunk_id).
- Return valid JSON only. No extra text.

Scoring (STRICT):
- Start at 100.
- Subtract 15 for each missing must-have.
- If ANY must-have is missing, overall_score must be <= 89.
- If TWO or more must-haves are missing, overall_score must be <= 79.
- Do NOT give 100 unless ALL must-haves are explicitly proven in evidence.
"""

def batch_score_candidates(
    *,
    settings: Settings,
    job_text: str,
    must_haves: list[str],
    nice_to_haves: list[str],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    client = make_client(settings)

    payload = {
        "job": {
            "text": job_text,
            "must_haves": must_haves,
            "nice_to_haves": nice_to_haves,
        },
        "candidates": candidates,
        "required_output": {
            "results": [
                {
                    "candidate_id": "string",
                    "overall_score": "int 0-100",
                    "must_have_hits": "string[]",
                    "missing_must_haves": "string[]",
                    "highlights": "string[]",
                    "risks": "string[]",
                    "citations": [{"chunk_id": "string", "excerpt": "string"}],
                }
            ]
        },
    }

    resp = client.chat.completions.create(
        model=settings.openai_reason_model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(payload)},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    text = resp.choices[0].message.content or "{}"
    data = json.loads(text)

    results = data.get("results", []) or []

    # Basic safety/shape normalization
    cleaned: list[dict[str, Any]] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        cid = r.get("candidate_id")
        if not cid:
            continue
        r.setdefault("overall_score", 0)
        r.setdefault("must_have_hits", [])
        r.setdefault("missing_must_haves", must_haves or [])
        r.setdefault("highlights", [])
        r.setdefault("risks", [])
        r.setdefault("citations", [])
        cleaned.append(r)

    return cleaned
