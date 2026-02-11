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

Output format (MUST be exact):
{
  "results": [
    {
      "candidate_id": "...",
      "overall_score": 0-100,
      "must_have_hits": ["..."],
      "missing_must_haves": ["..."],
      "highlights": ["..."],
      "risks": ["..."],
      "citations": [{"chunk_id":"...", "excerpt":"..."}]
    }
  ]
}
"""

def _safe_json_loads(text: str) -> dict[str, Any]:
    """
    Robust JSON parsing:
    - If model returns extra text, try to extract the first JSON object.
    """
    text = (text or "").strip()
    if not text:
        return {}

    # Fast path
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    # Extract first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        chunk = text[start : end + 1]
        try:
            obj = json.loads(chunk)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    return {}

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
        "job": {"text": job_text, "must_haves": must_haves, "nice_to_haves": nice_to_haves},
        "candidates": candidates,
    }

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    # Try JSON mode first; if it fails, fall back to normal response parsing
    try:
        resp = client.chat.completions.create(
            model=settings.openai_reason_model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=1500,
        )
        text = resp.choices[0].message.content or "{}"
        data = _safe_json_loads(text)
    except Exception:
        resp = client.chat.completions.create(
            model=settings.openai_reason_model,
            messages=messages,
            temperature=0.2,
            max_tokens=1500,
        )
        text = resp.choices[0].message.content or "{}"
        data = _safe_json_loads(text)

    results = data.get("results", [])
    if not isinstance(results, list):
        results = []

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

        # Clamp score
        try:
            r["overall_score"] = int(r["overall_score"])
        except Exception:
            r["overall_score"] = 0
        r["overall_score"] = max(0, min(100, r["overall_score"]))

        cleaned.append(r)

    return cleaned
