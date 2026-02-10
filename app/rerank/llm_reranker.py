from __future__ import annotations

import json
from typing import Any

from app.config import Settings
from app.llm.openai_client import make_client

SYSTEM = """You are a strict hiring assistant.
You must ONLY use the provided evidence snippets from the candidate's resume.

Rules:
- Do NOT infer or guess missing info (degrees, years, visa, skills). If not in evidence, mark as 'not found'.
- For any claim about skills/experience/requirements, cite at least one provided snippet.
- Return valid JSON only. No extra text.

Scoring rules (STRICT – must follow):
- Start from 100.
- For each missing must-have skill: subtract 15 points.
- If ANY must-have is missing, overall_score must be <= 89.
- If TWO or more must-haves are missing, overall_score must be <= 79.
- Do NOT give 100 unless ALL must-haves are explicitly proven in evidence.
- Weak or implied skills (not explicit) do NOT count as hits.

Output JSON keys:
- overall_score (int 0-100)
- must_have_hits (string[])
- missing_must_haves (string[])
- highlights (string[])
- risks (string[])
- citations: array of {chunk_id, excerpt}
"""


def score_candidate(
    *,
    settings: Settings,
    job_text: str,
    must_haves: list[str],
    nice_to_haves: list[str],
    candidate: dict[str, Any],
    evidence_snippets: list[dict[str, Any]],
) -> dict[str, Any]:
    client = make_client(settings)

    evidence = [
        {"chunk_id": s["chunk_id"], "excerpt": (s.get("text") or "")[:600]}
        for s in (evidence_snippets or [])
        if isinstance(s, dict) and s.get("chunk_id")
    ]

    payload = {
        "job": {"text": job_text, "must_haves": must_haves, "nice_to_haves": nice_to_haves},
        "candidate": {
            "candidate_id": candidate.get("candidate_id"),
            "name": candidate.get("name"),
            "years_experience": candidate.get("years_experience"),
            "location": candidate.get("location"),
            "work_authorization": candidate.get("work_authorization"),
            "top_skills": candidate.get("top_skills"),
        },
        "evidence_snippets": evidence,
        "output_schema": {
            "overall_score": "int 0-100",
            "must_have_hits": "string[]",
            "missing_must_haves": "string[]",
            "highlights": "string[]",
            "risks": "string[]",
            "citations": [{"chunk_id": "string", "excerpt": "string"}],
        },
    }

    # JSON mode (broadly compatible)
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

    # Parse robustly
    try:
        verdict = json.loads(text)
    except Exception:
        verdict = {}

    # Minimal defaults
    verdict.setdefault("overall_score", 0)
    verdict.setdefault("must_have_hits", [])
    verdict.setdefault("missing_must_haves", must_haves or [])
    verdict.setdefault("highlights", [])
    verdict.setdefault("risks", [])
    verdict.setdefault("citations", [])

    # Guard: ensure citations exist if evidence was provided
    if (not verdict.get("citations")) and evidence:
        verdict["citations"] = [
            {"chunk_id": evidence[0]["chunk_id"], "excerpt": evidence[0]["excerpt"]}
        ]

    # Enforce score caps based on missing must-haves (extra safety)
    missing = verdict.get("missing_must_haves") or []
    try:
        score = int(verdict.get("overall_score", 0))
    except Exception:
        score = 0

    if len(missing) >= 2:
        score = min(score, 79)
    elif len(missing) >= 1:
        score = min(score, 89)

    verdict["overall_score"] = max(0, min(100, score))

    return verdict
