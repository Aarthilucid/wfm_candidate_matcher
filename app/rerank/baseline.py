from __future__ import annotations

import re
from typing import Any


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def baseline_score(
    *,
    job_text: str,
    must_haves: list[str],
    nice_to_haves: list[str],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    text = " ".join(
        [
            str(candidate.get("top_skills") or ""),
            str(candidate.get("resume_text") or ""),
            str(candidate.get("summary") or ""),
        ]
    )
    text_n = _norm(text)

    must_hits = []
    missing = []
    for s in must_haves:
        if _norm(s) and _norm(s) in text_n:
            must_hits.append(s)
        else:
            missing.append(s)

    nice_hits = []
    for s in nice_to_haves:
        if _norm(s) and _norm(s) in text_n:
            nice_hits.append(s)

    # scoring: must-have dominates
    score = 0
    if must_haves:
        score += int(70 * (len(must_hits) / max(1, len(must_haves))))
    else:
        score += 50

    score += min(20, 5 * len(nice_hits))

    yrs = float(candidate.get("years_experience") or 0)
    score += min(10, int(yrs))

    score = max(0, min(100, score))

    highlights = []
    if yrs:
        highlights.append(f"{int(yrs)} years of experience")
    if must_hits:
        highlights.append(f"Must-have matches: {', '.join(must_hits[:6])}")
    if nice_hits:
        highlights.append(f"Nice-to-have matches: {', '.join(nice_hits[:6])}")

    risks = []
    if missing:
        risks.append(f"Missing must-haves: {', '.join(missing[:6])}")

    return {
        "overall_score": score,
        "must_have_hits": must_hits,
        "missing_must_haves": missing,
        "highlights": highlights,
        "risks": risks,
        "citations": [],
    }
