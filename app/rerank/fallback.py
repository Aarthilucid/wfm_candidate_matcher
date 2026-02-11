from __future__ import annotations

from typing import Any


def fallback_score(
    *,
    must_haves: list[str],
    candidate: dict[str, Any],
    evidence_snippets: list[dict[str, Any]],
) -> dict[str, Any]:
    # lightweight: score based on must-have occurrences in candidate top_skills + evidence text
    top = (candidate.get("top_skills") or "")
    blob = " ".join([(s.get("text") or "") for s in evidence_snippets]) + " " + str(top)
    blob_lower = blob.lower()

    hits: list[str] = []
    missing: list[str] = []
    for mh in must_haves:
        if mh and mh.lower() in blob_lower:
            hits.append(mh)
        else:
            missing.append(mh)

    # simple score: 60 base + must-have hit ratio
    score = 60
    if must_haves:
        score = int(60 + 40 * (len(hits) / max(1, len(must_haves))))

    citations = []
    if evidence_snippets:
        e0 = evidence_snippets[0]
        citations = [{"chunk_id": e0.get("chunk_id"), "excerpt": (e0.get("text") or "")[:220]}]

    return {
        "overall_score": max(0, min(100, score)),
        "must_have_hits": hits,
        "missing_must_haves": missing,
        "highlights": ["Fallback scoring (LLM unavailable)."],
        "risks": ["Used heuristic scoring; verify with recruiter review."],
        "citations": citations,
    }
