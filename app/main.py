from __future__ import annotations

import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from app.config import get_settings
from app.schemas import (
    IngestRequest,
    MatchRequest,
    MatchResponse,
    CandidateMatch,
    Citation,
    ExplainRequest,
    ExplainResponse,
)
from app.store import set_store, DataStore, get_store
from app.llm.embeddings import embed_query
from app.rag.retriever import retrieve
from app.search.keyword import load_keyword_index, keyword_search
from app.rerank.batch_reranker import batch_score_candidates

load_dotenv()

app = FastAPI(title="WFM Candidate Matcher", version="0.2.0")

# UI
app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")


def load_dataframes(candidates_path: str, jobs_path: str) -> DataStore:
    cand_df = pd.read_csv(candidates_path)
    jobs_df = pd.read_csv(jobs_path)
    return DataStore(candidates=cand_df, jobs=jobs_df)


def ensure_loaded_from_storage() -> None:
    # Optional auto-load if storage parquet exists
    if os.path.exists("storage/candidates.parquet") and os.path.exists("storage/jobs.parquet"):
        store = DataStore(
            candidates=pd.read_parquet("storage/candidates.parquet"),
            jobs=pd.read_parquet("storage/jobs.parquet"),
        )
        set_store(store)


@app.on_event("startup")
def _startup() -> None:
    ensure_loaded_from_storage()


@app.post("/ingest")
def ingest(req: IngestRequest):
    # This endpoint just loads CSVs into memory for API use.
    # The heavy index build is done via `python -m app.scripts.ingest`.
    if not os.path.exists(req.candidates_path):
        raise HTTPException(status_code=400, detail=f"candidates_path not found: {req.candidates_path}")
    if not os.path.exists(req.jobs_path):
        raise HTTPException(status_code=400, detail=f"jobs_path not found: {req.jobs_path}")
    store = load_dataframes(req.candidates_path, req.jobs_path)
    set_store(store)
    return {"ok": True, "candidates": len(store.candidates), "jobs": len(store.jobs)}


@app.post("/match", response_model=MatchResponse)
def match(req: MatchRequest):
    settings = get_settings()
    store = get_store()

    if not req.job_id and not req.job_description:
        raise HTTPException(status_code=400, detail="Provide job_id or job_description")

    # ---- Load job text + skills ----
    if req.job_id:
        job_row = store.jobs[store.jobs["job_id"] == req.job_id]
        if job_row.empty:
            raise HTTPException(status_code=404, detail=f"job_id not found: {req.job_id}")
        job = job_row.iloc[0].to_dict()

        job_text = f"{job.get('title','')}\n\n{job.get('description','')}"
        must_haves = [x.strip() for x in str(job.get("must_have_skills", "") or "").split(";") if x.strip()]
        nice = [x.strip() for x in str(job.get("nice_to_have_skills", "") or "").split(";") if x.strip()]
        min_years = int(job.get("min_years_experience", 0) or 0)
    else:
        job_text = req.job_description or ""
        must_haves, nice = [], []
        min_years = req.filters.min_years_experience or 0

    # ---- Hard filters ----
    cand_df = store.candidates.copy()

    if req.filters.location:
        cand_df = cand_df[cand_df["location"].fillna("").str.contains(req.filters.location, case=False, na=False)]

    if req.filters.work_authorization:
        cand_df = cand_df[
            cand_df["work_authorization"].fillna("").str.contains(req.filters.work_authorization, case=False, na=False)
        ]

    if req.filters.min_years_experience is not None:
        cand_df = cand_df[cand_df["years_experience"].fillna(0) >= req.filters.min_years_experience]
    elif min_years:
        cand_df = cand_df[cand_df["years_experience"].fillna(0) >= min_years]

    allowed_ids = set(cand_df["candidate_id"].astype(str).tolist())

    # ---- Hybrid retrieval ----
    # 1) Keyword candidates
    kw_index = load_keyword_index(settings.tfidf_path)
    kw_hits = keyword_search(kw_index, job_text, top_k=req.retrieve_k)
    kw_ids = [cid for cid, _ in kw_hits if cid in allowed_ids]

    # 2) Vector retrieval (semantic). Fail-soft.
    vec_chunks: list[dict] = []
    vec_ids: list[str] = []
    try:
        q_emb = embed_query(job_text, settings=settings)
        vec_chunks = retrieve(
            chroma_dir=settings.chroma_dir,
            query_embedding=q_emb,
            top_k=min(req.retrieve_k, 200),
            where=None,
        )
        for ch in vec_chunks:
            cid = str(ch["metadata"].get("candidate_id"))
            if cid in allowed_ids:
                vec_ids.append(cid)
    except Exception:
        vec_chunks = []
        vec_ids = []

    # Merge candidate ids (keyword first, then vector)
    merged: list[str] = []
    seen: set[str] = set()
    for cid in kw_ids + vec_ids:
        if cid not in seen:
            merged.append(cid)
            seen.add(cid)
        if len(merged) >= req.retrieve_k:
            break

    if not merged:
        return MatchResponse(job_id=req.job_id, results=[])

    # Evidence snippets per candidate from vector chunks
    chunks_by_cand: dict[str, list[dict]] = {}
    for ch in vec_chunks:
        cid = str(ch["metadata"].get("candidate_id"))
        if cid in seen:
            chunks_by_cand.setdefault(cid, []).append(ch)

    # ---- Batch LLM scoring (FAST) ----
    max_llm = min(10, len(merged))  # adjust to 10 for even faster

    batch_in: list[dict] = []
    for cid in merged[:max_llm]:
        row = store.candidates[store.candidates["candidate_id"].astype(str) == cid]
        if row.empty:
            continue
        cand = row.iloc[0].to_dict()
        evidence = chunks_by_cand.get(cid, [])[:6]

        batch_in.append(
            {
                "candidate_id": cid,
                "name": cand.get("name"),
                "years_experience": cand.get("years_experience"),
                "location": cand.get("location"),
                "work_authorization": cand.get("work_authorization"),
                "top_skills": cand.get("top_skills"),
                "evidence_snippets": [
                    {"chunk_id": e.get("chunk_id"), "excerpt": (e.get("text") or "")[:600]}
                    for e in evidence
                    if isinstance(e, dict)
                ],
            }
        )

    verdicts: list[dict] = []
    try:
        verdicts = batch_score_candidates(
            settings=settings,
            job_text=job_text,
            must_haves=must_haves,
            nice_to_haves=nice,
            candidates=batch_in,
        )
    except Exception as e:
        # If batch scoring fails, return empty (UI will show nothing). Better than slow per-candidate.
        # If you want, we can fallback to keyword-only scores here.
        raise HTTPException(status_code=500, detail=f"Batch scoring failed: {type(e).__name__}: {str(e)[:300]}")

    vmap = {str(v.get("candidate_id")): v for v in verdicts if v.get("candidate_id")}

    scored: list[CandidateMatch] = []
    for item in batch_in:
        cid = item["candidate_id"]
        row = store.candidates[store.candidates["candidate_id"].astype(str) == cid]
        if row.empty:
            continue
        cand = row.iloc[0].to_dict()

        v = vmap.get(cid, {})

        citations = v.get("citations", []) or []
        # Guard: if citations missing but evidence exists, add first evidence
        if (not citations) and item.get("evidence_snippets"):
            citations = [
                {
                    "chunk_id": item["evidence_snippets"][0].get("chunk_id"),
                    "excerpt": item["evidence_snippets"][0].get("excerpt"),
                }
            ]

        scored.append(
            CandidateMatch(
                candidate_id=cid,
                name=cand.get("name"),
                overall_score=int(v.get("overall_score", 0) or 0),
                must_have_hits=v.get("must_have_hits", []) or [],
                missing_must_haves=v.get("missing_must_haves", []) or [],
                highlights=v.get("highlights", []) or [],
                risks=v.get("risks", []) or [],
                citations=[Citation(**c) for c in citations if isinstance(c, dict)],
            )
        )

    scored.sort(key=lambda x: x.overall_score, reverse=True)
    return MatchResponse(job_id=req.job_id, results=scored[: req.top_k])


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    settings = get_settings()
    store = get_store()

    job_row = store.jobs[store.jobs["job_id"] == req.job_id]
    if job_row.empty:
        raise HTTPException(status_code=404, detail=f"job_id not found: {req.job_id}")
    job = job_row.iloc[0].to_dict()

    job_text = f"{job.get('title','')}\n\n{job.get('description','')}"
    must_haves = [x.strip() for x in str(job.get("must_have_skills", "") or "").split(";") if x.strip()]
    nice = [x.strip() for x in str(job.get("nice_to_have_skills", "") or "").split(";") if x.strip()]

    cand_row = store.candidates[store.candidates["candidate_id"].astype(str) == req.candidate_id]
    if cand_row.empty:
        raise HTTPException(status_code=404, detail=f"candidate_id not found: {req.candidate_id}")
    cand = cand_row.iloc[0].to_dict()

    # Pull evidence chunks for this candidate
    vec_chunks: list[dict] = []
    try:
        q_emb = embed_query(job_text, settings=settings)
        vec_chunks = retrieve(
            chroma_dir=settings.chroma_dir,
            query_embedding=q_emb,
            top_k=30,
            where={"candidate_id": str(req.candidate_id)},
        )
    except Exception:
        vec_chunks = []

    # Use the batch scorer with a single candidate for consistent behavior
    one = [
        {
            "candidate_id": str(req.candidate_id),
            "name": cand.get("name"),
            "years_experience": cand.get("years_experience"),
            "location": cand.get("location"),
            "work_authorization": cand.get("work_authorization"),
            "top_skills": cand.get("top_skills"),
            "evidence_snippets": [
                {"chunk_id": e.get("chunk_id"), "excerpt": (e.get("text") or "")[:600]}
                for e in (vec_chunks or [])[:10]
                if isinstance(e, dict)
            ],
        }
    ]

    verdicts = batch_score_candidates(
        settings=settings,
        job_text=job_text,
        must_haves=must_haves,
        nice_to_haves=nice,
        candidates=one,
    )
    verdict = verdicts[0] if verdicts else {}

    # Guard citations
    if (not verdict.get("citations")) and one[0].get("evidence_snippets"):
        verdict["citations"] = [
            {
                "chunk_id": one[0]["evidence_snippets"][0].get("chunk_id"),
                "excerpt": one[0]["evidence_snippets"][0].get("excerpt"),
            }
        ]

    return ExplainResponse(job_id=req.job_id, candidate_id=req.candidate_id, explanation=verdict)
