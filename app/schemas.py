from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Optional

class IngestRequest(BaseModel):
    candidates_path: str = Field(..., description="Path to candidates CSV (must include resume_text)")
    jobs_path: str = Field(..., description="Path to jobs CSV")

class MatchFilters(BaseModel):
    location: Optional[str] = None
    work_authorization: Optional[str] = None
    min_years_experience: Optional[int] = None

class MatchRequest(BaseModel):
    job_id: Optional[str] = None
    job_description: Optional[str] = None
    top_k: int = 10
    retrieve_k: int = 120
    filters: MatchFilters = MatchFilters()

class Citation(BaseModel):
    chunk_id: str
    excerpt: str

class CandidateMatch(BaseModel):
    candidate_id: str
    name: Optional[str] = None
    overall_score: int
    must_have_hits: list[str] = []
    missing_must_haves: list[str] = []
    highlights: list[str] = []
    risks: list[str] = []
    citations: list[Citation] = []

class MatchResponse(BaseModel):
    job_id: Optional[str] = None
    results: list[CandidateMatch]

class ExplainRequest(BaseModel):
    job_id: str
    candidate_id: str

class ExplainResponse(BaseModel):
    job_id: str
    candidate_id: str
    explanation: dict[str, Any]
