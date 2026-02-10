# WFM Candidate Matcher (RAG + Rerank + Evidence)

A production-leaning candidate matching service you can plug into a workforce management website.

## What it does
- Ingests candidates and jobs from CSV/JSON
- Chunks resumes, embeds them, and stores them in a persistent Chroma vector DB
- Builds a lightweight keyword index (TF‑IDF) for hybrid retrieval
- Matches a job description to the best candidates with:
  - ranked results
  - score breakdown
  - evidence snippets ("citations") from the resume text

## Data included (synthetic)
- `data/candidates.csv` (250 candidates, includes `resume_text`)
- `data/jobs.csv` (10 jobs)
- `data/applications.csv` (optional)

## Quickstart (local)
1) Create venv and install deps:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

2) Set env vars:
```bash
cp .env.example .env
# Put your OPENAI_API_KEY in .env
```

3) Run ingestion (build indexes):
```bash
python -m app.scripts.ingest --candidates data/candidates.csv --jobs data/jobs.csv
```

4) Start API:
```bash
uvicorn app.main:app --reload
```

5) Open UI:
- http://localhost:8000/ui/

## Docker
```bash
docker build -t wfm-matcher .
docker run --rm -p 8000:8000 --env-file .env wfm-matcher
```

## API
### POST /match
Input:
```json
{
  "job_id": "JOB-GENAI-001",
  "top_k": 10,
  "filters": { "location": "Remote", "min_years_experience": 3 }
}
```

Or:
```json
{
  "job_description": "Paste JD text here...",
  "top_k": 10
}
```

### POST /explain
Input:
```json
{ "job_id": "JOB-GENAI-001", "candidate_id": "CAND-1001" }
```

## Notes for scaling to millions
- Swap Chroma for Pinecone/Milvus/Weaviate
- Swap TF‑IDF for OpenSearch/Elastic (BM25 + filters)
- Keep the rerank stage limited to top ~200 candidates to control cost/latency
