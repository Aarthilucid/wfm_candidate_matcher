from __future__ import annotations
import argparse
import os
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from app.config import get_settings
from app.rag.chunker import chunk_text
from app.llm.embeddings import embed_texts
from app.rag.vector_store import upsert_chunks
from app.search.keyword import build_keyword_index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--jobs", required=True)
    args = parser.parse_args()

    settings = get_settings()

    cand_df = pd.read_csv(args.candidates)
    if "resume_text" not in cand_df.columns:
        raise SystemExit("candidates CSV must contain resume_text column")

    # Build keyword index over full resume text (per candidate)
    build_keyword_index(
        candidate_ids=cand_df["candidate_id"].tolist(),
        texts=cand_df["resume_text"].fillna("").tolist(),
        out_path=settings.tfidf_path,
    )

    # Chunk and embed for vector store
    ids = []
    docs = []
    metas = []

    for _, row in cand_df.iterrows():
        cid = str(row["candidate_id"])
        resume = str(row.get("resume_text") or "")
        chunks = chunk_text(
            resume,
            chunk_size=settings.max_chunk_chars,
            overlap=settings.chunk_overlap_chars,
        )
        for idx, ch in enumerate(chunks):
            chunk_id = f"{cid}::chunk_{idx:04d}"
            ids.append(chunk_id)
            docs.append(ch)
            metas.append({
                "candidate_id": cid,
                "section": "resume",
            })

    # Embed in batches
    embeddings = []
    batch_size = 96
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        embs = embed_texts(batch, settings=settings)
        embeddings.extend(embs)

    upsert_chunks(
        persist_dir=settings.chroma_dir,
        ids=ids,
        embeddings=embeddings,
        documents=docs,
        metadatas=metas,
    )

    # Save jobs copy (optional)
    jobs_df = pd.read_csv(args.jobs)
    os.makedirs("storage", exist_ok=True)
    cand_df.to_parquet("storage/candidates.parquet", index=False)
    jobs_df.to_parquet("storage/jobs.parquet", index=False)

    print(f"OK: indexed {len(cand_df)} candidates and {len(ids)} chunks")
    print(f"Chroma: {settings.chroma_dir}")
    print(f"TFIDF: {settings.tfidf_path}")

if __name__ == "__main__":
    main()


