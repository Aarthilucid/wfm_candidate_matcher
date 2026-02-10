from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import joblib
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class KeywordIndex:
    vectorizer: TfidfVectorizer
    matrix: any
    candidate_ids: list[str]

def build_keyword_index(candidate_ids: list[str], texts: list[str], *, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=200_000)
    matrix = vectorizer.fit_transform(texts)
    joblib.dump(KeywordIndex(vectorizer=vectorizer, matrix=matrix, candidate_ids=candidate_ids), out_path)

def load_keyword_index(path: str) -> KeywordIndex:
    return joblib.load(path)

def keyword_search(index: KeywordIndex, query: str, top_k: int = 200) -> list[tuple[str, float]]:
    qv = index.vectorizer.transform([query])
    sims = cosine_similarity(qv, index.matrix).ravel()
    # get top_k indices
    if top_k >= len(sims):
        idxs = sims.argsort()[::-1]
    else:
        idxs = sims.argpartition(-top_k)[-top_k:]
        idxs = idxs[np.argsort(sims[idxs])[::-1]]
    return [(index.candidate_ids[i], float(sims[i])) for i in idxs if sims[i] > 0]
