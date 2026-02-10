# Prompt strategy (grounded + audit-friendly)

## Principles
1) Evidence-first: the model can only claim what is supported by provided resume excerpts.
2) No hallucinated qualifications: if something isn't in evidence, mark it as "not found".
3) Transparent scoring: return rubric-based scores so hiring teams can understand ranking.

## Rerank / scoring prompt (used in `/match`)
We provide:
- Job description (raw text or structured)
- A candidate summary + several resume evidence snippets (each snippet has a chunk_id)

The model must return strict JSON:
- overall_score (0-100)
- must_have_hits (array)
- missing_must_haves (array)
- highlights (array)
- risks (array)
- citations: [{chunk_id, excerpt}]

We force:
- citations must reference the provided snippets only
- if evidence is insufficient, reduce score and list missing items as "not found"
