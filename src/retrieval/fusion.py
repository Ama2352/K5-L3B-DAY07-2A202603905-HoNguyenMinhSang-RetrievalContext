"""Reciprocal rank fusion of independent retrieval rankings."""

def rrf(dense, lexical, limit=3, k=60):
    scores, records = {}, {}
    for ranking in (dense, lexical):
        for rank, chunk in enumerate(ranking, 1):
            key = chunk['id']
            scores[key] = scores.get(key, 0.0) + 1 / (k + rank)
            records[key] = chunk
    return [{**records[key], 'score': scores[key]} for key in
            sorted(scores, key=lambda key: (-scores[key], key))[:limit]]
