# search/semantic_search.py

import numpy as np
from typing import List, Tuple
from app.vectors import VECTOR_STORE, compute_cosine_similarity


def hybrid_search(query: str, query_emb, top_k: int = 5) -> List[Tuple]:
    """
    Returns list of tuples:
    (final_score, chunk_text, semantic_score, keyword_score)
    """

    results = []

    for item in VECTOR_STORE:
        text = item["text"]
        emb = item["embedding"]

        sem = compute_cosine_similarity(query_emb, emb)
        key = 1.0 if query.lower() in text.lower() else 0.0

        final = (0.7 * sem) + (0.3 * key)

        results.append((final, text, sem, key))

    results.sort(key=lambda x: x[0], reverse=True)

    return results[:top_k]
