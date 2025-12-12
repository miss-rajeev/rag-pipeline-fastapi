from typing import List, Tuple
import numpy as np
from difflib import SequenceMatcher

# -----------------------------
# Cosine Similarity
# -----------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# -----------------------------
# Keyword Overlap Score
# -----------------------------
def keyword_score(query: str, text: str) -> float:
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())

    if not text_words:
        return 0.0

    overlap = query_words.intersection(text_words)
    return len(overlap) / len(query_words)  # normalized score

# -----------------------------
# Hybrid Search
# -----------------------------
def hybrid_rank(
    query: str,
    query_embedding: np.ndarray,
    stored_chunks: List[dict],
    top_k: int = 5
) -> List[Tuple[float, str, float, float]]:
    """
    Returns list of:
    (final_score, chunk_text, semantic_score, keyword_score)
    """

    results = []

    for entry in stored_chunks:
        chunk_text = entry["text"]
        chunk_embedding = entry["embedding"]

        sem_score = cosine_similarity(query_embedding, np.array(chunk_embedding))
        key_score = keyword_score(query, chunk_text)

        # Weighted combination
        final_score = (0.7 * sem_score) + (0.3 * key_score)

        results.append((final_score, chunk_text, sem_score, key_score))

    # Sort by final score desc
    results.sort(key=lambda x: x[0], reverse=True)

    return results[:top_k]
