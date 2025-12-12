# app/vectors.py
"""
Handles embedding generation and vector storage for semantic search.
This version uses an in-memory vector store (list of dicts).
"""

from typing import List, Dict
import numpy as np


# -------------------------
# Simple in-memory vector DB
# -------------------------
VECTOR_STORE: List[Dict] = []


def add_embedding(text: str, embedding: List[float]):
    """
    Save a text chunk + embedding into the vector store.
    """
    VECTOR_STORE.append({
        "text": text,
        "embedding": np.array(embedding, dtype=float)
    })


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def semantic_search(query_emb: List[float], top_k: int = 5):
    """
    Returns top K chunks ranked by cosine similarity.
    """
    query_vec = np.array(query_emb, dtype=float)

    scored = []
    for item in VECTOR_STORE:
        score = compute_cosine_similarity(query_vec, item["embedding"])
        scored.append({"text": item["text"], "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def keyword_search(query: str, top_k: int = 5):
    """
    Very simple keyword search to support hybrid search.
    """
    results = []
    for item in VECTOR_STORE:
        if query.lower() in item["text"].lower():
            results.append({"text": item["text"], "score": 1.0})

    return results[:top_k]
