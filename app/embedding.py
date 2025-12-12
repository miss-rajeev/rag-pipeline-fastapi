import os
from sentence_transformers import SentenceTransformer

# Load model once for whole app
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text: str):
    """
    Takes a string and returns a vector embedding.
    """
    return embedding_model.encode(text).tolist()


def embed_batch(texts):
    """
    Takes a list of strings and returns embeddings.
    """
    return embedding_model.encode(texts).tolist()
