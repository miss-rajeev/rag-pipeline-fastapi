# app/embedding.py

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text: str):
    return embedding_model.encode(text).tolist()

def embed_batch(texts):
    return embedding_model.encode(texts).tolist()
