# app/utils.py

import re
from typing import List

# ----------------------------------------
# Extract text from PDF (stub for now)
# ----------------------------------------
def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract raw text from a PDF file.
    Later we will implement using PyPDF2 or pdfminer.
    """
    return "PDF text extraction not implemented yet."


# ----------------------------------------
# Chunk text into smaller sections
# ----------------------------------------
def chunk_text(text: str, chunk_size: int = 800) -> List[str]:
    """
    Splits long text into smaller chunks for embedding.
    """
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks


# ----------------------------------------
# Normalize a user query
# ----------------------------------------
def normalize_query(query: str) -> str:
    """
    Lowercase, trim spaces, remove weird characters.
    """
    query = query.lower().strip()
    query = re.sub(r"[^a-zA-Z0-9\s\?\.]", "", query)
    return query


# ----------------------------------------
# Detect intent (greeting vs info query)
# ----------------------------------------
def detect_intent(query: str) -> str:
    """
    Simple rule-based intent detection.
    """
    greetings = {"hi", "hello", "hey", "good morning", "good evening", "whats up", "yo"}

    if query.lower().strip() in greetings:
        return "greeting"

    return "information_query"
