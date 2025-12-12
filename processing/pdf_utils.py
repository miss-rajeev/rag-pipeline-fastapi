import re
from typing import List
from pypdf import PdfReader


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts raw text from a PDF file using PyPDF.
    """
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text.strip()


def clean_text(text: str) -> str:
    """
    Light cleaning: remove excessive whitespace.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """
    Splits text into chunks of ~500 characters.
    Very simple algorithm — perfect for an interview exercise.
    """
    words = text.split()
    chunks = []
    current = []

    current_len = 0

    for word in words:
        if current_len + len(word) + 1 > chunk_size:
            chunks.append(" ".join(current))
            current = []
            current_len = 0

        current.append(word)
        current_len += len(word) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks
