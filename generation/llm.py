import os
from mistralai import Mistral
from typing import List

# Load API key
API_KEY = os.getenv("MISTRAL_API_KEY")

client = Mistral(api_key=API_KEY)


def build_prompt(question: str, retrieved_chunks: List[str]) -> str:
    """
    Build the final prompt sent to the LLM using retrieved context.
    """
    context_text = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are an AI assistant. Answer the question using ONLY the provided context. 
If the answer is not present in the context, say "I don't have enough information to answer that."

### Context:
{context_text}

### Question:
{question}

### Answer:
"""

    return prompt.strip()


def generate_answer(question: str, retrieved_chunks: List[str]) -> str:
    """
    Calls the Mistral model with our prompt template + context.
    """
    prompt = build_prompt(question, retrieved_chunks)

    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message["content"]
