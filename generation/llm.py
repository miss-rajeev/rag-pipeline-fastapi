import os
from mistralai.client import MistralClient
from dotenv import load_dotenv

# Load API key
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
API_KEY = os.getenv("MISTRAL_API_KEY")

client = MistralClient(api_key=API_KEY)
LLM_MODEL = "mistral-small-latest"


PROMPT_TEMPLATE = """
You are an AI assistant answering based strictly on the provided context.

If the answer is not in the context, say: "Insufficient evidence."

Context:
{context}

User question:
{question}

Answer the question clearly and concisely:
"""


def generate_answer(context: str, question: str):
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    # THE CORRECT CALL
    response = client.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # NEW client uses `message.content`
    return response.choices[0].message.content
