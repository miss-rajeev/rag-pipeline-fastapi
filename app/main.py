from pydantic import BaseModel
from search.semantic_search import hybrid_search
from generation.llm import generate_answer
from app.utils import detect_intent, normalize_query
from app.embedding import embed_text, embed_batch
from fastapi import FastAPI, UploadFile, File
from app.vectors import VECTOR_STORE, add_embedding
from processing.pdf_utils import extract_text_from_pdf, clean_text, chunk_text
import tempfile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi import Request
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="app/templates")

app = FastAPI()
@app.get("/", response_class=HTMLResponse)
def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})






@app.post("/ingest")
async def ingest_pdfs(files: list[UploadFile] = File(...)):
    all_chunks = []

    for file in files:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Run ingestion pipeline
        raw = extract_text_from_pdf(tmp_path)
        cleaned = clean_text(raw)
        chunks = chunk_text(cleaned)
        all_chunks.extend(chunks)

    # Embed and store
    vectors = embed_batch(all_chunks)
    for vec, chunk in zip(vectors, all_chunks):
        add_embedding(chunk, vec)

    return {
        "message": "Ingestion completed",
        "chunks_stored": len(all_chunks)
    }


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
async def query_rag(request: QueryRequest):
    question = request.question

    # 1️⃣ Intent detection
    intent = detect_intent(question)
    if intent == "greeting":
        return {"answer": "Hello! How can I assist you today?"}

    # 2️⃣ Normalize query
    clean_question = normalize_query(question)

    # 3️⃣ Embed query
    query_vec = embed_text(clean_question)

    # 4️⃣ Hybrid search
    results = hybrid_search(clean_question, query_vec, top_k=5)

    # 5️⃣ Extract final chunks
    retrieved_chunks = [chunk for _, chunk, _, _ in results]

    # 6️⃣ Generate grounded LLM answer
    answer = generate_answer(clean_question, retrieved_chunks)

    # 7️⃣ Return structured output
    return {
        "query": clean_question,
        "intent": intent,
        "sources": retrieved_chunks,
        "scores": [
            {
                "final_score": score,
                "semantic_score": sem,
                "keyword_score": key
            }
            for score, chunk, sem, key in results
        ],
        "answer": answer
    }