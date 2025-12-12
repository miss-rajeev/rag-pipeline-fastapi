from pydantic import BaseModel
from search.semantic_search import hybrid_search
from generation.llm import generate_answer
from app.utils import detect_intent, normalize_query
from app.embedding import embed_text


@app.post("/ingest")
async def ingest_pdfs(files: List[UploadFile] = File(...)):
    all_chunks = []

    for file in files:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Extract + chunk
        chunks = process_pdf(tmp_path)
        all_chunks.extend(chunks)

    # Embed and store in vector DB
    vectors = embed_text_batch(all_chunks)
    for vec, chunk in zip(vectors, all_chunks):
        vector_store.add(vec, {"text": chunk})

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

    # 4️⃣ Hybrid search (semantic + keyword fusion)
    results = hybrid_search(clean_question, query_vec, top_k=5)

    # 5️⃣ Prepare context for LLM
    context_text = ""
    for score, chunk, sem, key in results:
        context_text += f"- {chunk}\n"

    # 6️⃣ Generate final grounded answer
    answer = generate_answer(context_text, clean_question)

    return {
        "query": clean_question,
        "intent": intent,
        "sources": [chunk for _, chunk, _, _ in results],
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