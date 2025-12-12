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
