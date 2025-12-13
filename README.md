Retrieval-Augmented Generation (RAG) System

This repository contains a simple but production-oriented Retrieval-Augmented Generation (RAG) system built using FastAPI. The system ingests PDF documents, indexes them using vector embeddings, and answers user questions by retrieving relevant document chunks and generating grounded responses using a language model.

The design emphasizes correctness, transparency, and safety over raw fluency.

System Overview

At a high level, the system has two main workflows:

Document ingestion: PDFs are parsed, cleaned, chunked, embedded, and stored.

Query answering: User questions are validated, retrieved against the indexed documents, and answered using only retrieved evidence.

The system explicitly refuses to answer when there is insufficient evidence or when a query violates safety policies.
User
 │
 ▼
Frontend (HTML UI)
 │
 ▼
FastAPI Application
 │
 ├── Query validation & policy checks
 ├── Intent detection and normalization
 ├── Embedding generation
 ├── Hybrid retrieval (semantic + keyword)
 ├── Evidence validation
 ├── LLM answer generation
 ├── Hallucination detection
 └── Response formatting
 │
 ▼
Response
Document Ingestion Flow

User uploads one or more PDF files.

Each PDF is:

Parsed to extract raw text

Cleaned to remove noise

Split into semantically meaningful chunks

Each chunk is embedded using an embedding model.

Chunks and embeddings are stored in the vector store.

This process is handled by the /ingest endpoint.
Query Processing Flow

When a user submits a question, the system executes the following steps:

1. Safety and Policy Checks

The query is first checked for restricted categories such as:

Personally identifiable information (PII)

Medical advice

Legal advice

If a violation is detected, the system refuses to answer.

2. Intent Detection and Normalization

The query is normalized (lowercasing, cleanup) and classified by intent (e.g., greeting, informational query, list-style request).
Intent is later used to shape the final response format.

3. Query Embedding

The normalized query is converted into a vector embedding using the same embedding model used during ingestion.

4. Hybrid Retrieval

The system performs hybrid search, combining:

Semantic similarity (vector distance)

Keyword matching

The top-K most relevant chunks are returned, along with retrieval scores.

5. Evidence Threshold Enforcement

If none of the retrieved chunks exceed a predefined similarity threshold, the system returns:

Insufficient evidence to answer this question.


This prevents the language model from guessing when the document does not support an answer.

6. Answer Generation

The language model generates an answer using only the retrieved document chunks as context.
No external knowledge is injected at this stage.

7. Hallucination Detection

After generation, the answer is checked against the retrieved evidence.
If the system cannot verify that the answer is supported by the retrieved text, the response is rejected.

8. Response Formatting

The answer is shaped based on detected intent (e.g., paragraph vs. list) and returned to the frontend in a format compatible with the demo UI.

Design Principles

Grounding over fluency: The system prefers refusing to answer rather than producing unsupported output.

Explicit failure modes: Insufficient evidence and policy violations are clearly surfaced to the user.

Separation of concerns: Ingestion, retrieval, generation, and safety checks are isolated and composable.

Enterprise-oriented behavior: The system mirrors how real-world RAG systems behave under uncertainty.
Example Queries

“What is this PDF about?”

“What capabilities are described in the document?”

“How does this system use RAG pipelines?”

“List the enterprise features mentioned.”

Limitations and Future Work

Per-sentence citations can be added for finer-grained attribution.

A formal evaluation harness can be introduced to measure retrieval and answer quality.

The hallucination check can be upgraded to embedding-based verification.

Multi-document reasoning and cross-document citations can be supported.

Summary

This repository demonstrates a complete RAG system with clear system boundaries, safety checks, and evidence-based answering. It is intentionally designed to behave conservatively and transparently, reflecting real-world requirements for deploying language models on private or enterprise data.
