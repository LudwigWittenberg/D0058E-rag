# RAG App API Documentation

## Base URL

```
http://localhost:8002
```

---

## Endpoints

### GET /api/query

REST API endpoint for querying the RAG pipeline. Designed for inter-app communication (e.g., the Chatbot or Agents app calling the RAG app for context).

**Query Parameters:**

| Parameter    | Type   | Required | Default   | Description                              |
|-------------|--------|----------|-----------|------------------------------------------|
| `q`         | string | Yes      | —         | The search query                         |
| `top_k`     | int    | No       | 5         | Maximum number of chunks to retrieve     |
| `collection`| string | No       | "default" | ChromaDB collection to search            |

**Example Request:**

```
GET /api/query?q=What is retrieval augmented generation?&top_k=3
```

**Success Response (200):**

```json
{
    "answer": "Retrieval-Augmented Generation (RAG) is a technique that...",
    "context": [
        "RAG combines retrieval with generation to produce...",
        "The retrieval step finds relevant documents...",
        "Generation uses the retrieved context to..."
    ],
    "sources": [
        {
            "text": "RAG combines retrieval with generation to produce...",
            "metadata": {"source": "rag_paper.pdf", "chunk_index": 3},
            "score": 0.87
        },
        {
            "text": "The retrieval step finds relevant documents...",
            "metadata": {"source": "rag_paper.pdf", "chunk_index": 5},
            "score": 0.82
        },
        {
            "text": "Generation uses the retrieved context to...",
            "metadata": {"source": "rag_paper.pdf", "chunk_index": 7},
            "score": 0.76
        }
    ],
    "model": "llama3.1",
    "query": "What is retrieval augmented generation?"
}
```

**Error Response (400):**

```json
{
    "error": "Missing 'q' query parameter"
}
```

**Error Response (500):**

```json
{
    "error": "Description of what went wrong"
}
```

**Notes:**
- If the AI modules are not yet implemented, returns a placeholder response with `"model": "placeholder"`.
- The `context` field contains raw chunk texts; `sources` includes metadata and relevance scores.
- Scores range from 0.0 (no relevance) to 1.0 (exact match).

---

### POST /upload

Upload a document for processing through the RAG pipeline (chunking, embedding, and storage).

**Content-Type:** `multipart/form-data`

**Form Fields:**

| Field  | Type | Required | Description                          |
|--------|------|----------|--------------------------------------|
| `file` | file | Yes      | The document file (PDF or TXT)       |

**Allowed File Types:** `.pdf`, `.txt`

**Example Request (curl):**

```bash
curl -X POST http://localhost:8002/upload \
  -F "file=@document.pdf"
```

**Example Request (JavaScript):**

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('/upload', {
    method: 'POST',
    body: formData
});
const result = await response.json();
```

**Success Response (200):**

```json
{
    "status": "success",
    "filename": "document.pdf",
    "message": "Document processed: 12 chunks created and stored."
}
```

**Partial Response (200) — AI modules not implemented:**

```json
{
    "status": "partial",
    "filename": "document.pdf",
    "message": "File saved. AI processing not yet implemented (see ai/ modules)."
}
```

**Warning Response (200) — No text extracted:**

```json
{
    "status": "warning",
    "filename": "document.pdf",
    "message": "File uploaded but no text content could be extracted."
}
```

**Error Responses:**

| Status | Body | Condition |
|--------|------|-----------|
| 400 | `{"status": "error", "message": "No file provided"}` | No `file` field in request |
| 400 | `{"status": "error", "message": "No file selected"}` | Empty filename |
| 400 | `{"status": "error", "message": "File type not allowed..."}` | Unsupported file extension |
| 500 | `{"status": "error", "filename": "...", "message": "..."}` | Processing failure |

**Processing Pipeline:**
1. File is saved to the `data/` directory.
2. Text is extracted (plain read for `.txt`, PyPDF2 for `.pdf`).
3. Text is chunked using configured `chunk_size` and `chunk_overlap`.
4. Chunks are embedded using the configured embedding model.
5. Embeddings are stored in ChromaDB under the "default" collection.

---

### POST /chat

Submit a question for RAG-based question answering. Retrieves relevant chunks from stored documents and generates an answer.

**Content-Type:** `application/json`

**Request Body:**

| Field   | Type   | Required | Description          |
|---------|--------|----------|----------------------|
| `query` | string | Yes      | The question to ask  |

**Example Request:**

```bash
curl -X POST http://localhost:8002/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main components of a RAG system?"}'
```

**Example Request (JavaScript):**

```javascript
const response = await fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: 'What are the main components of a RAG system?' })
});
const result = await response.json();
```

**Success Response (200):**

```json
{
    "answer": "A RAG system consists of three main components: a retriever that finds relevant documents, an embedding model that converts text to vectors, and a generator that produces answers based on the retrieved context.",
    "sources": [
        {
            "text": "The retrieval component searches a vector database...",
            "metadata": {"source": "rag_overview.txt", "chunk_index": 2},
            "score": 0.91
        },
        {
            "text": "Embedding models convert text into dense vectors...",
            "metadata": {"source": "rag_overview.txt", "chunk_index": 4},
            "score": 0.85
        }
    ],
    "model": "llama3.1"
}
```

**Placeholder Response (200) — AI modules not implemented:**

```json
{
    "answer": "AI module not yet implemented. See ai/retriever.py and ai/generator.py",
    "sources": [],
    "model": "placeholder"
}
```

**Error Responses:**

| Status | Body | Condition |
|--------|------|-----------|
| 400 | `{"error": "Missing 'query' field"}` | No `query` in request body |
| 500 | `{"error": "Description of error"}` | Internal processing failure |

---

## Configuration

The RAG app reads configuration from `config.json` (managed by the web-based config panel) with these relevant parameters:

| Parameter        | Default              | Description                              |
|-----------------|----------------------|------------------------------------------|
| `chunk_size`    | 500                  | Maximum characters per chunk             |
| `chunk_overlap` | 50                   | Characters of overlap between chunks     |
| `top_k`         | 5                    | Number of chunks to retrieve per query   |
| `embedding_model`| "all-MiniLM-L6-v2" | Sentence-transformers model for embeddings |
| `llm_backend`   | "ollama"             | LLM backend for answer generation        |
| `llm_model`     | "llama3.1"             | Model name for the selected backend      |

---

## Error Handling

All endpoints return JSON error responses. The general pattern:

```json
{
    "error": "Human-readable error description"
}
```

Or for upload-specific errors:

```json
{
    "status": "error",
    "message": "Human-readable error description"
}
```

Common error scenarios:
- **No documents uploaded:** Retrieval returns empty results; the generator acknowledges insufficient context.
- **LLM backend unavailable:** Returns a 500 error with the backend name and failure reason.
- **ChromaDB collection not found:** Returns empty results gracefully (no error).
