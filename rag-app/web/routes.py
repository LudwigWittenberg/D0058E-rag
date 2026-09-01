"""
Route definitions for the RAG app.

Defines endpoints for document upload, Q&A chat,
REST API for inter-app queries, configuration, and the main UI.
"""

import os
import sys
from pathlib import Path

from flask import (
    Blueprint,
    current_app,
    render_template,
    request,
    jsonify,
)
from werkzeug.utils import secure_filename

# Ensure shared module is importable
_app_root = Path(__file__).resolve().parent.parent
_project_root = _app_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from shared.llm_client import LLMClient, LLMConfig, LLMBackend, LLMClientError  # noqa: E402

bp = Blueprint("main", __name__)

# Valid configuration keys and their expected types for validation
_VALID_CONFIG_KEYS = {
    "llm_backend": str,
    "llm_model": str,
    "api_key_openai": str,
    "api_key_gemini": str,
    "ollama_base_url": str,
    "temperature": float,
    "max_tokens": int,
    "system_prompt": str,
    "chunk_size": int,
    "chunk_overlap": int,
    "top_k": int,
    "embedding_model": str,
    "crew_process_type": str,
    "max_iterations": int,
    "integration_mode": bool,
}

_VALID_BACKENDS = {"ollama", "openai", "gemini"}

ALLOWED_EXTENSIONS = {"pdf", "txt", "md"}


def _allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.route("/")
def index():
    """Render the main RAG interface."""
    return render_template("index.html")


@bp.route("/settings")
def settings():
    """Render the configuration settings page."""
    return render_template("settings.html")


@bp.route("/upload", methods=["POST"])
def upload():
    """
    Handle document upload for RAG processing.

    Accepts multipart/form-data with a 'file' field (PDF or TXT).
    Saves the file to data/, triggers AI processing pipeline.
    Returns JSON: {"status": str, "filename": str, "message": str}
    """
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No file selected"}), 400

    if not _allowed_file(file.filename):
        return jsonify({
            "status": "error",
            "message": "File type not allowed. Please upload PDF, TXT, or MD files.",
        }), 400

    filename = secure_filename(file.filename)
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)

    # Attempt to process the document through the AI pipeline
    try:
        from ai.chunker import chunk_document
        from ai.embeddings import embed_batch
        from ai.vectorstore import VectorStore
        from config import config

        # Read file content
        if filename.lower().endswith(".txt") or filename.lower().endswith(".md"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
        elif filename.lower().endswith(".pdf"):
            # Try to extract text from PDF
            try:
                import PyPDF2
                with open(filepath, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""
            except ImportError:
                # PyPDF2 not installed — read as raw text fallback
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
        else:
            text = ""

        if not text.strip():
            return jsonify({
                "status": "warning",
                "filename": filename,
                "message": "File uploaded but no text content could be extracted.",
            })

        # Chunk the document
        chunk_size = config.get("chunk_size", 500)
        chunk_overlap = config.get("chunk_overlap", 50)
        chunks = chunk_document(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            metadata={"source": filename},
        )

        if not chunks:
            raise NotImplementedError("Chunker not yet implemented")

        # Embed chunks
        embedding_model = config.get("embedding_model", "all-MiniLM-L6-v2")
        chunk_texts = [c.text for c in chunks]
        embeddings = embed_batch(chunk_texts, model=embedding_model)

        if embeddings is None:
            raise NotImplementedError("Embeddings not yet implemented")

        # Store in vector store
        store = VectorStore(persist_directory=os.path.join(upload_folder, "chromadb"))
        metadatas = [{"source": filename, "chunk_index": c.index} for c in chunks]
        store.add_documents(
            texts=chunk_texts,
            embeddings=embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings,
            metadatas=metadatas,
            collection_name="default",
        )

        return jsonify({
            "status": "success",
            "filename": filename,
            "message": f"Document processed: {len(chunks)} chunks created and stored.",
        })

    except NotImplementedError:
        # AI module not yet implemented — file saved but not processed
        return jsonify({
            "status": "partial",
            "filename": filename,
            "message": "File saved. AI processing not yet implemented (see ai/ modules).",
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "filename": filename,
            "message": f"File saved but processing failed: {str(e)}",
        }), 500


@bp.route("/chat", methods=["POST"])
def chat():
    """
    Handle Q&A chat requests with RAG context.

    Accepts JSON: {"query": str}
    Returns JSON: {"answer": str, "sources": list, "model": str}
    """
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field"}), 400

    query = data["query"]

    try:
        from ai.retriever import retrieve
        from ai.generator import generate_answer
        from config import config
        from ai.models import create_client

        # Retrieve relevant chunks
        top_k = config.get("top_k", 5)
        results = retrieve(query=query, top_k=top_k, collection_name="default")

        if results is None:
            raise NotImplementedError("Retriever not yet implemented")

        context_chunks = [r.text for r in results]
        sources = [
            {"text": r.text, "metadata": r.metadata, "score": r.score}
            for r in results
        ]

        # Generate answer
        backend = config.get("llm_backend", "ollama")
        llm_client = create_client(backend, config.get_all())
        result = generate_answer(
            query=query,
            context_chunks=context_chunks,
            llm_client=llm_client,
        )

        if result is None:
            raise NotImplementedError("Generator not yet implemented")

        return jsonify({
            "answer": result.get("answer", ""),
            "sources": sources,
            "model": result.get("model", "unknown"),
        })

    except NotImplementedError:
        return jsonify({
            "answer": "AI module not yet implemented. See ai/retriever.py and ai/generator.py",
            "sources": [],
            "model": "placeholder",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/query", methods=["GET"])
def api_query():
    """
    REST API endpoint for inter-app RAG queries.

    Accepts query params: ?q=<query>&top_k=5&collection=default
    Returns JSON: {"answer": str, "context": list, "sources": list, "model": str, "query": str}
    """
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Missing 'q' query parameter"}), 400

    top_k = request.args.get("top_k", 5, type=int)
    collection = request.args.get("collection", "default")

    try:
        from ai.retriever import retrieve
        from ai.generator import generate_answer
        from config import config
        from ai.models import create_client

        # Retrieve relevant chunks
        results = retrieve(query=query, top_k=top_k, collection_name=collection)

        if results is None:
            raise NotImplementedError("Retriever not yet implemented")

        context_chunks = [r.text for r in results]
        sources = [
            {"text": r.text, "metadata": r.metadata, "score": r.score}
            for r in results
        ]

        # Generate answer
        backend = config.get("llm_backend", "ollama")
        llm_client = create_client(backend, config.get_all())
        result = generate_answer(
            query=query,
            context_chunks=context_chunks,
            llm_client=llm_client,
        )

        if result is None:
            raise NotImplementedError("Generator not yet implemented")

        return jsonify({
            "answer": result.get("answer", ""),
            "context": context_chunks,
            "sources": sources,
            "model": result.get("model", "unknown"),
            "query": query,
        })

    except NotImplementedError:
        return jsonify({
            "answer": "AI module not yet implemented. See ai/ modules.",
            "context": [],
            "sources": [],
            "model": "placeholder",
            "query": query,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------------------------------
# Configuration API routes
# -------------------------------------------------------------------------
# RAG Memory Chat API (vector store as conversation memory)
# Uses: ai/rag_memory.py, ai/rag_chat.py, ai/rag_prompts.py
# -------------------------------------------------------------------------

# Singleton RAG memory instance
_rag_memory = None


def _get_rag_memory():
    """Get or create the RAG memory singleton."""
    global _rag_memory
    if _rag_memory is None:
        from ai.rag_memory import RAGMemory
        _rag_memory = RAGMemory()
    return _rag_memory


@bp.route("/api/rag-memory/chat", methods=["POST"])
def rag_memory_chat():
    """
    Chat using vector store as memory.

    Accepts JSON: {"message": str, "top_k": int, "min_score": float}
    Returns JSON: {"response": str, "retrieved_context": [...], "store_contents": [...], "prompt_context": str, "stats": {...}}
    """
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400

    message = data["message"]
    top_k = data.get("top_k", 3)
    min_score = data.get("min_score", 0.0)

    try:
        from ai.rag_chat import generate_rag_response
        from shared.llm_client import LLMClient, LLMConfig, LLMBackend
        from config import config

        memory = _get_rag_memory()

        # Create LLM client from config
        backend_str = config.get("llm_backend", "ollama")
        llm_config = LLMConfig(
            backend=LLMBackend(backend_str),
            model_name=config.get("llm_model", "Llama-3.2-3B-Instruct-Q4_K_M"),
            api_key=config.get(f"api_key_{backend_str}"),
            base_url=config.get("ollama_base_url", "http://localhost:11434"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 512),
        )
        llm_client = LLMClient(llm_config)

        result = generate_rag_response(
            message=message,
            memory=memory,
            llm_client=llm_client,
            top_k=top_k,
            min_score=min_score,
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/rag-memory/clear", methods=["POST"])
def rag_memory_clear():
    """Clear the RAG memory store."""
    global _rag_memory
    try:
        memory = _get_rag_memory()
        memory.clear()
        _rag_memory = None  # Force re-creation on next use
        return jsonify({"status": "ok", "message": "RAG memory cleared"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/collection/clear", methods=["POST"])
def clear_collection():
    """Clear the document vector collection (delete all uploaded document embeddings)."""
    try:
        from ai.vectorstore import VectorStore

        upload_folder = current_app.config["UPLOAD_FOLDER"]
        store = VectorStore(persist_directory=os.path.join(upload_folder, "chromadb"))
        store.delete_collection("default")

        return jsonify({
            "status": "ok",
            "message": "Vector collection cleared. You can now upload new documents.",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------------------------------
# Chunking visualization API
# -------------------------------------------------------------------------


@bp.route("/api/chunks/visualize", methods=["POST"])
def visualize_chunks():
    """
    Chunk text and return the chunks with positions for visualization.

    Accepts JSON: {"text": str, "chunk_size": int, "chunk_overlap": int}
    Returns JSON: {"chunks": [{"text": str, "start": int, "end": int, "index": int}], "stats": {...}}
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"]
    chunk_size = data.get("chunk_size", 500)
    chunk_overlap = data.get("chunk_overlap", 50)

    if chunk_size < 1:
        return jsonify({"error": "chunk_size must be positive"}), 400
    if chunk_overlap < 0:
        return jsonify({"error": "chunk_overlap must be non-negative"}), 400
    if chunk_overlap >= chunk_size:
        return jsonify({"error": "chunk_overlap must be less than chunk_size"}), 400

    try:
        from ai.chunker import chunk_document

        chunks = chunk_document(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        result_chunks = []
        for c in chunks:
            result_chunks.append({
                "text": c.text,
                "start": c.start_char,
                "end": c.end_char,
                "index": c.index,
                "chars": len(c.text),
            })

        stats = {
            "total_chars": len(text),
            "num_chunks": len(chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "avg_chunk_chars": round(sum(len(c.text) for c in chunks) / max(len(chunks), 1)),
            "step_size": chunk_size - chunk_overlap,
        }

        return jsonify({"chunks": result_chunks, "stats": stats})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------------------------------
# Embedding similarity API
# -------------------------------------------------------------------------


@bp.route("/api/embeddings/similarity", methods=["POST"])
def compute_similarity():
    """
    Compute pairwise cosine similarities for a list of sentences.

    Accepts JSON: {"sentences": list[str], "model": str}
    Returns JSON: {"similarities": list[list[float]], "model": str, "dimensions": int}
    """
    data = request.get_json()
    if not data or "sentences" not in data:
        return jsonify({"error": "Missing 'sentences' field"}), 400

    sentences = data["sentences"]
    model_name = data.get("model", "all-MiniLM-L6-v2")

    if len(sentences) < 2:
        return jsonify({"error": "Need at least 2 sentences"}), 400

    try:
        from ai.embeddings import embed_batch
        import numpy as np

        # Embed all sentences
        embeddings = embed_batch(sentences, model=model_name)

        # Compute cosine similarity matrix
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = embeddings / norms

        # Cosine similarity = dot product of normalized vectors
        sim_matrix = (normalized @ normalized.T).tolist()

        return jsonify({
            "similarities": sim_matrix,
            "model": model_name,
            "dimensions": embeddings.shape[1],
            "num_sentences": len(sentences),
        })

    except ImportError as e:
        return jsonify({"error": f"Missing dependency: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------------------------------


@bp.route("/api/config", methods=["GET"])
def get_config():
    """
    Return the current application configuration as JSON.

    Returns JSON: full config dictionary from ConfigManager.
    """
    from config import config

    return jsonify(config.get_all())


@bp.route("/api/config", methods=["POST"])
def update_config():
    """
    Accept partial configuration updates, validate, and persist.

    Accepts JSON: {"key": value, ...} with any valid config keys.
    Returns JSON: {"status": "ok", "updated": list[str]} on success,
                  or {"status": "error", "message": str} on failure.
    """
    from config import config

    data = request.get_json()
    if data is None or not isinstance(data, dict):
        return jsonify({"status": "error", "message": "Request body must be a JSON object"}), 400

    errors = []
    updated_keys = []

    for key, value in data.items():
        # Validate key is recognized
        if key not in _VALID_CONFIG_KEYS:
            errors.append(f"Unknown configuration key: '{key}'")
            continue

        # Validate llm_backend value
        if key == "llm_backend" and value not in _VALID_BACKENDS:
            errors.append(
                f"Invalid llm_backend '{value}'. Must be one of: {', '.join(_VALID_BACKENDS)}"
            )
            continue

        # Validate numeric ranges
        if key == "temperature":
            try:
                val = float(value)
                if val < 0.0 or val > 2.0:
                    errors.append("temperature must be between 0.0 and 2.0")
                    continue
            except (ValueError, TypeError):
                errors.append("temperature must be a number")
                continue

        if key == "max_tokens":
            try:
                val = int(value)
                if val < 1:
                    errors.append("max_tokens must be a positive integer")
                    continue
            except (ValueError, TypeError):
                errors.append("max_tokens must be an integer")
                continue

        if key == "chunk_size":
            try:
                val = int(value)
                if val < 1:
                    errors.append("chunk_size must be a positive integer")
                    continue
            except (ValueError, TypeError):
                errors.append("chunk_size must be an integer")
                continue

        if key == "chunk_overlap":
            try:
                val = int(value)
                if val < 0:
                    errors.append("chunk_overlap must be a non-negative integer")
                    continue
            except (ValueError, TypeError):
                errors.append("chunk_overlap must be an integer")
                continue

        if key == "top_k":
            try:
                val = int(value)
                if val < 1:
                    errors.append("top_k must be a positive integer")
                    continue
            except (ValueError, TypeError):
                errors.append("top_k must be an integer")
                continue

        if key == "max_iterations":
            try:
                val = int(value)
                if val < 1:
                    errors.append("max_iterations must be a positive integer")
                    continue
            except (ValueError, TypeError):
                errors.append("max_iterations must be an integer")
                continue

        # Apply the update
        config.set(key, value)
        updated_keys.append(key)

    if errors:
        return jsonify({"status": "error", "message": "; ".join(errors), "updated": updated_keys}), 400

    return jsonify({"status": "ok", "updated": updated_keys})


@bp.route("/api/config/test-connection", methods=["POST"])
def test_connection():
    """
    Test LLM backend connectivity and return status.

    Optionally accepts JSON: {"backend": str, "api_key": str, "model": str}
    to test a specific backend. If no body is provided, tests the currently
    configured backend.

    Returns JSON: {"status": "connected"|"error", "backend": str, "model": str, "message": str}
    """
    from config import config

    data = request.get_json() or {}

    # Use provided values or fall back to current config
    backend_name = data.get("backend", config.get("llm_backend", "ollama"))
    model_name = data.get("model", config.get("llm_model", "Llama-3.2-3B-Instruct-Q4_K_M"))

    # Determine API key
    if "api_key" in data:
        api_key = data["api_key"]
    elif backend_name == "openai":
        api_key = config.get("api_key_openai")
    elif backend_name == "gemini":
        api_key = config.get("api_key_gemini")
    else:
        api_key = None

    base_url = data.get("base_url", config.get("ollama_base_url", "http://localhost:11434"))

    # Map backend string to enum
    backend_map = {"ollama": LLMBackend.OLLAMA, "openai": LLMBackend.OPENAI, "gemini": LLMBackend.GEMINI}
    backend_enum = backend_map.get(backend_name)
    if backend_enum is None:
        return jsonify({
            "status": "error",
            "backend": backend_name,
            "model": model_name,
            "message": f"Unknown backend '{backend_name}'. Must be one of: ollama, openai, gemini",
        }), 400

    try:
        llm_config = LLMConfig(
            backend=backend_enum,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
        )
        client = LLMClient(llm_config)
        result = client.test_connection()
        return jsonify(result)
    except LLMClientError as e:
        return jsonify({
            "status": "error",
            "backend": backend_name,
            "model": model_name,
            "message": e.reason,
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "backend": backend_name,
            "model": model_name,
            "message": str(e),
        })
