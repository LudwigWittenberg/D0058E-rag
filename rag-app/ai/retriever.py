"""
Retrieval module for the RAG pipeline.

Finds the most relevant document chunks for a given query
using vector similarity search against the ChromaDB store.

Students can experiment with:
- Different top_k values (more results = more context but more noise)
- Score thresholds (filter out low-relevance results)
- MMR (Maximal Marginal Relevance) for diverse results
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure shared module is importable
_app_root = Path(__file__).resolve().parent.parent
_project_root = _app_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


@dataclass
class RetrievalResult:
    """A retrieved chunk with relevance score."""

    text: str
    score: float  # 0.0 to 1.0, higher = more relevant
    metadata: dict
    chunk_index: int


def retrieve(
    query: str,
    top_k: int = 5,
    collection_name: str = "default",
    persist_directory: str = None,
) -> list:
    """
    Retrieve the top-K most relevant chunks for a query.

    Embeds the query, searches the vector store, and converts
    cosine distances to relevance scores.

    Invariants:
    - len(result) <= top_k
    - Results are ordered by score descending
    - All scores are in [0.0, 1.0]

    Args:
        query: The search query.
        top_k: Maximum number of results to return.
        collection_name: ChromaDB collection to search.
        persist_directory: Optional path to ChromaDB directory.
            Defaults to ./data/chromadb relative to the app root.

    Returns:
        List of RetrievalResult objects ordered by relevance (highest first).
    """
    from ai.embeddings import embed_text
    from ai.vectorstore import VectorStore

    # Determine persist directory
    if persist_directory is None:
        app_root = Path(__file__).resolve().parent.parent
        persist_directory = str(app_root / "data" / "chromadb")

    # Embed the query
    query_embedding = embed_text(query)
    query_embedding_list = query_embedding.tolist()

    # Query the vector store
    store = VectorStore(persist_directory=persist_directory)
    results = store.query(
        query_embedding=query_embedding_list,
        top_k=top_k,
        collection_name=collection_name,
    )

    documents = results.get("documents", [])
    distances = results.get("distances", [])
    metadatas = results.get("metadatas", [])

    # Convert distances to relevance scores
    # ChromaDB cosine distance: 0 = identical, 2 = opposite
    # Convert to score: 1.0 = identical, 0.0 = opposite
    retrieval_results = []
    for i, (doc, dist, meta) in enumerate(zip(documents, distances, metadatas)):
        # Cosine distance to similarity score: score = 1 - (distance / 2)
        score = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
        chunk_index = meta.get("chunk_index", i) if meta else i

        retrieval_results.append(RetrievalResult(
            text=doc,
            score=score,
            metadata=meta if meta else {},
            chunk_index=int(chunk_index),
        ))

    # Sort by score descending (should already be sorted by ChromaDB, but ensure)
    retrieval_results.sort(key=lambda r: r.score, reverse=True)

    return retrieval_results



