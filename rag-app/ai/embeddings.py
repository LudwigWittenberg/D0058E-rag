"""
Embedding generation module for the RAG pipeline.

Generates vector embeddings for text using sentence-transformers models.
These embeddings capture semantic meaning and enable similarity search.

Students can experiment with:
- Different embedding models (all-MiniLM-L6-v2, all-mpnet-base-v2, etc.)
- Comparing embedding dimensions and quality trade-offs
- Batch vs. single embedding performance
"""

import numpy as np

# Cache loaded models to avoid reloading on every call
_model_cache: dict = {}


def _get_model(model: str):
    """
    Load and cache a sentence-transformers model.

    Args:
        model: Model identifier (e.g., "all-MiniLM-L6-v2").

    Returns:
        SentenceTransformer model instance.
    """
    if model not in _model_cache:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers package not installed. "
                "Run: pip install sentence-transformers"
            )
        _model_cache[model] = SentenceTransformer(model)
    return _model_cache[model]


def embed_text(text: str, model: str = "all-mpnet-base-v2") -> np.ndarray:
    """
    Generate an embedding vector for a text string.

    Args:
        text: Input text to embed.
        model: Embedding model identifier.

    Returns:
        Numpy array of shape (embedding_dim,) — consistent dimension
        for a given model.
    """
    if not text or not text.strip():
        # Return a zero vector for empty text — dimension depends on model
        st_model = _get_model(model)
        dim = st_model.get_sentence_embedding_dimension()
        return np.zeros(dim, dtype=np.float32)

    st_model = _get_model(model)
    embedding = st_model.encode(text, convert_to_numpy=True)
    return embedding.astype(np.float32)


def embed_batch(texts: list, model: str = "all-mpnet-base-v2") -> np.ndarray:
    """
    Generate embeddings for a batch of texts.

    More efficient than calling embed_text() in a loop because
    sentence-transformers can batch encode.

    Args:
        texts: List of input texts.
        model: Embedding model identifier.

    Returns:
        Numpy array of shape (len(texts), embedding_dim).
    """
    if not texts:
        st_model = _get_model(model)
        dim = st_model.get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype=np.float32)

    st_model = _get_model(model)
    embeddings = st_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.astype(np.float32)



