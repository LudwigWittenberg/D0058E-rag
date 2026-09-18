"""
Document chunking module for the RAG pipeline.

Splits documents into overlapping text chunks suitable
for embedding and retrieval.

Students can experiment with:
- Different chunk sizes (smaller = more precise, larger = more context)
- Different overlap values (more overlap = better continuity)
- Sentence-aware chunking strategies
"""

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A text chunk with metadata."""

    text: str
    index: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)  # source filename, page number, etc.


def chunk_document(
    text: str,
    chunk_size: int = 200,
    chunk_overlap: int = 50,
    metadata: dict = None,
) -> list:
    """
    Split a document into overlapping chunks.

    Uses a sliding window approach: each chunk starts at
    (chunk_size - chunk_overlap) characters after the previous one,
    ensuring consecutive chunks share exactly chunk_overlap characters.

    Invariants:
    - Each chunk.text length <= chunk_size
    - Consecutive chunks overlap by chunk_overlap characters
    - Union of all chunks covers the entire input text

    Args:
        text: The full document text.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.
        metadata: Metadata to attach to each chunk (e.g., source file).

    Returns:
        List of Chunk objects in document order.
    """
    if metadata is None:
        metadata = {}

    # Validate parameters
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    # Handle empty text
    if not text:
        return []

    chunks = []
    step = chunk_size - chunk_overlap
    start = 0
    index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]

        chunks.append(Chunk(
            text=chunk_text,
            index=index,
            start_char=start,
            end_char=end,
            metadata=dict(metadata),  # Copy metadata for each chunk
        ))

        index += 1
        start += step

        # If we've reached the end, stop
        if end == len(text):
            break

    return chunks


# TODO: Implement sentence-aware chunking that splits on sentence boundaries
