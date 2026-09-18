"""
RAG-based conversation memory module.

This module implements conversation memory using a vector store (ChromaDB).
Instead of keeping all messages in a fixed-size array, each message is
embedded and stored in a vector database. Relevant past messages are
retrieved by semantic similarity for each new query.

Students implement:
- add_message(): embed and store a message
- retrieve_context(): find relevant past messages for a query
"""

import os
from pathlib import Path
from ai.embeddings import embed_text

_app_root = Path(__file__).resolve().parent.parent
import sys
if str(_app_root) not in sys.path:
    sys.path.insert(0, str(_app_root))


class RAGMemory:
    """
    Vector store-based conversation memory.

    Stores each message as an embedding in ChromaDB and retrieves
    relevant past messages by semantic similarity.
    """

    def __init__(self, persist_dir: str = None, collection_name: str = "rag_memory"):
        """
        Initialize RAG memory.

        Args:
            persist_dir: Directory for ChromaDB storage.
            collection_name: ChromaDB collection name.
        """
        from ai.vectorstore import VectorStore

        if persist_dir is None:
            persist_dir = str(_app_root / "data" / "rag_memory_db")

        self._store = VectorStore(persist_directory=persist_dir)
        self._collection_name = collection_name
        self._msg_count = 0

    def add_message(self, role: str, content: str) -> int:
        """
        Embed and store a message in the vector database.

        TODO: Implement this method:
        1. Increment self._msg_count
        2. Embed the content using embed_text() from ai.embeddings
        3. Store in ChromaDB via self._store.add_documents()
           with metadata: {"role": role, "index": self._msg_count}

        Args:
            role: "user" or "assistant"
            content: Message text

        Returns:
            The message index.
        """
        self._msg_count += 1
        
        embedding = embed_text(text=content)
        
        metadata = {"role": role, "index": self._msg_count}
        
        self._store.add_documents(
            texts=[content],
            embeddings=[embedding],
            metadatas=[metadata],
            collection_name=self._collection_name,
        )
        
        return self._msg_count
        

    def retrieve_context(self, query: str, top_k: int = 3, min_score: float = 0.0) -> list:
        """
        Retrieve relevant past messages for a query.

        TODO: Implement this method:
        1. Embed the query using embed_text()
        2. Search ChromaDB via self._store.query()
        3. Convert distances to scores: score = 1.0 - (distance / 2.0)
        4. Filter by min_score
        5. Sort by original conversation order (index)

        Args:
            query: The new message to find context for.
            top_k: Maximum number of past messages to retrieve.
            min_score: Minimum similarity score (0.0-1.0).

        Returns:
            List of dicts: [{"text": str, "score": float, "role": str, "index": int}]
        """
        query_emb = embed_text(query)
        
        result = self._store.query(
            query_embedding=query_emb, # type: ignore
            top_k=top_k,
            collection_name=self._collection_name,
        )
        
        matches = []

        for text, distance, metadata in zip(
            result["documents"],
            result["distances"],
            result["metadatas"],
        ):
            score = 1.0 - (distance / 2.0)

            if score >= min_score:
                matches.append({
                    "text": text,
                    "score": score,
                    "role": metadata["role"],
                    "index": metadata["index"],
                })

        matches.sort(key=lambda x: x["index"])

        return matches
        

    def get_all_messages(self) -> list:
        """
        Return every stored message, in conversation order.

        Used by the UI to display the current contents of the store.

        Returns:
            List of dicts: [{"text": str, "role": str, "index": int}]
        """
        try:
            collection = self._store.client.get_collection(
                name=self._collection_name
            )
            result = collection.get(include=["documents", "metadatas"])
        except Exception:
            return []

        messages = [
            {
                "text": text,
                "role": metadata.get("role", ""),
                "index": metadata.get("index", 0),
            }
            for text, metadata in zip(
                result["documents"] or [], result["metadatas"] or []
            )
        ]
        messages.sort(key=lambda m: m["index"])
        return messages

    def get_message_count(self) -> int:
        """Return the total number of messages stored."""
        return self._msg_count

    def clear(self) -> None:
        """Clear all stored messages."""
        self._store.delete_collection(self._collection_name)
        self._msg_count = 0
