"""
RAG application AI module (student-modifiable).

This package contains the core AI logic for the RAG pipeline:
- chunker: Document splitting into overlapping chunks
- embeddings: Vector embedding generation using sentence-transformers
- vectorstore: ChromaDB-based storage and retrieval
- retriever: Similarity search and result ranking
- generator: Augmented prompt construction and LLM answer generation

Students modify these modules to experiment with different
RAG strategies and parameters.
"""
