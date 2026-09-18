"""
Vector store module for the RAG pipeline.

Provides a ChromaDB-based vector store for storing and
querying document embeddings. ChromaDB handles persistence
and similarity search efficiently.

In this task you will implement:
- Connecting to ChromaDB with persistent storage
- Adding documents with embeddings and metadata
- Querying for similar documents using cosine similarity
- Managing collections (delete, list)
"""

import uuid
from typing import Optional
import itertools

try:
    import chromadb
except ImportError:
    raise ImportError("Chroma is not installed")


class VectorStore:
    """ChromaDB-based vector store for document embeddings.

    ChromaDB is a vector database that stores embeddings and enables
    fast similarity search. It uses HNSW (Hierarchical Navigable Small
    World) graphs internally for approximate nearest neighbor search.

    Key concepts:
    - Collection: a named group of documents (like a database table)
    - Embedding: a vector (list of floats) representing a text chunk
    - Cosine similarity: measures how similar two vectors are (0-2 distance scale)
    - Persistence: data is saved to disk so it survives app restarts
    """

    def __init__(self, persist_directory: str = "./data/chromadb"):
        """
        Initialize the vector store.

        TODO: Implement this method:
        1. Check if chromadb is installed (raise ImportError if not)
        2. Store the persist_directory
        3. Create a ChromaDB PersistentClient pointing to that directory

        Hint: Use chromadb.PersistentClient(path=persist_directory)

        Args:
            persist_directory: Path to ChromaDB persistence directory.

        Raises:
            ImportError: If chromadb is not installed.
        """

        # chromaDB is checked in the top if its isnstalled or not.
       
        self.client = chromadb.PersistentClient(path=persist_directory)

    def add_documents(
        self,
        texts: list,
        embeddings: list,
        metadatas: list,
        collection_name: str = "default",
    ) -> None:
        """
        Add documents with their embeddings to the store.

        TODO: Implement this method:
        1. Return early if texts is empty
        2. Get or create the collection using self.client.get_or_create_collection()
           - Set metadata={"hnsw:space": "cosine"} for cosine similarity
        3. Generate a unique ID for each document (use uuid.uuid4())
        4. Clean metadatas: ensure all values are str, int, float, or bool
           (ChromaDB doesn't accept other types)
        5. Call collection.add(ids=..., documents=..., embeddings=..., metadatas=...)

        Args:
            texts: Document text chunks.
            embeddings: Corresponding embedding vectors (list of lists of floats).
            metadatas: Metadata for each chunk (e.g., {"chunk_index": 0, "source": "file.pdf"}).
            collection_name: Target collection name.
        """
        
        # List is empty
        if not texts:
            return
        
        collection = self.client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
        
        ids = []
        
        ids = [str(uuid.uuid4()) for _ in texts]
        
        cleaned_metadatas = []

        for metadata in metadatas:
            cleaned = {}

            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    cleaned[key] = value
                else:
                    cleaned[key] = str(value)

            cleaned_metadatas.append(cleaned)
        
        collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=cleaned_metadatas)
        


    def query(
        self,
        query_embedding: list,
        top_k: int = 5,
        collection_name: str = "default",
    ) -> dict:
        """
        Query the vector store for similar documents.

        TODO: Implement this method:
        1. Try to get the collection (return empty results if it doesn't exist)
        2. Check if collection has documents (collection.count()), return empty if 0
        3. Limit top_k to the actual number of documents available
        4. Call collection.query(query_embeddings=[query_embedding], n_results=top_k)
        5. Flatten the results (ChromaDB returns nested lists for batch queries)
        6. Return {"documents": list[str], "distances": list[float], "metadatas": list[dict]}

        Note: ChromaDB returns cosine DISTANCES (not similarities):
        - Distance 0.0 = identical vectors
        - Distance 2.0 = opposite vectors
        - To convert to similarity: similarity = 1.0 - (distance / 2.0)

        Args:
            query_embedding: The query vector (list of floats).
            top_k: Number of results to return.
            collection_name: Collection to search.

        Returns:
            {"documents": list[str], "distances": list[float], "metadatas": list[dict]}
        """
        try:
            collection = self.client.get_collection(name=collection_name)
            
            collection_count = collection.count()
            
            if collection_count <= 0:
                raise Exception("Collection is empty")
            
            if top_k > collection_count:
                top_k = collection_count
                
            result = collection.query(query_embeddings=[query_embedding], n_results=top_k)
            
            # relevant_docs = collection.get(include=["documents", "distances", "metadatas"])

            
            return {
                "documents": result["documents"][0],
                "distances": result["distances"][0],
                "metadatas": result["metadatas"][0]
            }
        except:
            return  {
                "documents": [],
                "distances": [],
                "metadatas": [],
            }
        
        

    def delete_collection(self, collection_name: str = "default") -> None:
        """
        Delete a collection and all its documents.

        TODO: Implement this method:
        1. Try to delete the collection using self.client.delete_collection(name=...)
        2. Catch any exception (collection may not exist — that's fine)

        Args:
            collection_name: Name of collection to delete.
        """
        try:
           self.client.delete_collection(name=collection_name)
        except:
            pass

    def list_collections(self) -> list:
        """
        List all collection names.

        TODO: Implement this method:
        1. Call self.client.list_collections()
        2. Return a list of collection name strings

        Returns:
            List of collection name strings.
        """
        collections = self.client.list_collections()
        return [collection.name for collection in collections]
