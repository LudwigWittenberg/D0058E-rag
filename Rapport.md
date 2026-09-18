# Rapport Lab 2 - RAG

## Table of contents

- [Task 2.1](#task-21)
- [Task 2.2](#task-22)
- [Task 2.3](#task-23)
- [Task 2.4](#task-24)
- [Task 2.5](#task-25)
- [Task 2.6](#task-26)

## Task 2.1

### Questions to answer

#### In this project, is ChromaDB a separate server or embedded in the Flask process?

It embedded into the flask process or as I would called it he backend/server.

#### What are the two main flows in a RAG pipeline?

I would way that it is the embedding/ingest and retrival parts. 

#### Name one advantage and one limitation of client-side vector stores.

Advantage: Fast and secure. Nothing leaves the users phone. 

Limitation: Takes more power and cant be shared cross devices.

## Task 2.2

### Questions to answer

#### How does increasing chunk_size affect the number of chunks?

When increasing the chunk size we increase the number of words in that chunk, which does as we will get more words into one chunk which leades to less chunks.

#### What information is at risk when overlap is 0?

I think the rest is that it may miss some extra important words. We can ofcourse still  argue that it could still miss important words whenwe have overlap. But with 0 overlap we may miss some important knowledge.

#### What chunk_size would you choose for a FAQ document vs. a research paper?

I think for a FAQ we can have a bit smaller. Around 200 I think is good. But of cource it depends on the FAQ section. If the answeres are long maybe a bigger shuck size would be better. But for a normal something around 200 I think would be good. When it comes to the research paper I would have a higher chunk size with atleast 600.

## Task 2.3

### Question to answer

#### What is the dimensionality of the default embedding model?

The all-MiniLM-L6-v2 (Default) model have 384 dimensions.

#### Why is mean pooling preferred over [CLS] token pooling?

CLS only captures what the model compress into a single token. While Mean, every token cintibutes equally.

#### In the similarity explorer, which two sentences had the highest similarity?

It was the sentence 1 and 2.

1. The cat sat on the mat and purred
2. A kitten is sleeping on a soft rug

## Task 2.4

**My edits**

```python
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

```

### Test result

```bash
============================================================
Task 2.4 — VectorStore Test (Synthetic Embeddings)
============================================================

--- Part 1: Initialization ---
✓ VectorStore initialized with persistent storage at /tmp/test_chromadb_task24

--- Part 2: Adding documents ---
  Creating synthetic 384-dim embeddings with known directions:
    vec_cats = [1, 0, 0, ...]  (points along dimension 0)
    vec_dogs = [0, 1, 0, ...]  (points along dimension 1)
    vec_pets = [0.7, 0.7, 0, ...] (45° between cats and dogs)

✓ 3 documents added to collection 'animals'

--- Part 3: Querying (vector close to 'cats') ---
  Query vector: [0.9, 0.1, 0, ...]  (close to cats direction)

  NOTE: ChromaDB returns COSINE DISTANCE, not similarity.
    The cosine formula cos(θ) = (A·B)/(||A||×||B||) computes SIMILARITY (-1 to +1).
    ChromaDB converts to DISTANCE for sorting: distance = 1 - similarity
    Distance 0 = identical direction (similarity 1)
    Distance 1 = orthogonal (similarity 0)
    Distance 2 = opposite direction (similarity -1)

  Results (top_k=3, sorted by cosine distance — lower = more similar):
    1. distance=0.0061  (similarity=0.9939)  Cats are independent animals that groom themselves
    2. distance=0.2191  (similarity=0.7809)  Pets bring joy and comfort to families
    3. distance=0.8896  (similarity=0.1104)  Dogs are loyal companions that love walks

✓ Correct! 'Cats' document is closest (smallest cosine distance)

--- Part 4: Querying (vector close to 'dogs') ---
  Query vector: [0.1, 0.9, 0, ...]  (close to dogs direction)

  Results (top_k=3, sorted by cosine distance — lower = more similar):
    1. distance=0.0061  (similarity=0.9939)  Dogs are loyal companions that love walks
    2. distance=0.2191  (similarity=0.7809)  Pets bring joy and comfort to families
    3. distance=0.8896  (similarity=0.1104)  Cats are independent animals that groom themselves

✓ Correct! 'Dogs' document is closest when query points toward dim 1

--- Part 5: List collections ---
  Collections: ['animals']
✓ Collection 'animals' exists

--- Part 6: Delete collection ---
  Collections after delete: []
✓ Collection 'animals' successfully deleted

============================================================
ALL TESTS PASSED ✓
============================================================

Your VectorStore implementation correctly:
  ✓ Initializes with persistent ChromaDB storage
  ✓ Adds documents with embeddings and metadata
  ✓ Queries return results sorted by cosine distance
  ✓ Lists existing collections
  ✓ Deletes collections

Next: Proceed to Task 2.5 (RAG Memory) to add conversation
context and real embeddings from the Ollama model.
```

### Reflection questions

#### What is the relationship between cosine distance and cosine similarity? Why does ChromaDB use distance?

Cosine similarity is how silimar two vectors are. If they point in the same direction. Distance is used for see the distance between two vectors, which is used by chromaDB. Vector DB performs nearest neighbour. The goal is to find the vectors which is nearest the current we have. Therefor vector DB uses distance instead of similarity.

#### What happens if you try to query a collection that doesn't exist?

Currently the application will return empty arrays for a collection that dosent exist

#### Why is hnsw:space: cosine important? What would change with L2 (Euclidean) distance?

Cosine focuses on the the directions of the vectors while Euclidian will measure the straight line distance between vectors.

#### What is the purpose of the metadata attached to each document?

It stores some extra data from each document.

#### Look at Part 3 and Part 4 output — why does the "pets" document appear as the second result in both queries?

Because Pets is a collective name for both dogs and cats. When I talk about pets it can be a dog or a cat. But when I talk about a dog it cannot be a cat but it can be a pet.

## Task 2.5

**My edits**

```python
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
    
    self._store.add_documents(texts=[content], embeddings=[embedding], metadatas=[metadata])
    
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
    
    result = self._store.query(query_embedding=query_emb, top_k=top_k)
    
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
```

```python
def build_rag_memory_prompt(user_message: str, retrieved_context: list) -> str:
    """
    Build the prompt with retrieved memory context.

    TODO: Implement this function:
    1. If retrieved_context is empty, return just the user message formatted
    2. Format each retrieved memory (include role, score, text)
    3. Combine memories + user message into a prompt string

    Args:
        user_message: The user's current message.
        retrieved_context: List of retrieved memory dicts from RAGMemory.retrieve_context():
                          [{"text": str, "score": float, "role": str, "index": int}]

    Returns:
        Formatted prompt string containing memories + user message.
    """
    if not retrieved_context:
        return f"User: {user_message}"
    
    memory_lines = []

    for memory in retrieved_context:
        role = memory["role"]
        score = memory["score"]
        text = memory["text"]

        memory_lines.append(
            f"{role} (score: {score:.2f}): {text}"
        )

    memories = "\n".join(memory_lines)

    prompt = (
        "Relevant conversation memory:\n"
        f"{memories}\n\n"
        f"User: {user_message}"
    )

    return prompt
```

| Step | Phase | Message | Store size | Retrived | Context tokens | Notes |
|------|-------|---------|------------|----------|----------------|-------|
| 1 | fact | My name is Alice and I study computer science at LTU. | 2 | 0  | 0 | First message |
| 2 | fact | My favorite color is purple and my lucky number is 42. | 4 | 2 | 86 | Starts to retrive and fill context |
| 3 | fact | I'm working on autonomous drones for forest monitoring. | 6 | 3 | 214 |  |
| 4 | filler | Can you explain how binary search works? | 8 | 3 | 294 |  |
| 5 | filler | What is the difference between TCP and UDP? | 10 | 3 | 210 | Does not retrive these kind of questions |
| 6 | filler | Explain the concept of Big O notation. | 12 | 3 | 575 | Big jump in context token |
| 7 | filler | How does garbage collection work in Python? | 14 | 3 | 574 |  |
| 8 | filler | What are design patterns in software engineering? | 16 | 3 | 210 | Smaller context now  |
| 9 | recall |  What is my name and where do I study? | 18 | 3 | 214 | Remembers name and what I study. |
| 10 | recall | What is my favorite color?  | 20 | 3 | 150 | Remeberg color and number |
| 11 | recall | What is my project about?  | 22 | 3 | 450 | Remember drones |

### Question to answer

#### At step 9 (recall), which stored messages were retrieved? Were they the right ones?

```bash
★ 0.7742846012115479 [user #1]
My name is Alice and I study computer science at LTU.
```

```bash
★ 0.6532377600669861 [assistant #2]
Nice to meet you, Alice! I've retrieved a memory from our previous conversation. It seems you mentio
```

```bash
★ 0.6231347322463989 [assistant #4]
Nice to continue our conversation, Alice. It seems like we didn't dive into any specific topics rela
```

For this questions tes they all remember my name. One of the messages mention the LTU but I cant see the last parts of the retrived information.

#### How do context tokens change between fact steps (short context) and recall steps (retrieved context)?

From what I saw we had a very small context at the fact question. The context grew a little bit to the recall question but not much. we still had aroun 200 tokens in context.

#### Compare with LAB1 token trimming: does RAG memory use more or fewer tokens per turn?

RAG uses so much less tokens. I can see from lab 1 that the context was between 1800 and 2900 tokens. With RAG its around 200 tokens. But ofcourse it depends on the application also.

#### Try changing min_score to 0.5 — do any recall steps lose their retrieved context?

No, not that I can see it.

### Reflect on trade-offs

#### When does RAG over history work better than a sliding window?

I would say bigger tasks with more information. And it also depends on the application what its goal is. When the conversation is long.

#### When does it fail? (e.g., "continue what you were saying" — needs recency, not relevance)

On recent questions. For example if I say continue. It has no previous knowledge of what we talked about.

#### How would you combine both approaches? (recent N messages + RAG for older ones)

I would use RAG for all the information but also use the recent N messages for solving the part of what we exactly talked about. This makes it more like a conversation instead of it just answering my questions.

## Task 2.6