"""
Test script for Task 2.4 — VectorStore implementation.

This script tests your VectorStore class using SYNTHETIC embeddings
(hand-crafted vectors). No embedding model is needed — we test the
storage and retrieval logic in isolation.

WHY SYNTHETIC EMBEDDINGS?
  Real embeddings come from a model (e.g., all-MiniLM-L6-v2) and are
  384-dimensional vectors where direction encodes meaning. Here we craft
  vectors manually with known directions so we can predict which documents
  should be "closest" to a query — making the test deterministic.

HOW IT WORKS:
  - vec_cats points along dimension 0:  [1, 0, 0, 0, ...]
  - vec_dogs points along dimension 1:  [0, 1, 0, 0, ...]
  - vec_pets is between them (45°):     [0.7, 0.7, 0, 0, ...]

  A query vector close to vec_cats (e.g., [0.9, 0.1, 0, ...]) should
  return "cats" first (smallest cosine distance), then "pets" (moderate
  distance), and "dogs" last (largest distance).

RUN FROM: LLM_UNDER_THE_HOOD_COURSE/ (distribution root) directory:
  python lab_resources/LAB2_RAG/test_vectorstore.py

PREREQUISITE: Task 2.4 must be completed (vectorstore.py implemented or copied).
"""

import sys
import os

# Add rag-app to path so we can import ai.vectorstore
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'web-based-course', 'rag-app'))

from ai.vectorstore import VectorStore


def test_vectorstore():
    """Test all VectorStore operations with synthetic embeddings."""

    print("=" * 60)
    print("Task 2.4 — VectorStore Test (Synthetic Embeddings)")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # PART 1: Initialize the vector store
    # ------------------------------------------------------------------
    print("--- Part 1: Initialization ---")
    store = VectorStore(persist_directory='/tmp/test_chromadb_task24')
    print("✓ VectorStore initialized with persistent storage at /tmp/test_chromadb_task24")
    print()

    # ------------------------------------------------------------------
    # PART 2: Add documents with synthetic embeddings
    # ------------------------------------------------------------------
    print("--- Part 2: Adding documents ---")
    print("  Creating synthetic 384-dim embeddings with known directions:")
    print("    vec_cats = [1, 0, 0, ...]  (points along dimension 0)")
    print("    vec_dogs = [0, 1, 0, ...]  (points along dimension 1)")
    print("    vec_pets = [0.7, 0.7, 0, ...] (45° between cats and dogs)")
    print()

    vec_cats = [1.0] + [0.0] * 383          # points along dimension 0
    vec_dogs = [0.0, 1.0] + [0.0] * 382     # points along dimension 1
    vec_pets = [0.7, 0.7] + [0.0] * 382     # between cats and dogs (45 degrees)

    texts = [
        'Cats are independent animals that groom themselves',
        'Dogs are loyal companions that love walks',
        'Pets bring joy and comfort to families',
    ]
    embeddings = [vec_cats, vec_dogs, vec_pets]
    metadatas = [
        {'source': 'test', 'topic': 'cats'},
        {'source': 'test', 'topic': 'dogs'},
        {'source': 'test', 'topic': 'pets'},
    ]

    store.add_documents(
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        collection_name='animals'
    )
    print("✓ 3 documents added to collection 'animals'")
    print()

    # ------------------------------------------------------------------
    # PART 3: Query with a vector close to "cats"
    # ------------------------------------------------------------------
    print("--- Part 3: Querying (vector close to 'cats') ---")
    print("  Query vector: [0.9, 0.1, 0, ...]  (close to cats direction)")
    print()
    print("  NOTE: ChromaDB returns COSINE DISTANCE, not similarity.")
    print("    The cosine formula cos(θ) = (A·B)/(||A||×||B||) computes SIMILARITY (-1 to +1).")
    print("    ChromaDB converts to DISTANCE for sorting: distance = 1 - similarity")
    print("    Distance 0 = identical direction (similarity 1)")
    print("    Distance 1 = orthogonal (similarity 0)")
    print("    Distance 2 = opposite direction (similarity -1)")
    print()

    query_vec = [0.9, 0.1] + [0.0] * 382
    results = store.query(
        query_embedding=query_vec,
        top_k=3,
        collection_name='animals'
    )

    print(f"  Results (top_k=3, sorted by cosine distance — lower = more similar):")
    for i, (text, dist) in enumerate(zip(results['documents'], results['distances'])):
        sim = 1 - dist
        print(f"    {i+1}. distance={dist:.4f}  (similarity={sim:.4f})  {text}")
    print()

    # Verify ordering
    assert results['documents'][0] == texts[0], \
        f"Expected 'cats' document first, got: {results['documents'][0]}"
    print("✓ Correct! 'Cats' document is closest (smallest cosine distance)")
    print()

    # ------------------------------------------------------------------
    # PART 4: Query with a vector close to "dogs"
    # ------------------------------------------------------------------
    print("--- Part 4: Querying (vector close to 'dogs') ---")
    print("  Query vector: [0.1, 0.9, 0, ...]  (close to dogs direction)")
    print()

    query_vec_dogs = [0.1, 0.9] + [0.0] * 382
    results2 = store.query(
        query_embedding=query_vec_dogs,
        top_k=3,
        collection_name='animals'
    )

    print(f"  Results (top_k=3, sorted by cosine distance — lower = more similar):")
    for i, (text, dist) in enumerate(zip(results2['documents'], results2['distances'])):
        sim = 1 - dist
        print(f"    {i+1}. distance={dist:.4f}  (similarity={sim:.4f})  {text}")
    print()

    assert results2['documents'][0] == texts[1], \
        f"Expected 'dogs' document first, got: {results2['documents'][0]}"
    print("✓ Correct! 'Dogs' document is closest when query points toward dim 1")
    print()

    # ------------------------------------------------------------------
    # PART 5: List collections
    # ------------------------------------------------------------------
    print("--- Part 5: List collections ---")
    collections = store.list_collections()
    print(f"  Collections: {collections}")
    assert 'animals' in collections, f"Expected 'animals' in collections, got: {collections}"
    print("✓ Collection 'animals' exists")
    print()

    # ------------------------------------------------------------------
    # PART 6: Delete collection and verify cleanup
    # ------------------------------------------------------------------
    print("--- Part 6: Delete collection ---")
    store.delete_collection('animals')
    collections_after = store.list_collections()
    print(f"  Collections after delete: {collections_after}")
    assert 'animals' not in collections_after, "Collection 'animals' should be deleted"
    print("✓ Collection 'animals' successfully deleted")
    print()

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
    print()
    print("Your VectorStore implementation correctly:")
    print("  ✓ Initializes with persistent ChromaDB storage")
    print("  ✓ Adds documents with embeddings and metadata")
    print("  ✓ Queries return results sorted by cosine distance")
    print("  ✓ Lists existing collections")
    print("  ✓ Deletes collections")
    print()
    print("Next: Proceed to Task 2.5 (RAG Memory) to add conversation")
    print("context and real embeddings from the Ollama model.")


if __name__ == '__main__':
    try:
        test_vectorstore()
    except NotImplementedError as e:
        print()
        print("=" * 60)
        print("TEST FAILED — VectorStore not yet implemented")
        print("=" * 60)
        print()
        print(f"  Error: {e}")
        print()
        print("  You need to complete Task 2.4 first:")
        print("    Option A (easy): cp ../lab_resources/LAB2_RAG/lab2_rag_implementations/vectorstore_full.py rag-app/ai/vectorstore.py")
        print("    Option B (hard): Implement the methods in rag-app/ai/vectorstore.py yourself")
        print()
        print("  Then re-run this test.")
