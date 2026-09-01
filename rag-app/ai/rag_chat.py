"""
RAG Memory Chat — generation logic.

This module implements the chat generation flow using RAG memory:
1. Retrieve relevant past messages from the vector store
2. Build an augmented prompt with retrieved context
3. Generate a response using the LLM
4. Store both user message and response in the vector store

Students can experiment with:
- Different prompt templates for context injection
- How many retrieved messages to include (top_k)
- Minimum similarity threshold (min_score)
- The effect of context on response quality
"""

import sys
from pathlib import Path

_app_root = Path(__file__).resolve().parent.parent
_project_root = _app_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from shared.llm_client import LLMClient, LLMResponse  # noqa: E402
from ai.rag_memory import RAGMemory  # noqa: E402
from ai.rag_prompts import build_rag_memory_prompt, RAG_MEMORY_SYSTEM_PROMPT  # noqa: E402


def generate_rag_response(
    message: str,
    memory: RAGMemory,
    llm_client: LLMClient,
    top_k: int = 3,
    min_score: float = 0.0,
) -> dict:
    """
    Generate a chat response using RAG memory as context.

    Flow:
    1. Retrieve relevant past messages from memory
    2. Build prompt with retrieved context
    3. Call LLM to generate response
    4. Store both user message and response in memory

    Args:
        message: The user's current message.
        memory: RAGMemory instance (vector store).
        llm_client: Configured LLM client.
        top_k: How many past messages to retrieve.
        min_score: Minimum similarity score for retrieval.

    Returns:
        {
            "response": str,
            "retrieved_context": list[dict],
            "store_contents": list[dict],
            "prompt_context": str,
            "stats": dict
        }
    """
    # Step 1: Retrieve relevant past messages
    retrieved_context = memory.retrieve_context(
        query=message,
        top_k=top_k,
        min_score=min_score,
    )

    # Step 2: Build augmented prompt
    prompt = build_rag_memory_prompt(message, retrieved_context)

    # Step 3: Generate response
    response: LLMResponse = llm_client.generate(
        prompt=prompt,
        system_message=RAG_MEMORY_SYSTEM_PROMPT,
    )
    response_text = response.text

    # Step 4: Store both messages in memory
    memory.add_message("user", message)
    memory.add_message("assistant", response_text)

    # Step 5: Get current store state for display
    store_contents = memory.get_all_messages()

    # Estimate token usage
    prompt_tokens = len(prompt) // 4
    response_tokens = len(response_text) // 4
    context_tokens = sum(len(r["text"]) for r in retrieved_context) // 4

    return {
        "response": response_text,
        "retrieved_context": retrieved_context,
        "store_contents": store_contents,
        "prompt_context": prompt,
        "stats": {
            "total_messages": memory.get_message_count(),
            "retrieved_count": len(retrieved_context),
            "top_k": top_k,
            "min_score": min_score,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "context_tokens": context_tokens,
            "total_tokens_this_turn": prompt_tokens + response_tokens,
        },
    }
