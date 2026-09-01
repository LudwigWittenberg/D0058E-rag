"""
Answer generation module for the RAG pipeline.

Constructs augmented prompts from retrieved context and
generates answers using an LLM backend. This is the "G" in RAG —
the generation step that produces answers grounded in retrieved documents.

Students can experiment with:
- Different prompt templates and instructions
- Context formatting strategies
- Source attribution approaches
- Temperature and max_tokens settings
"""

import sys
from pathlib import Path

# Ensure shared module is importable
_app_root = Path(__file__).resolve().parent.parent
_project_root = _app_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from shared.llm_client import LLMClient, LLMResponse  # noqa: E402


# Default RAG prompt template
_RAG_PROMPT_TEMPLATE = """Answer the following question based on the provided context.
If the context doesn't contain enough information to answer the question,
say so clearly rather than making up an answer.

Context:
{context}

Question: {query}

Answer:"""


def build_augmented_prompt(query: str, context_chunks: list) -> str:
    """
    Construct an augmented prompt from query and retrieved context.

    The resulting prompt contains both the query text and all context chunks,
    formatted in a way that instructs the LLM to answer based on the context.

    Args:
        query: The user's question.
        context_chunks: List of relevant text chunks (strings).

    Returns:
        Formatted prompt string containing context and query.
    """
    # Format context chunks with separators for clarity
    if context_chunks:
        formatted_context = "\n\n---\n\n".join(
            f"[Source {i + 1}]: {chunk}"
            for i, chunk in enumerate(context_chunks)
        )
    else:
        formatted_context = "(No relevant context found)"

    prompt = _RAG_PROMPT_TEMPLATE.format(
        context=formatted_context,
        query=query,
    )

    return prompt


def generate_answer(query: str, context_chunks: list, llm_client=None) -> dict:
    """
    Generate an answer using retrieved context.

    Builds an augmented prompt from the query and context chunks,
    then sends it to the LLM for generation.

    Args:
        query: The user's question.
        context_chunks: Retrieved relevant chunks (list of strings).
        llm_client: Configured LLM client instance (LLMClient).

    Returns:
        {"answer": str, "sources": list[str], "model": str}
    """
    # Build the augmented prompt
    prompt = build_augmented_prompt(query, context_chunks)

    # System message for RAG generation
    system_message = (
        "You are a helpful assistant that answers questions based on provided context. "
        "Always cite which source(s) you used. If the context is insufficient, "
        "acknowledge the limitation."
    )

    if llm_client is None:
        # Return a placeholder if no LLM client is configured
        return {
            "answer": "LLM client not configured. Please set up an LLM backend.",
            "sources": context_chunks,
            "model": "none",
        }

    # Generate the answer using the LLM
    response: LLMResponse = llm_client.generate(
        prompt=prompt,
        system_message=system_message,
    )

    return {
        "answer": response.text,
        "sources": context_chunks,
        "model": response.model,
    }



