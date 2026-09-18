"""
Prompt templates for RAG Memory Chat.

This module defines how retrieved context is injected into the LLM prompt.
Students implement build_rag_memory_prompt() to format retrieved memories
and combine them with the user's current message.
"""

# System prompt for RAG memory chat
RAG_MEMORY_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to conversation memory. "
    "Relevant past messages from this conversation are provided as retrieved memories. "
    "Use them to maintain context and recall previously mentioned facts. "
    "If the memories contain relevant information, incorporate it naturally into your response. "
    "If the memories are not relevant to the current question, you may ignore them."
)


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
