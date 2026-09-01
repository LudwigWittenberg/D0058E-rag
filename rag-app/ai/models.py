"""
Model selection for the RAG App.

This module handles routing requests to the configured LLM backend
and listing available backends.
"""

import sys
from pathlib import Path

# Ensure shared module is importable
_app_root = Path(__file__).resolve().parent.parent
_project_root = _app_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from shared.llm_client import LLMClient, LLMConfig, LLMBackend  # noqa: E402


# Map string names to LLMBackend enum values
_BACKEND_MAP = {
    "ollama": LLMBackend.OLLAMA,
    "openai": LLMBackend.OPENAI,
    "gemini": LLMBackend.GEMINI,
}


def create_client(backend: str, config: dict) -> LLMClient:
    """
    Create an LLM client for the specified backend.

    Args:
        backend: One of "ollama", "openai", "gemini".
        config: Backend-specific configuration dict. Supported keys:
            - model_name (str): Model to use
            - api_key (str): API key for cloud backends
            - base_url (str): Base URL for Ollama
            - temperature (float): Sampling temperature
            - max_tokens (int): Maximum tokens to generate

    Returns:
        Configured LLMClient instance.

    Raises:
        ValueError: If the backend name is not recognized.
    """
    backend_enum = _BACKEND_MAP.get(backend.lower())
    if backend_enum is None:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            f"Supported backends: {', '.join(_BACKEND_MAP.keys())}"
        )

    # Resolve API key based on backend
    api_key = None
    if backend == "openai":
        api_key = config.get("api_key_openai") or config.get("api_key")
    elif backend == "gemini":
        api_key = config.get("api_key_gemini") or config.get("api_key")
    elif backend == "ollama":
        api_key = config.get("api_key_ollama") or config.get("api_key")

    llm_config = LLMConfig(
        backend=backend_enum,
        model_name=config.get("llm_model", config.get("model_name", "Llama-3.2-3B-Instruct-Q4_K_M")),
        api_key=api_key,
        base_url=config.get("ollama_base_url", config.get("base_url")),
        temperature=config.get("temperature", 0.7),
        max_tokens=config.get("max_tokens", 1024),
    )

    return LLMClient(llm_config)
