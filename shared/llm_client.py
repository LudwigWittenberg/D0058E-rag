"""
Unified LLM client supporting Ollama, OpenAI, and Gemini backends.

This module provides a single interface for all three applications to interact
with different LLM providers. It abstracts away provider-specific API differences
behind a common generate() method.

Usage:
    from shared.llm_client import LLMClient, LLMConfig, LLMBackend

    config = LLMConfig(backend=LLMBackend.OLLAMA, model_name="Llama-3.2-3B-Instruct-Q4_K_M")
    client = LLMClient(config)
    response = client.generate("Hello, world!")
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum

import requests


class LLMBackend(Enum):
    """Supported LLM backend providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    GEMINI = "gemini"


@dataclass
class LLMConfig:
    """Configuration for an LLM backend."""
    backend: LLMBackend
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # For Ollama: http://localhost:11434
    temperature: float = 0.7
    max_tokens: int = 1024


@dataclass
class LLMResponse:
    """Standardized response from any LLM backend."""
    text: str
    model: str
    backend: str
    usage: dict  # {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}


class LLMClientError(Exception):
    """Raised when an LLM backend call fails."""
    def __init__(self, backend: str, reason: str):
        self.backend = backend
        self.reason = reason
        super().__init__(f"LLM backend '{backend}' failed: {reason}")


class LLMClient:
    """Unified LLM client supporting Ollama, OpenAI, and Gemini."""

    def __init__(self, config: LLMConfig):
        """Initialize the LLM client with the given configuration."""
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration for the selected backend."""
        if self.config.backend == LLMBackend.OLLAMA:
            if not self.config.base_url:
                self.config.base_url = "http://localhost:11434"
        elif self.config.backend in (LLMBackend.OPENAI, LLMBackend.GEMINI):
            if not self.config.api_key:
                raise LLMClientError(
                    self.config.backend.value,
                    "API key required for cloud backends"
                )

    def generate(self, prompt: str, system_message: Optional[str] = None) -> LLMResponse:
        """
        Generate a response from the configured LLM backend.

        Args:
            prompt: The user prompt to send
            system_message: Optional system message for context

        Returns:
            LLMResponse with generated text and metadata

        Raises:
            LLMClientError: If the backend is unavailable or returns an error
        """
        if self.config.backend == LLMBackend.OLLAMA:
            return self._generate_ollama(prompt, system_message)
        elif self.config.backend == LLMBackend.OPENAI:
            return self._generate_openai(prompt, system_message)
        elif self.config.backend == LLMBackend.GEMINI:
            return self._generate_gemini(prompt, system_message)
        else:
            raise LLMClientError(
                self.config.backend.value,
                f"Unsupported backend: {self.config.backend.value}"
            )

    def generate_chat(
        self,
        messages: list,
        system_message: Optional[str] = None
    ) -> LLMResponse:
        """
        Generate a response from a conversation history.

        Args:
            messages: List of message dicts with 'role' and 'content'
            system_message: Optional system message prepended to history

        Returns:
            LLMResponse with generated text and metadata
        """
        if self.config.backend == LLMBackend.OLLAMA:
            return self._chat_ollama(messages, system_message)
        elif self.config.backend == LLMBackend.OPENAI:
            return self._chat_openai(messages, system_message)
        elif self.config.backend == LLMBackend.GEMINI:
            return self._chat_gemini(messages, system_message)
        else:
            raise LLMClientError(
                self.config.backend.value,
                f"Unsupported backend: {self.config.backend.value}"
            )

    def test_connection(self) -> dict:
        """
        Test connectivity to the configured backend.

        Returns:
            {"status": "connected"|"error", "backend": str, "model": str, "message": str}
        """
        try:
            if self.config.backend == LLMBackend.OLLAMA:
                return self._test_ollama()
            elif self.config.backend == LLMBackend.OPENAI:
                return self._test_openai()
            elif self.config.backend == LLMBackend.GEMINI:
                return self._test_gemini()
            else:
                return {
                    "status": "error",
                    "backend": self.config.backend.value,
                    "model": self.config.model_name,
                    "message": f"Unsupported backend: {self.config.backend.value}"
                }
        except LLMClientError as e:
            return {
                "status": "error",
                "backend": self.config.backend.value,
                "model": self.config.model_name,
                "message": e.reason
            }
        except Exception as e:
            return {
                "status": "error",
                "backend": self.config.backend.value,
                "model": self.config.model_name,
                "message": str(e)
            }

    @staticmethod
    def list_available_models(backend: LLMBackend, config: dict) -> list:
        """
        List available models for a given backend.

        Args:
            backend: The LLM backend to query
            config: Backend-specific config (api_key, base_url)

        Returns:
            List of model name strings
        """
        try:
            if backend == LLMBackend.OLLAMA:
                return LLMClient._list_models_ollama(config)
            elif backend == LLMBackend.OPENAI:
                return LLMClient._list_models_openai(config)
            elif backend == LLMBackend.GEMINI:
                return LLMClient._list_models_gemini(config)
            else:
                return []
        except Exception as e:
            raise LLMClientError(
                backend.value,
                f"Failed to list models: {str(e)}"
            )

    # -------------------------------------------------------------------------
    # Ollama backend implementation
    # -------------------------------------------------------------------------

    def _generate_ollama(self, prompt: str, system_message: Optional[str] = None) -> LLMResponse:
        """Generate a response using Ollama's /api/generate endpoint."""
        url = f"{self.config.base_url}/api/generate"
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            }
        }
        if system_message:
            payload["system"] = system_message

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            data = response.json()
        except requests.ConnectionError:
            raise LLMClientError(
                "ollama",
                f"Cannot connect to Ollama at {self.config.base_url}. Is Ollama running?"
            )
        except requests.Timeout:
            raise LLMClientError("ollama", "Request timed out")
        except requests.HTTPError as e:
            raise LLMClientError("ollama", f"HTTP error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise LLMClientError("ollama", f"Unexpected error: {str(e)}")

        return LLMResponse(
            text=data.get("response", ""),
            model=data.get("model", self.config.model_name),
            backend="ollama",
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            }
        )

    def _chat_ollama(self, messages: list, system_message: Optional[str] = None) -> LLMResponse:
        """Generate a chat response using Ollama's /api/chat endpoint."""
        url = f"{self.config.base_url}/api/chat"

        chat_messages = []
        if system_message:
            chat_messages.append({"role": "system", "content": system_message})
        chat_messages.extend(messages)

        payload = {
            "model": self.config.model_name,
            "messages": chat_messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            }
        }

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            data = response.json()
        except requests.ConnectionError:
            raise LLMClientError(
                "ollama",
                f"Cannot connect to Ollama at {self.config.base_url}. Is Ollama running?"
            )
        except requests.Timeout:
            raise LLMClientError("ollama", "Request timed out")
        except requests.HTTPError as e:
            raise LLMClientError("ollama", f"HTTP error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise LLMClientError("ollama", f"Unexpected error: {str(e)}")

        message_data = data.get("message", {})
        return LLMResponse(
            text=message_data.get("content", ""),
            model=data.get("model", self.config.model_name),
            backend="ollama",
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            }
        )

    def _test_ollama(self) -> dict:
        """Test connection to Ollama backend."""
        url = f"{self.config.base_url}/api/tags"
        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            if self.config.model_name in models or any(
                m.startswith(self.config.model_name) for m in models
            ):
                return {
                    "status": "connected",
                    "backend": "ollama",
                    "model": self.config.model_name,
                    "message": f"Connected to Ollama. Model '{self.config.model_name}' available."
                }
            else:
                return {
                    "status": "connected",
                    "backend": "ollama",
                    "model": self.config.model_name,
                    "message": (
                        f"Connected to Ollama but model '{self.config.model_name}' not found. "
                        f"Available models: {', '.join(models) if models else 'none'}"
                    )
                }
        except requests.ConnectionError:
            raise LLMClientError(
                "ollama",
                f"Cannot connect to Ollama at {self.config.base_url}. Is Ollama running?"
            )
        except Exception as e:
            raise LLMClientError("ollama", f"Connection test failed: {str(e)}")

    @staticmethod
    def _list_models_ollama(config: dict) -> list:
        """List available Ollama models."""
        base_url = config.get("base_url", "http://localhost:11434")
        url = f"{base_url}/api/tags"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except requests.ConnectionError:
            raise LLMClientError(
                "ollama",
                f"Cannot connect to Ollama at {base_url}. Is Ollama running?"
            )
        except Exception as e:
            raise LLMClientError("ollama", f"Failed to list models: {str(e)}")

    # -------------------------------------------------------------------------
    # OpenAI backend implementation
    # -------------------------------------------------------------------------

    def _generate_openai(self, prompt: str, system_message: Optional[str] = None) -> LLMResponse:
        """Generate a response using OpenAI's ChatCompletion API."""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        return self._chat_openai(messages, system_message=None)

    def _chat_openai(self, messages: list, system_message: Optional[str] = None) -> LLMResponse:
        """Generate a chat response using OpenAI's ChatCompletion API."""
        try:
            import openai
        except ImportError:
            raise LLMClientError("openai", "openai package not installed. Run: pip install openai")

        chat_messages = []
        if system_message:
            chat_messages.append({"role": "system", "content": system_message})
        chat_messages.extend(messages)

        try:
            client = openai.OpenAI(api_key=self.config.api_key)
            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=chat_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        except openai.AuthenticationError:
            raise LLMClientError("openai", "Invalid API key. Check your OPENAI_API_KEY.")
        except openai.RateLimitError:
            raise LLMClientError("openai", "Rate limit exceeded. Please wait and try again.")
        except openai.APIConnectionError:
            raise LLMClientError("openai", "Cannot connect to OpenAI API. Check your network.")
        except openai.BadRequestError as e:
            raise LLMClientError("openai", f"Bad request: {str(e)}")
        except Exception as e:
            raise LLMClientError("openai", f"Unexpected error: {str(e)}")

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            text=choice.message.content or "",
            model=response.model,
            backend="openai",
            usage={
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            }
        )

    def _test_openai(self) -> dict:
        """Test connection to OpenAI backend."""
        try:
            import openai
        except ImportError:
            raise LLMClientError("openai", "openai package not installed. Run: pip install openai")

        try:
            client = openai.OpenAI(api_key=self.config.api_key)
            models = client.models.list()
            model_ids = [m.id for m in models.data]
            if self.config.model_name in model_ids:
                return {
                    "status": "connected",
                    "backend": "openai",
                    "model": self.config.model_name,
                    "message": f"Connected to OpenAI. Model '{self.config.model_name}' available."
                }
            else:
                return {
                    "status": "connected",
                    "backend": "openai",
                    "model": self.config.model_name,
                    "message": (
                        f"Connected to OpenAI but model '{self.config.model_name}' "
                        f"may not be available in your account."
                    )
                }
        except openai.AuthenticationError:
            raise LLMClientError("openai", "Invalid API key. Check your OPENAI_API_KEY.")
        except Exception as e:
            raise LLMClientError("openai", f"Connection test failed: {str(e)}")

    @staticmethod
    def _list_models_openai(config: dict) -> list:
        """List available OpenAI models."""
        try:
            import openai
        except ImportError:
            raise LLMClientError("openai", "openai package not installed. Run: pip install openai")

        api_key = config.get("api_key")
        if not api_key:
            raise LLMClientError("openai", "API key required to list models")

        try:
            client = openai.OpenAI(api_key=api_key)
            models = client.models.list()
            return sorted([m.id for m in models.data])
        except openai.AuthenticationError:
            raise LLMClientError("openai", "Invalid API key. Check your OPENAI_API_KEY.")
        except Exception as e:
            raise LLMClientError("openai", f"Failed to list models: {str(e)}")

    # -------------------------------------------------------------------------
    # Gemini backend implementation
    # -------------------------------------------------------------------------

    def _generate_gemini(self, prompt: str, system_message: Optional[str] = None) -> LLMResponse:
        """Generate a response using Google Gemini API."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise LLMClientError(
                "gemini",
                "google-generativeai package not installed. Run: pip install google-generativeai"
            )

        try:
            genai.configure(api_key=self.config.api_key)

            generation_config = genai.types.GenerationConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
            )

            model = genai.GenerativeModel(
                model_name=self.config.model_name,
                system_instruction=system_message if system_message else None,
                generation_config=generation_config,
            )

            response = model.generate_content(prompt)
        except Exception as e:
            error_msg = str(e)
            if "API_KEY_INVALID" in error_msg or "PERMISSION_DENIED" in error_msg:
                raise LLMClientError("gemini", "Invalid API key. Check your GEMINI_API_KEY.")
            elif "RESOURCE_EXHAUSTED" in error_msg:
                raise LLMClientError("gemini", "Rate limit exceeded. Please wait and try again.")
            elif "NOT_FOUND" in error_msg:
                raise LLMClientError(
                    "gemini",
                    f"Model '{self.config.model_name}' not found."
                )
            else:
                raise LLMClientError("gemini", f"Unexpected error: {error_msg}")

        text = ""
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                text = candidate.content.parts[0].text or ""

        # Gemini usage metadata
        usage_metadata = getattr(response, "usage_metadata", None)
        usage = {
            "prompt_tokens": getattr(usage_metadata, "prompt_token_count", 0) if usage_metadata else 0,
            "completion_tokens": getattr(usage_metadata, "candidates_token_count", 0) if usage_metadata else 0,
            "total_tokens": getattr(usage_metadata, "total_token_count", 0) if usage_metadata else 0,
        }

        return LLMResponse(
            text=text,
            model=self.config.model_name,
            backend="gemini",
            usage=usage,
        )

    def _chat_gemini(self, messages: list, system_message: Optional[str] = None) -> LLMResponse:
        """Generate a chat response using Google Gemini API."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise LLMClientError(
                "gemini",
                "google-generativeai package not installed. Run: pip install google-generativeai"
            )

        try:
            genai.configure(api_key=self.config.api_key)

            generation_config = genai.types.GenerationConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
            )

            model = genai.GenerativeModel(
                model_name=self.config.model_name,
                system_instruction=system_message if system_message else None,
                generation_config=generation_config,
            )

            # Convert messages to Gemini chat format
            chat = model.start_chat(history=[])

            # Send all messages except the last one as history
            for msg in messages[:-1]:
                role = "user" if msg["role"] == "user" else "model"
                chat.history.append(
                    genai.types.ContentDict(role=role, parts=[msg["content"]])
                )

            # Send the last message to get a response
            last_message = messages[-1]["content"] if messages else ""
            response = chat.send_message(last_message)

        except Exception as e:
            error_msg = str(e)
            if "API_KEY_INVALID" in error_msg or "PERMISSION_DENIED" in error_msg:
                raise LLMClientError("gemini", "Invalid API key. Check your GEMINI_API_KEY.")
            elif "RESOURCE_EXHAUSTED" in error_msg:
                raise LLMClientError("gemini", "Rate limit exceeded. Please wait and try again.")
            elif "NOT_FOUND" in error_msg:
                raise LLMClientError(
                    "gemini",
                    f"Model '{self.config.model_name}' not found."
                )
            else:
                raise LLMClientError("gemini", f"Unexpected error: {error_msg}")

        text = ""
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                text = candidate.content.parts[0].text or ""

        usage_metadata = getattr(response, "usage_metadata", None)
        usage = {
            "prompt_tokens": getattr(usage_metadata, "prompt_token_count", 0) if usage_metadata else 0,
            "completion_tokens": getattr(usage_metadata, "candidates_token_count", 0) if usage_metadata else 0,
            "total_tokens": getattr(usage_metadata, "total_token_count", 0) if usage_metadata else 0,
        }

        return LLMResponse(
            text=text,
            model=self.config.model_name,
            backend="gemini",
            usage=usage,
        )

    def _test_gemini(self) -> dict:
        """Test connection to Gemini backend."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise LLMClientError(
                "gemini",
                "google-generativeai package not installed. Run: pip install google-generativeai"
            )

        try:
            genai.configure(api_key=self.config.api_key)
            models = genai.list_models()
            model_names = [m.name for m in models]
            # Gemini model names are like "models/gemini-pro"
            full_name = f"models/{self.config.model_name}"
            if full_name in model_names or self.config.model_name in model_names:
                return {
                    "status": "connected",
                    "backend": "gemini",
                    "model": self.config.model_name,
                    "message": f"Connected to Gemini. Model '{self.config.model_name}' available."
                }
            else:
                return {
                    "status": "connected",
                    "backend": "gemini",
                    "model": self.config.model_name,
                    "message": (
                        f"Connected to Gemini but model '{self.config.model_name}' "
                        f"may not be available."
                    )
                }
        except Exception as e:
            error_msg = str(e)
            if "API_KEY_INVALID" in error_msg or "PERMISSION_DENIED" in error_msg:
                raise LLMClientError("gemini", "Invalid API key. Check your GEMINI_API_KEY.")
            raise LLMClientError("gemini", f"Connection test failed: {error_msg}")

    @staticmethod
    def _list_models_gemini(config: dict) -> list:
        """List available Gemini models."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise LLMClientError(
                "gemini",
                "google-generativeai package not installed. Run: pip install google-generativeai"
            )

        api_key = config.get("api_key")
        if not api_key:
            raise LLMClientError("gemini", "API key required to list models")

        try:
            genai.configure(api_key=api_key)
            models = genai.list_models()
            return [m.name.replace("models/", "") for m in models
                    if "generateContent" in (m.supported_generation_methods or [])]
        except Exception as e:
            error_msg = str(e)
            if "API_KEY_INVALID" in error_msg or "PERMISSION_DENIED" in error_msg:
                raise LLMClientError("gemini", "Invalid API key. Check your GEMINI_API_KEY.")
            raise LLMClientError("gemini", f"Failed to list models: {error_msg}")
