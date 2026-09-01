"""
Configuration manager with file persistence and runtime updates.

This module handles loading, saving, and updating application configuration.
It supports both .env files (for initial setup) and a JSON config file
(for runtime changes via the web configuration panel).

Usage:
    from shared.config_manager import ConfigManager
    from pathlib import Path

    config = ConfigManager(Path("./chatbot-app"))
    backend = config.get("llm_backend", "ollama")
    config.set("temperature", 0.9)  # Persists to config.json
"""

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional


CONFIG_FILE = "config.json"

# Mapping from environment variable names to config keys
_ENV_TO_CONFIG = {
    "OPENAI_API_KEY": "api_key_openai",
    "GEMINI_API_KEY": "api_key_gemini",
    "OLLAMA_BASE_URL": "ollama_base_url",
    "LLM_BACKEND": "llm_backend",
    "LLM_MODEL": "llm_model",
    "TEMPERATURE": "temperature",
    "MAX_TOKENS": "max_tokens",
    "SYSTEM_PROMPT": "system_prompt",
    "CHUNK_SIZE": "chunk_size",
    "CHUNK_OVERLAP": "chunk_overlap",
    "TOP_K": "top_k",
    "EMBEDDING_MODEL": "embedding_model",
    "CREW_PROCESS_TYPE": "crew_process_type",
    "MAX_ITERATIONS": "max_iterations",
    "INTEGRATION_MODE": "integration_mode",
}


@dataclass
class AppConfig:
    """Application-specific configuration with defaults."""

    # LLM settings
    llm_backend: str = "ollama"
    llm_model: str = "Llama-3.2-3B-Instruct-Q4_K_M"
    api_key_openai: Optional[str] = None
    api_key_gemini: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 1024

    # Chatbot-specific
    system_prompt: str = "You are a helpful assistant."

    # RAG-specific
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    embedding_model: str = "all-MiniLM-L6-v2"

    # Agents-specific
    crew_process_type: str = "sequential"
    max_iterations: int = 10

    # Mode
    integration_mode: bool = False


def _coerce_value(key: str, value: Any) -> Any:
    """Coerce a value to the correct type based on AppConfig field types."""
    field_types = {f.name: f.type for f in fields(AppConfig)}
    expected_type = field_types.get(key)
    if expected_type is None:
        return value

    # Handle Optional[str] — keep as-is if None or str
    if expected_type is Optional[str]:
        if value is None or value == "":
            return None
        return str(value)

    if expected_type is float:
        try:
            return float(value)
        except (ValueError, TypeError):
            return value
    elif expected_type is int:
        try:
            return int(value)
        except (ValueError, TypeError):
            return value
    elif expected_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)
    elif expected_type is str:
        return str(value)

    return value


class ConfigManager:
    """Manages application configuration with file persistence."""

    def __init__(self, app_root: Path):
        """
        Initialize the config manager.

        Args:
            app_root: Path to the application root directory.
                      config.json is stored here.
        """
        self.app_root = Path(app_root)
        self.config_path = self.app_root / CONFIG_FILE
        self._config: dict = {}
        self._load()

    def _load(self) -> None:
        """Load config from JSON file, falling back to .env and defaults.

        Loading order (later overrides earlier):
        1. AppConfig defaults
        2. .env file values (via python-dotenv if available, else os.environ)
        3. config.json values
        4. Environment variable overrides (explicit env vars override everything)
        """
        # Start with defaults from AppConfig
        defaults = asdict(AppConfig())
        self._config = dict(defaults)

        # Try loading .env file (best-effort, python-dotenv may not be installed)
        env_path = self.app_root / ".env"
        if env_path.exists():
            try:
                from dotenv import dotenv_values

                env_values = dotenv_values(env_path)
                for env_key, config_key in _ENV_TO_CONFIG.items():
                    if env_key in env_values and env_values[env_key]:
                        self._config[config_key] = _coerce_value(
                            config_key, env_values[env_key]
                        )
            except ImportError:
                # python-dotenv not available, skip .env file parsing
                pass

        # Load from config.json if it exists
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    json_config = json.load(f)
                for key, value in json_config.items():
                    self._config[key] = _coerce_value(key, value)
            except (json.JSONDecodeError, OSError):
                # If config.json is corrupted, keep defaults
                pass

        # Apply environment variable overrides last
        self.apply_env_overrides()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: The configuration key to look up.
            default: Value to return if key is not found.

        Returns:
            The configuration value, or default if not found.
        """
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value and persist to file.

        Args:
            key: The configuration key to set.
            value: The value to store.
        """
        self._config[key] = _coerce_value(key, value)
        self.save()

    def get_all(self) -> dict:
        """Return all configuration as a dictionary.

        Returns:
            A copy of the full configuration dictionary.
        """
        return dict(self._config)

    def save(self) -> None:
        """Persist current configuration to config.json."""
        # Ensure the directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=2, default=str)

    def apply_env_overrides(self) -> None:
        """Apply environment variable overrides.

        Checks os.environ for known configuration variables and applies
        them to the current config. This allows environment variables to
        take precedence over file-based configuration.
        """
        for env_key, config_key in _ENV_TO_CONFIG.items():
            env_value = os.environ.get(env_key)
            if env_value is not None and env_value != "":
                self._config[config_key] = _coerce_value(config_key, env_value)
