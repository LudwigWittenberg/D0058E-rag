"""
General utility functions shared across all applications.

This module provides common helper functions used by the web layer
and AI modules across all three applications.
"""


def format_timestamp(dt=None) -> str:
    """
    Format a datetime object as ISO 8601 string.

    Args:
        dt: datetime object (defaults to current UTC time)

    Returns:
        ISO 8601 formatted timestamp string
    """
    pass


def truncate_text(text: str, max_length: int = 200) -> str:
    """
    Truncate text to a maximum length with ellipsis.

    Args:
        text: Input text to truncate
        max_length: Maximum character length

    Returns:
        Truncated text with '...' suffix if shortened
    """
    pass


def safe_json_loads(text: str, default=None):
    """
    Safely parse a JSON string, returning a default on failure.

    Args:
        text: JSON string to parse
        default: Value to return if parsing fails

    Returns:
        Parsed JSON object or default value
    """
    pass
