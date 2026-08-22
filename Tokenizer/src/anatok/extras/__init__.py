"""Extras module for anatok.

Additional utility functions and extended features."""
import json
import base64
import os

from ..utils import ensure_dir

def encode_token_to_base64(token) -> str:
    """Encode a token to base64 string."""
    if isinstance(token, str):
        token = token.encode("utf-8")
    return base64.b64encode(token).decode("ascii")

def decode_base64_to_token(encoded: str):
    """Decode a base64 string back to token."""
    return base64.b64decode(encoded)

def save_session(results: dict, filepath: str) -> bool:
    """Save operation results to a JSON session file.

    Args:
        results: Dictionary of results to save.
        filepath: Path to save the session file.

    Returns:
        True if saved successfully.
    """
    try:
        ensure_dir(filepath)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        return True
    except Exception:
        return False

def load_session(filepath: str) -> dict:
    """Load operation results from a JSON session file.

    Args:
        filepath: Path to the session file.

    Returns:
        Dictionary of loaded results.
    """
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def generate_token_report(results: dict) -> str:
    """Generate a human-readable tokenization report.

    Args:
        results: Dictionary of tokenization results.

    Returns:
        Formatted report string.
    """
    lines = [
        "=== Tokenization Report ===",
        f"Original tokens: {results.get('original_tokens', 'N/A')}",
        f"Optimized tokens: {results.get('optimized_tokens', 'N/A')}",
        f"Duplicates removed: {results.get('duplicates_removed', 'N/A')}",
        f"Compression ratio: {results.get('compression_ratio', 'N/A'):.2%}",
        "===========================",
    ]
    return "\n".join(lines)
