"""Utility functions for the Tokenizer project."""

import hashlib
import os
import json
from pathlib import Path

def compute_hash(data: bytes) -> str:
    """Compute SHA-256 hash of data."""
    return hashlib.sha256(data).hexdigest()

def read_file_content(filepath: str) -> bytes:
    """Read file content as bytes."""
    with open(filepath, "rb") as f:
        return f.read()

DEFAULT_BLOCK_SIZE = 4 * 1024 * 1024

def iter_file_blocks(filepath: str, block_size: int = DEFAULT_BLOCK_SIZE):
    """Yield file contents block by block without loading the whole file.

    Args:
        filepath: Path to the file to read.
        block_size: Bytes per yielded block.

    Yields:
        bytes blocks of up to block_size.
    """
    with open(filepath, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                return
            yield block

def read_file_text(filepath: str, encoding: str = "utf-8") -> str:
    """Read file content as text."""
    with open(filepath, "r", encoding=encoding, errors="replace") as f:
        return f.read()

def write_file_content(filepath: str, data: bytes) -> None:
    """Write bytes data to file."""
    with open(filepath, "wb") as f:
        f.write(data)

def ensure_dir(filepath: str) -> None:
    """Ensure parent directory exists, create if needed."""
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)

def format_bytes(size: int) -> str:
    """Format byte size to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"

def validate_path(path: str) -> bool:
    """Validate that a path exists and is accessible."""
    return os.path.exists(path)

def count_tokens(text: str) -> int:
    """Count approximate tokens in text (simple word-based count)."""
    return len(text.split())

def truncate_text(text: str, max_chars: int = 1000) -> str:
    """Truncate text to maximum characters."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."

def get_file_info(filepath: str) -> dict:
    """Get file information (size, hash, etc.)."""
    if not os.path.exists(filepath):
        return {}
    stat = os.stat(filepath)
    return {
        "size": stat.st_size,
        "hash": compute_hash(read_file_content(filepath)),
        "modified": stat.st_mtime,
    }
