"""Extras compression utilities."""

import zlib

def zlib_compress(data: bytes) -> bytes:
    """Compress data using Python's zlib module."""
    return zlib.compress(data)

def zlib_decompress(data: bytes) -> bytes:
    """Decompress data using Python's zlib module."""
    return zlib.decompress(data)

def zlib_ratio(data: bytes) -> float:
    """Calculate compression ratio using zlib."""
    compressed = zlib_compress(data)
    return len(compressed) / len(data) if len(data) > 0 else 1.0
