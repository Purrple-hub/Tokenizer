"""Decompression modules for the Tokenizer project.

Reverse operations of compression algorithms. Matches the corrected
formats documented in compressors.py.
"""

import zlib

def zlib_decompress(data: bytes) -> bytes:
    """Decompress zlib data.

    Args:
        data: Compressed bytes.

    Returns:
        Decompressed bytes.
    """
    return zlib.decompress(data)

def simple_decompress(data: bytes) -> bytes:
    """Decompress data compressed with simple_compress.

    Args:
        data: Compressed bytes.

    Returns:
        Decompressed bytes.
    """
    if len(data) < 2 or len(data) % 2 != 0:
        return bytes(data)

    result = bytearray()
    i = 0
    n = len(data)
    while i < n:
        count = data[i]
        byte_val = data[i + 1]
        result.extend(bytes([byte_val]) * count)
        i += 2
    return bytes(result)

def lz77_decompress(data: bytes, window_size: int = 65535) -> bytes:
    """Decompress LZ77 v2 data.

    Format:
        Literal: flag byte 0x80 followed by one verbatim byte.
        Match:   flag byte 0x00, u16 LE offset, u8 length. Overlapping
                 matches are copied byte-by-byte as the output grows,
                 per LZ77 semantics.

    Args:
        data: Compressed bytes.
        window_size: Kept for signature compatibility; offsets are now
            validated against actual output size instead of being clamped
            (the old clamp silently corrupted data).

    Returns:
        Decompressed bytes.
    """
    out = bytearray()
    pos = 0
    n = len(data)

    while pos < n:
        flag = data[pos]
        pos += 1

        if flag == 0x80:
            if pos >= n:
                break
            out.append(data[pos])
            pos += 1
        elif flag == 0x00:
            if pos + 3 > n:
                break
            offset = int.from_bytes(data[pos:pos + 2], "little")
            length = data[pos + 2]
            pos += 3

            current_len = len(out)
            if offset == 0 or offset > current_len:
                break

            start = current_len - offset
            for i in range(length):
                out.append(out[start + i])
        else:
            break

    return bytes(out)

def huffman_decompress(compressed: dict) -> bytes:
    """Decompress data compressed with huffman_compress.

    Args:
        compressed: Dict with "compressed_bytes", "code_table" and
            "padding" produced by huffman_compress.

    Returns:
        Decompressed bytes.
    """
    code_table = compressed.get("code_table") or {}
    raw = compressed.get("compressed_bytes")
    if raw is None:
        raise ValueError(
            "huffman payload missing 'compressed_bytes'; "
            "pass the dict returned by huffman_compress()"
        )
    if not raw:
        return b""

    reverse = {}
    for symbol, code in code_table.items():
        if code:
            reverse[(len(code), int(code, 2))] = symbol

    padding = int(compressed.get("padding", 0) or 0)
    total_bits = len(raw) * 8 - padding

    out = bytearray()
    cur = 0
    cur_len = 0
    consumed = 0

    for byte in raw:
        if consumed >= total_bits:
            break
        for shift in range(7, -1, -1):
            if consumed >= total_bits:
                break
            cur = (cur << 1) | ((byte >> shift) & 1)
            cur_len += 1
            consumed += 1
            symbol = reverse.get((cur_len, cur))
            if symbol is not None:
                out.append(symbol)
                cur = 0
                cur_len = 0

    return bytes(out)

def decompress_data(data: bytes, method: str = "zlib") -> bytes:
    """Decompress data using the specified method.

    Args:
        data: Compressed bytes (for "huffman", pass the dict returned by
            huffman_compress).
        method: "zlib" (default), "simple", "lz77" or "huffman".

    Returns:
        Decompressed bytes.
    """
    if method == "zlib":
        return zlib_decompress(data)
    elif method == "simple":
        return simple_decompress(data)
    elif method == "lz77":
        return lz77_decompress(data)
    elif method == "huffman":
        if isinstance(data, dict):
            return huffman_decompress(data)
        raise ValueError(
            "huffman decompression requires the dict produced by "
            "huffman_compress(), not raw bytes"
        )
    else:
        raise ValueError(f"Unknown compression method: {method}")
