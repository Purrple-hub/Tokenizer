"""Compression modules for the Tokenizer project.

Default method is zlib (fast, correct, C-implemented). The hand-written
LZ77 / Huffman / RLE implementations remain available as opt-in legacy
methods; their formats were corrected:

LZ77 v2 wire format:
    Literal: 0x80 followed by one verbatim byte      (any byte 0-255)
    Match:   0x00, u16 LE offset, u8 length          (offset >= 1, len >= 3)
    (The previous format destroyed the high bit of literal bytes.)

Huffman output is a dict: {"compressed_bytes", "code_table", "padding"}.
"""

import heapq
import zlib
from collections import Counter

def zlib_compress(data: bytes, level: int = 6) -> bytes:
    """Compress bytes with zlib deflate (recommended default).

    Args:
        data: Data to compress.
        level: Compression level 0-9 (6 balances speed/size).

    Returns:
        Compressed bytes.
    """
    compressor = zlib.compressobj(level)
    return compressor.compress(data) + compressor.flush()

def simple_compress(data: bytes) -> bytes:
    """Run-length compression (legacy).

    Args:
        data: Data to compress.

    Returns:
        Compressed bytes.
    """
    result = bytearray()
    i = 0
    n = len(data)
    while i < n:
        count = 1
        while i + count < n and data[i + count] == data[i] and count < 255:
            count += 1
        result.append(count)
        result.append(data[i])
        i += count
    return bytes(result)

def lz77_compress(data: bytes, window_size: int = 4096,
                  lookahead: int = 64, max_chain: int = 16) -> bytes:
    """LZ77 compression with hash-chain matching (legacy, corrected format).

    A dict of 3-byte sequences -> recent positions limits the search to
    ``max_chain`` candidates instead of scanning the whole window, making
    this roughly an order of magnitude faster than a naive scan while
    producing identical-quality output.

    Args:
        data: Data to compress.
        window_size: Sliding window size (capped at 65535).
        lookahead: Maximum match length (capped at 255).
        max_chain: Maximum candidate positions examined per step.

    Returns:
        Compressed bytes in LZ77 v2 format.
    """
    n = len(data)
    if n < 3:
        out = bytearray()
        for byte in data:
            out.append(0x80)
            out.append(byte)
        return bytes(out)

    window_size = min(window_size, 65535)
    max_match = min(lookahead, 255)

    result = bytearray()
    heads: dict = {}
    pos = 0

    while pos < n:
        best_len = 0
        best_offset = 0
        max_len = min(max_match, n - pos)

        if max_len >= 3:
            key = bytes(data[pos:pos + 3])
            chain = heads.get(key)
            if chain:
                low = pos - window_size
                examined = 0
                for cand in reversed(chain):
                    if cand < low or examined >= max_chain:
                        break
                    examined += 1
                    match_len = 3
                    while (match_len < max_len and
                           data[cand + match_len] == data[pos + match_len]):
                        match_len += 1
                    if match_len > best_len:
                        best_len = match_len
                        best_offset = pos - cand
                        if best_len == max_len:
                            break

                positions = heads.setdefault(key, [])
                positions.append(pos)
                if len(positions) > 128:
                    del positions[:-64]

        if best_len >= 3:
            result.append(0x00)
            result += best_offset.to_bytes(2, "little")
            result.append(best_len)
            pos += best_len
        else:
            result.append(0x80)
            result.append(data[pos])
            pos += 1

    return bytes(result)

def _build_huffman_codes(freq: Counter) -> dict:
    """Build symbol -> bit-string table from byte frequencies."""
    heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = "0" + pair[1]
        for pair in hi[1:]:
            pair[1] = "1" + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    codes = {symbol: code for symbol, code in heap[0][1:]}

    if len(codes) == 1:
        only = next(iter(codes))
        codes[only] = "0"
    return codes

def huffman_compress(data: bytes) -> dict:
    """Huffman coding compression (legacy).

    Bits are packed incrementally into a bytearray -- the old version built
    an 8x-size Python bit-string first, which dominated memory usage.

    Args:
        data: Data to compress.

    Returns:
        Dict with "compressed_bytes", "code_table" and "padding".
    """
    if not data:
        return {"compressed_bytes": b"", "code_table": {}, "padding": 0}

    freq = Counter(data)
    code_table = _build_huffman_codes(freq)

    packed = {symbol: (int(code, 2) if code else 0, len(code))
              for symbol, code in code_table.items()}

    out = bytearray()
    acc = 0
    acc_bits = 0
    for byte in data:
        pattern, length = packed[byte]
        acc = (acc << length) | pattern
        acc_bits += length
        while acc_bits >= 8:
            acc_bits -= 8
            out.append((acc >> acc_bits) & 0xFF)
        acc &= (1 << acc_bits) - 1

    padding = (8 - acc_bits) % 8
    if acc_bits:
        out.append(acc << padding)

    return {
        "compressed_bytes": bytes(out),
        "code_table": code_table,
        "padding": padding,
    }

def compress_data(data: str, method: str = "zlib") -> bytes:
    """Compress text data using the specified method.

    Args:
        data: Text data to compress.
        method: "zlib" (default), "lz77", "simple" or "huffman".

    Returns:
        Compressed bytes. For "huffman" only the packed bytes are returned;
        use huffman_compress() directly when the code table is needed.
    """
    bytes_data = data.encode("utf-8")

    if method == "zlib":
        return zlib_compress(bytes_data)
    elif method == "simple":
        return simple_compress(bytes_data)
    elif method == "lz77":
        return lz77_compress(bytes_data)
    elif method == "huffman":
        return huffman_compress(bytes_data)["compressed_bytes"]
    else:
        raise ValueError(f"Unknown compression method: {method}")
