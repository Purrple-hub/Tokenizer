"""Deduplication module for token streams.

Optimized deduplication for reducing redundant tokens in tokenization output.
"""

def remove_duplicate_tokens(token_list: list) -> list:
    """Remove duplicate tokens from a token list while preserving order.

    Args:
        token_list: List of tokens (strings or bytes).

    Returns:
        List with duplicates removed, order preserved.
    """
    seen = set()
    result = []
    for token in token_list:
        key = token if isinstance(token, str) else token.decode("utf-8", errors="replace")
        if key not in seen:
            seen.add(key)
            result.append(token)
    return result

def remove_duplicate_bytes(data: bytes) -> bytes:
    """Remove duplicate consecutive bytes from data.

    Args:
        data: Input byte data.

    Returns:
        Byte data with consecutive duplicates removed.
    """
    if not data:
        return data

    result = bytearray([data[0]])
    for byte in data[1:]:
        if byte != result[-1]:
            result.append(byte)
    return bytes(result)

def token_stream_optimize(tokens: list) -> dict:
    """Optimize a token stream by removing duplicates and analyzing patterns.

    Args:
        tokens: List of tokens to optimize.

    Returns:
        Dictionary with optimized tokens and statistics.
    """
    original_count = len(tokens)
    optimized = remove_duplicate_tokens(tokens)
    removed = original_count - len(optimized)

    return {
        "original_tokens": original_count,
        "optimized_tokens": len(optimized),
        "duplicates_removed": removed,
        "compression_ratio": removed / original_count if original_count > 0 else 0,
        "optimized_list": optimized,
    }

def find_common_prefix(tokens: list) -> str:
    """Find common prefix among a list of token strings.

    Args:
        tokens: List of token strings.

    Returns:
        Common prefix string (may be empty).
    """
    if not tokens:
        return ""

    shortest = min(tokens, key=len)
    prefix = ""
    for i in range(len(shortest)):
        char = shortest[i]
        if all(token[i] == char for token in tokens):
            prefix += char
        else:
            break
    return prefix
