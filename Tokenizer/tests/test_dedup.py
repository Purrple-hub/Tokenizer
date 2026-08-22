from anatok.dedup import (
    find_common_prefix,
    remove_duplicate_bytes,
    remove_duplicate_tokens,
    token_stream_optimize,
)

def test_remove_duplicate_tokens():
    assert remove_duplicate_tokens(["a", "b", "a", "c", "b"]) == \
        ["a", "b", "c"]

def test_remove_duplicate_bytes():
    assert remove_duplicate_bytes(b"\x00\x00\x01\x01\x02") == b"\x00\x01\x02"

def test_token_stream_optimize():
    stats = token_stream_optimize(["a", "b", "a", "c", "b", "d"])
    assert stats["original_tokens"] == 6
    assert stats["optimized_tokens"] == 4
    assert stats["duplicates_removed"] == 2
    assert abs(stats["compression_ratio"] - (2 / 6)) < 1e-9

def test_find_common_prefix():
    assert find_common_prefix(["apple", "application", "appetizer"]) == "app"
    assert find_common_prefix([]) == ""
