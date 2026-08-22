from anatok.extras import (
    decode_base64_to_token,
    encode_token_to_base64,
    generate_token_report,
    load_session,
    save_session,
)
from anatok.extras.compression import zlib_compress, zlib_ratio

def test_base64_token_roundtrip():
    encoded = encode_token_to_base64("hello")
    assert isinstance(encoded, str)
    assert decode_base64_to_token(encoded) == b"hello"

def test_save_load_session(tmp_path):
    path = str(tmp_path / "session.json")
    assert save_session({"k": [1, 2]}, path) is True
    assert load_session(path) == {"k": [1, 2]}

def test_load_missing_session(tmp_path):
    assert load_session(str(tmp_path / "missing.json")) == {}

def test_generate_token_report():
    report = generate_token_report({
        "original_tokens": 10, "optimized_tokens": 8,
        "duplicates_removed": 2, "compression_ratio": 0.2})
    assert "Original tokens: 10" in report

def test_extras_compression_helpers():
    data = b"extras compression helper data " * 50
    compressed = zlib_compress(data)
    assert zlib_ratio(data) < 1.0
    import zlib as _z
    assert _z.decompress(compressed) == data
