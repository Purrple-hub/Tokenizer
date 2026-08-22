import pytest

from anatok.compressors import (
    compress_data,
    huffman_compress,
    lz77_compress,
    simple_compress,
    zlib_compress,
)
from anatok.decompressors import (
    decompress_data,
    huffman_decompress,
    lz77_decompress,
    simple_decompress,
    zlib_decompress,
)

def test_zlib_roundtrip():
    data = b"zlib roundtrip payload " * 100
    assert zlib_decompress(zlib_compress(data)) == data

def test_simple_roundtrip():
    original = b"aaaaabbbbbccccc"
    assert simple_decompress(simple_compress(original)) == original

def test_lz77_roundtrip():
    original = b"this is a test of lz77 compression method" * 5
    assert lz77_decompress(lz77_compress(original)) == original

def test_huffman_roundtrip():
    original = b"the quick brown fox jumps over the lazy dog" * 3
    packed = huffman_compress(original)
    assert set(packed) == {"compressed_bytes", "code_table", "padding"}
    assert huffman_decompress(packed) == original

def test_short_inputs_survive():
    for raw in (b"", b"a", b"ab", b"\x00\x80\xff"):
        assert simple_decompress(simple_compress(raw)) == raw
        assert lz77_decompress(lz77_compress(raw)) == raw

def test_compress_data_dispatch():
    text = "This is a test of the compress data function"
    for method in ("zlib", "simple", "lz77"):
        out = compress_data(text, method=method)
        assert isinstance(out, bytes)
    assert compress_data(text, method="huffman") == \
        huffman_compress(text.encode("utf-8"))["compressed_bytes"]

def test_invalid_method_raises():
    with pytest.raises(ValueError):
        compress_data("x", method="bogus")
    with pytest.raises(ValueError):
        decompress_data(b"", method="bogus")

def test_corrupt_lz77_stops_defensively():
    bad = bytearray(lz77_compress(b"aaaaaaaaaaaaaaaaaaaa"))
    bad[0] = 0x7F
    out = lz77_decompress(bytes(bad))
    assert isinstance(out, bytes)
