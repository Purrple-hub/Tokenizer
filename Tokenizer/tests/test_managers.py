import pytest

from anatok.managers import (
    get_compression_manager,
    get_tokenization_manager,
    memory_report,
    quick_tokenize,
)

def test_quick_tokenize():
    result = quick_tokenize("Quick test of tokenization")
    assert "input_ids" in result and "tokens" in result
    assert len(result["input_ids"]) == len(result["tokens"])

def test_process_path_hf(sample_file):
    tm = get_tokenization_manager()
    result = tm.process_path(sample_file, use_hf=True, export=False)
    assert "error" not in result
    assert result["method"] == "hf_tokenizer"
    assert result["original_tokens"] > 0
    assert result["duplicates_removed"] >= 0

def test_process_path_custom_and_compress(sample_file):
    tm = get_tokenization_manager()
    result = tm.process_path(sample_file, use_hf=False, compress=True,
                             decompress=True, export=False)
    assert "error" not in result
    assert result["method"] == "custom_tokenizer"
    comp = result.get("compressed")
    assert comp and comp["compressed_size"] > 0
    assert result.get("roundtrip_ok") is True

def test_compress_manager_streaming_roundtrip(tmp_path):
    p = tmp_path / "stream.txt"
    p.write_text("compress me repeatedly " * 500)
    cm = get_compression_manager()
    cres = cm.compress_path_streaming(str(p), method="zlib")
    assert "error" not in cres
    verification = cm.decompress_verify(
        cres["data"], method="zlib", expected_sha256=cres["sha256"])
    assert verification["verified"] is True
    assert verification["decompressed_size"] == cres["original_size"]

def test_decompress_verify_payload_dropped():
    cm = get_compression_manager()
    res = cm.decompress_verify(b"", data_available=False)
    assert res["verified"] is None
    assert "note" in res

def test_memory_report_structure():
    report = memory_report()
    assert "memory_manager" in report
    assert "peak_usage_mb" in report
    assert "library" in report["ctypes_wrapper"]
