"""Testing module for the Tokenizer project.

Contains test functions and validation for all core modules including
HuggingFace Tokenizers, ctypes integration, and advanced RAM management.
Aim for 80%+ code coverage."""
import os
import sys
import time
import tempfile

from .utils import compute_hash, validate_path, count_tokens, truncate_text, format_bytes, read_file_content
from .tokenizer import (
    HFTokenizerWrapper,
    text_tokenize,
    byte_based_tokenize,
    tokenize_file,
    merge_tokens,
    detokenize,
    tokenize_with_hf,
    benchmark_tokenization,
)
from .compressors import (
    simple_compress,
    lz77_compress,
    huffman_compress,
    compress_data,
)
from .decompressors import (
    simple_decompress,
    lz77_decompress,
    huffman_decompress,
    decompress_data,
)
from .dedup import (
    remove_duplicate_tokens,
    remove_duplicate_bytes,
    token_stream_optimize,
    find_common_prefix,
)
from .memory_manager import get_memory_manager, memory_pressure_test
from .ctypes_wrapper import CTypesWrapper, get_ctypes_wrapper, load_c_library
from .managers import get_tokenization_manager, get_compression_manager, quick_tokenize, memory_report
from .workers import TokenizationWorker, CompressionWorker, DecompressionWorker, CTypesWorker, MemoryPressureTestWorker, get_ctypes_wrapper

def run_all_tests():
    """Run all test functions and return summary."""
    tests = [
        ("Utility Tests", run_utils_tests),
        ("Tokenization Tests", run_tokenization_tests),
        ("Compression Tests", run_compression_tests),
        ("Decompression Tests", run_decompression_tests),
        ("Deduplication Tests", run_dedup_tests),
        ("HF Tokenizer Tests", run_hf_tokenizer_tests),
        ("Memory Manager Tests", run_memory_manager_tests),
        ("CTypes Tests", run_ctypes_tests),
        ("Manager Tests", run_manager_tests),
        ("Worker Tests", run_worker_tests),
    ]

    total = 0
    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")
        result = test_func()
        total += result["total"]
        passed += result["passed"]
        failed += result["total"] - result["passed"]
        print(f"  Passed: {result['passed']}/{result['total']}")

    print(f"\n{'='*60}")
    print(f"OVERALL: {passed}/{total} passed, {failed} failed")
    print(f"{'='*60}")
    return passed, total, failed

def run_utils_tests():
    """Run utility function tests."""
    total = 0
    passed = 0

    total += 1
    h = compute_hash(b"hello")
    if isinstance(h, str) and len(h) == 64:
        passed += 1
    else:
        print("  FAIL: compute_hash")

    total += 1
    if validate_path(__file__):
        passed += 1
    else:
        print("  FAIL: validate_path valid file")

    total += 1
    if not validate_path("nonexistent_file_xyz.py"):
        passed += 1
    else:
        print("  FAIL: validate_path invalid file")

    total += 1
    formatted = format_bytes(1024)
    if "KB" in formatted:
        passed += 1
    else:
        print("  FAIL: format_bytes")

    total += 1
    tc = count_tokens("hello world test")
    if tc == 3:
        passed += 1
    else:
        print(f"  FAIL: count_tokens expected 3 got {tc}")

    total += 1
    truncated = truncate_text("a" * 1000, 50)
    if len(truncated) <= 53:
        passed += 1
    else:
        print(f"  FAIL: truncate_text len={len(truncated)}")

    total += 1
    info = read_file_content(__file__) if os.path.exists(__file__) else b""
    if len(info) > 0:
        passed += 1
    else:
        print("  FAIL: read_file_content")

    return {"total": total, "passed": passed}

def run_tokenization_tests():
    """Run tokenization function tests."""
    total = 0
    passed = 0

    total += 1
    tokens = text_tokenize("hello world this is a test")
    if isinstance(tokens, list) and len(tokens) > 0:
        passed += 1
    else:
        print("  FAIL: text_tokenize")

    total += 1
    data = b"test data for byte tokenization"
    tokens = byte_based_tokenize(data)
    if isinstance(tokens, list):
        passed += 1
    else:
        print("  FAIL: byte_based_tokenize")

    total += 1
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Hello World Tokenization Test")
        tmp_path = f.name
    try:
        result = tokenize_file(tmp_path)
        if "tokens" in result and "token_count" in result:
            passed += 1
        else:
            print("  FAIL: tokenize_file result structure")
    finally:
        os.unlink(tmp_path)

    total += 1
    list_a = ["a", "b", "c"]
    list_b = ["d", "e"]
    merged = merge_tokens([list_a, list_b])
    if merged == ["a", "b", "c", "d", "e"]:
        passed += 1
    else:
        print(f"  FAIL: merge_tokens got {merged}")

    total += 1
    tokens = ["hello", "world", "test"]
    text = detokenize(tokens)
    if text == "hello world test":
        passed += 1
    else:
        print(f"  FAIL: detokenize got '{text}'")

    total += 1
    try:
        result = tokenize_with_hf("Hello world test of HF tokenizers")
        if "input_ids" in result and "tokens" in result:
            passed += 1
        else:
            print(f"  FAIL: tokenize_with_hf structure: {result}")
    except Exception as e:
        print(f"  FAIL: tokenize_with_hf raised {e}")

    total += 1
    try:
        test_texts = ["Hello world", "Tokenizer project", "AI optimization"]
        bench = benchmark_tokenization(test_texts, iterations=3)
        if "hf_tokenizer" in bench and "byte_based" in bench:
            passed += 1
        else:
            print(f"  FAIL: benchmark structure: {list(bench.keys())}")
    except Exception as e:
        print(f"  FAIL: benchmark_tokenization raised {e}")

    return {"total": total, "passed": passed}

def run_compression_tests():
    """Run compression function tests."""
    total = 0
    passed = 0

    total += 1
    original = b"aaaaabbbbbccccc"
    compressed = simple_compress(original)
    decompressed = simple_decompress(compressed)
    if decompressed == original:
        passed += 1
    else:
        print(f"  FAIL: simple compress/decompress roundtrip")

    total += 1
    original = b"this is a test of lz77 compression method"
    compressed = lz77_compress(original)
    decompressed = lz77_decompress(compressed)
    if decompressed == original:
        passed += 1
    else:
        print(f"  FAIL: lz77 compress/decompress roundtrip")

    total += 1
    original = b"the quick brown fox jumps over the lazy dog"
    huffman_result = huffman_compress(original)
    decompressed = huffman_decompress(huffman_result)
    if decompressed == original:
        passed += 1
    else:
        print(f"  FAIL: huffman compress/decompress roundtrip")

    total += 1
    text = "This is a test of the compress data function"
    compressed = compress_data(text, method="simple")
    if isinstance(compressed, bytes):
        passed += 1
    else:
        print("  FAIL: compress_data simple method")

    return {"total": total, "passed": passed}

def run_decompression_tests():
    """Run decompression function tests."""
    total = 0
    passed = 0

    total += 1
    try:
        decompress_data(b"", method="invalid")
        print("  FAIL: should have raised ValueError")
    except ValueError:
        passed += 1

    total += 1
    huffman_result = huffman_compress(b"test data")
    if isinstance(huffman_result, dict) and "code_table" in huffman_result:
        passed += 1
    else:
        print("  FAIL: huffman_compress output structure")

    return {"total": total, "passed": passed}

def run_dedup_tests():
    """Run deduplication function tests."""
    total = 0
    passed = 0

    total += 1
    tokens = ["a", "b", "a", "c", "b"]
    optimized = remove_duplicate_tokens(tokens)
    if optimized == ["a", "b", "c"]:
        passed += 1
    else:
        print(f"  FAIL: remove_duplicate_tokens got {optimized}")

    total += 1
    data = b"\x00\x00\x01\x01\x02"
    result = remove_duplicate_bytes(data)
    if result == b"\x00\x01\x02":
        passed += 1
    else:
        print(f"  FAIL: remove_duplicate_bytes got {result}")

    total += 1
    tokens = ["a", "b", "a", "c", "b", "d"]
    stats = token_stream_optimize(tokens)
    if (
        stats["original_tokens"] == 6
        and stats["optimized_tokens"] == 4
        and stats["duplicates_removed"] == 2
    ):
        passed += 1
    else:
        print(f"  FAIL: token_stream_optimize stats: {stats}")

    total += 1
    tokens = ["apple", "application", "appetizer"]
    prefix = find_common_prefix(tokens)
    if prefix == "app":
        passed += 1
    else:
        print(f"  FAIL: find_common_prefix got '{prefix}'")

    return {"total": total, "passed": passed}

def run_hf_tokenizer_tests():
    """Run HuggingFace Tokenizer integration tests."""
    total = 0
    passed = 0

    total += 1
    try:
        tokenizer = HFTokenizerWrapper("bert-base-uncased")
        if tokenizer is not None and hasattr(tokenizer, 'encode'):
            passed += 1
        else:
            print("  FAIL: HFTokenizerWrapper basic checks")
    except Exception as e:
        print(f"  FAIL: HFTokenizerWrapper init raised {e}")

    total += 1
    try:
        tokenizer = HFTokenizerWrapper("bert-base-uncased")
        encoded = tokenizer.encode("Hello world")
        decoded = tokenizer.decode(encoded)
        if isinstance(encoded, list) and len(encoded) > 0:
            passed += 1
        else:
            print(f"  FAIL: encode returned {type(encoded)}: {encoded}")
    except Exception as e:
        print(f"  FAIL: HF tokenizer encode/decode raised {e}")

    total += 1
    try:
        tokenizer = HFTokenizerWrapper("bert-base-uncased")
        vocab_size = tokenizer.get_vocab_size()
        if isinstance(vocab_size, int) and vocab_size > 0:
            passed += 1
        else:
            print(f"  FAIL: vocab_size={vocab_size}")
    except Exception as e:
        print(f"  FAIL: get_vocab_size raised {e}")

    total += 1
    try:
        result = tokenize_with_hf("Hello world HF tokenization test")
        if "input_ids" in result and "tokens" in result:
            passed += 1
        else:
            print(f"  FAIL: tokenize_with_hf result: {list(result.keys())}")
    except Exception as e:
        print(f"  FAIL: tokenize_with_hf raised {e}")

    total += 1
    try:
        test_texts = ["Hello world", "Tokenizer", "AI"]
        bench = benchmark_tokenization(test_texts, iterations=2)
        if "hf_tokenizer" in bench and "custom" in bench:
            passed += 1
        else:
            print(f"  FAIL: benchmark keys: {list(bench.keys())}")
    except Exception as e:
        print(f"  FAIL: benchmark_tokenization raised {e}")

    total += 1
    try:
        result = quick_tokenize("Quick tokenization test", model="bert-base-uncased")
        if "input_ids" in result and "tokens" in result:
            passed += 1
        else:
            print(f"  FAIL: quick_tokenize result: {list(result.keys())}")
    except Exception as e:
        print(f"  FAIL: quick_tokenize raised {e}")

    return {"total": total, "passed": passed}

def run_memory_manager_tests():
    """Run memory manager integration tests."""
    total = 0
    passed = 0

    total += 1
    try:
        mm = get_memory_manager()
        mm.initialize()
        if mm._initialized:
            passed += 1
        else:
            print("  FAIL: memory manager not initialized")
    except Exception as e:
        print(f"  FAIL: memory manager initialize raised {e}")

    total += 1
    try:
        mm = get_memory_manager()
        if not mm._initialized:
            mm.initialize()
        addr = mm.allocate(1024)
        if addr is not None:
            passed += 1
        else:
            print("  FAIL: memory allocation returned None")
    except Exception as e:
        print(f"  FAIL: memory allocate raised {e}")

    total += 1
    try:
        mm = get_memory_manager()
        if not mm._initialized:
            mm.initialize()
        addr = mm.allocate(512)
        if addr is not None:
            released = mm.release(addr)
            if released:
                passed += 1
            else:
                print("  FAIL: memory release returned False")
        else:
            print("  FAIL: could not allocate first")
    except Exception as e:
        print(f"  FAIL: memory release raised {e}")

    total += 1
    try:
        mm = get_memory_manager()
        if not mm._initialized:
            mm.initialize()
        stats = mm.get_stats()
        required_keys = ['pool_size', 'pool_offset', 'actual_used', 'free',
                         'peak_usage', 'memory_efficiency']
        if all(k in stats for k in required_keys):
            passed += 1
        else:
            missing = [k for k in required_keys if k not in stats]
            print(f"  FAIL: missing keys: {missing}")
    except Exception as e:
        print(f"  FAIL: get_stats raised {e}")

    total += 1
    try:
        result = memory_pressure_test(iterations=10)
        required_keys = ['iterations', 'elapsed_time', 'allocations_successful']
        if all(k in result for k in required_keys):
            passed += 1
        else:
            missing = [k for k in required_keys if k not in result]
            print(f"  FAIL: missing keys: {missing}")
    except Exception as e:
        print(f"  FAIL: memory_pressure_test raised {e}")

    total += 1
    try:
        report = memory_report()
        if "memory_manager" in report:
            passed += 1
        else:
            print("  FAIL: memory_report missing memory_manager key")
    except Exception as e:
        print(f"  FAIL: memory_report raised {e}")

    return {"total": total, "passed": passed}

def run_ctypes_tests():
    """Run ctypes integration tests."""
    total = 0
    passed = 0

    total += 1
    try:
        cw = get_ctypes_wrapper()
        if cw is not None:
            passed += 1
        else:
            print("  FAIL: ctypes wrapper is None")
    except Exception as e:
        print(f"  FAIL: ctypes wrapper init raised {e}")

    total += 1
    try:
        cw = get_ctypes_wrapper()
        result = load_c_library(None)
        passed += 1
    except Exception as e:
        print(f"  FAIL: load_c_library raised {e}")

    total += 1
    try:
        cw = get_ctypes_wrapper()
        buffer = cw.create_buffer(1024)
        if buffer is not None and len(buffer) == 1024:
            passed += 1
        else:
            print(f"  FAIL: create_buffer returned buffer of unexpected size")
    except Exception as e:
        print(f"  FAIL: ctypes methods raised {e}")

    total += 1
    try:
        cw1 = get_ctypes_wrapper()
        cw2 = get_ctypes_wrapper()
        if cw1 is cw2:
            passed += 1
        else:
            print("  FAIL: get_ctypes_wrapper not returning same instance")
    except Exception as e:
        print(f"  FAIL: get_ctypes_wrapper singleton raised {e}")

    return {"total": total, "passed": passed}

def run_manager_tests():
    """Run manager integration tests."""
    total = 0
    passed = 0

    total += 1
    try:
        tm = get_tokenization_manager()
        if tm is not None and hasattr(tm, 'process_path'):
            passed += 1
        else:
            print("  FAIL: get_tokenization_manager returned invalid object")
    except Exception as e:
        print(f"  FAIL: get_tokenization_manager raised {e}")

    total += 1
    try:
        cm = get_compression_manager()
        if cm is not None and hasattr(cm, 'compress_file'):
            passed += 1
        else:
            print("  FAIL: get_compression_manager returned invalid object")
    except Exception as e:
        print(f"  FAIL: get_compression_manager raised {e}")

    total += 1
    try:
        result = quick_tokenize("Quick test of tokenization", model="bert-base-uncased")
        if "input_ids" in result and "tokens" in result:
            passed += 1
        else:
            print(f"  FAIL: quick_tokenize result: {list(result.keys())}")
    except Exception as e:
        print(f"  FAIL: quick_tokenize raised {e}")

    total += 1
    try:
        report = memory_report()
        if "memory_manager" in report:
            passed += 1
        else:
            print("  FAIL: memory_report structure invalid")
    except Exception as e:
        print(f"  FAIL: memory_report raised {e}")

    total += 1
    try:
        tm = get_tokenization_manager("bert-base-uncased")
        result = tm.process_path if hasattr(tm, 'process_path') else {"skip": "no path"}
        if tm.hf_tokenizer is not None:
            passed += 1
        else:
            print("  FAIL: tokenization manager HF tokenizer not loaded")
    except Exception as e:
        print(f"  FAIL: tokenization manager HF test raised {e}")

    return {"total": total, "passed": passed}

def run_worker_tests():
    """Run worker thread integration tests."""
    total = 0
    passed = 0

    total += 1
    try:
        worker = TokenizationWorker("nonexistent_path.test", use_hf=False)
        if worker is not None and hasattr(worker, 'start'):
            passed += 1
        else:
            print("  FAIL: TokenizationWorker creation failed")
    except Exception as e:
        print(f"  FAIL: TokenizationWorker creation raised {e}")

    total += 1
    try:
        worker = CompressionWorker(b"test data", method="lz77")
        if worker is not None and hasattr(worker, 'start'):
            passed += 1
        else:
            print("  FAIL: CompressionWorker creation failed")
    except Exception as e:
        print(f"  FAIL: CompressionWorker creation raised {e}")

    total += 1
    try:
        worker = DecompressionWorker(b"compressed data")
        if worker is not None and hasattr(worker, 'start'):
            passed += 1
        else:
            print("  FAIL: DecompressionWorker creation failed")
    except Exception as e:
        print(f"  FAIL: DecompressionWorker creation raised {e}")

    total += 1
    try:
        worker = CTypesWorker(None)
        if worker is not None and hasattr(worker, 'start'):
            passed += 1
        else:
            print("  FAIL: CTypesWorker creation failed")
    except Exception as e:
        print(f"  FAIL: CTypesWorker creation raised {e}")

    total += 1
    try:
        worker = MemoryPressureTestWorker(iterations=10)
        if worker is not None and hasattr(worker, 'start'):
            passed += 1
        else:
            print("  FAIL: MemoryPressureTestWorker creation failed")
    except Exception as e:
        print(f"  FAIL: MemoryPressureTestWorker creation raised {e}")

    return {"total": total, "passed": passed}

if __name__ == "__main__":
    run_all_tests()
