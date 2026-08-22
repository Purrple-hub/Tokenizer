"""Runtime smoke test for anatok: exercise real code paths end to end.

Run with:  python -m anatok.smoke_test
"""
import os
import tempfile

def _make_sample(text: str) -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    tmp.write(text)
    tmp.close()
    return tmp.name

def main() -> int:
    from .testing import run_all_tests

    print("=" * 60)
    print("SMOKE 1: Test suite")
    print("=" * 60)
    run_all_tests()

    print()
    print("=" * 60)
    print("SMOKE 2: TokenizationManager.process_path "
          "(HF + compress + decompress)")
    print("=" * 60)
    from .managers import get_tokenization_manager

    sample = _make_sample(
        "The quick brown fox jumps over the lazy dog. " * 20)
    try:
        tm = get_tokenization_manager()
        result = tm.process_path(sample, compress=True, decompress=True,
                                 use_hf=True)
        assert "error" not in result, result.get("error")
        print(f"  tokens={result['original_tokens']} "
              f"unique={result['optimized_tokens']}")
        print(f"  compressed_size={result['compressed']['compressed_size']} "
              f"ratio={result['compressed']['ratio']:.3f}")
        assert isinstance(result["compressed"]["ratio"], float), \
            "ratio must be float"
        print(f"  roundtrip_ok={result['roundtrip_ok']}")
        assert result["roundtrip_ok"], "roundtrip failed"

        exports = result.get("exports") or {}
        if exports.get("arrow"):
            assert os.path.exists(exports["arrow"]), "arrow file missing"
            print(f"  arrow export ok: {exports['arrow']}")

        print("  custom path:")
        result2 = tm.process_path(sample, use_hf=False)
        assert "error" not in result2
        print(f"  method={result2['method']} "
              f"tokens={result2['original_tokens']}")
    finally:
        os.unlink(sample)

    print()
    print("=" * 60)
    print("SMOKE 3: CompressionManager byte-level API (used by workers/gui)")
    print("=" * 60)
    from .managers import get_compression_manager
    cm = get_compression_manager()
    cres = cm.compress_bytes(b"hello world hello world hello world", "lz77")
    assert "data" in cres and "compression_ratio" in cres
    print(f"  compress_bytes ok: {cres['original_size']} -> "
          f"{cres['compressed_size']} bytes")
    dres = cm.decompress_bytes(cres["data"], "lz77")
    assert dres["data"].decode() == "hello world hello world hello world"
    print("  decompress_bytes ok: roundtrip verified")

    print()
    print("=" * 60)
    print("SMOKE 4: Workers actually execute")
    print("=" * 60)
    from .workers import (TokenizationWorker, CompressionWorker,
                          DecompressionWorker, MemoryPressureTestWorker)

    worker_sample = _make_sample("worker thread sample text " * 50)

    done = {}
    try:
        tw = TokenizationWorker(worker_sample, use_hf=False,
                                callback=lambda ev, r: done.__setitem__("tok", True),
                                error_callback=lambda e: done.__setitem__("tok_err", e))
        tw.start(); tw.join(30)
        assert done.get("tok"), f"tokenization worker failed: {done}"
        print("  TokenizationWorker OK")

        cw = CompressionWorker(b"abcabcabcabc",
                               callback=lambda ev, r: done.__setitem__("comp", r),
                               error_callback=lambda e: done.__setitem__("comp_err", e))
        cw.start(); cw.join(15)
        assert "comp" in done, f"compression worker failed: {done}"
        print(f"  CompressionWorker OK ({done['comp']['compressed_size']} bytes)")

        dw = DecompressionWorker(done["comp"]["data"],
                                 callback=lambda ev, r: done.__setitem__("decomp", r))
        dw.start(); dw.join(15)
        assert "decomp" in done and done["decomp"]["data"] == b"abcabcabcabc"
        print("  DecompressionWorker OK (roundtrip verified)")

        mw = MemoryPressureTestWorker(iterations=10,
                                      callback=lambda ev, r: done.__setitem__("mem", True))
        mw.start(); mw.join(30)
        assert done.get("mem"), "memory test worker failed"
        print("  MemoryPressureTestWorker OK")
    finally:
        os.unlink(worker_sample)

    print()
    print("=" * 60)
    print("SMOKE 5: FolderWorker over a temp directory")
    print("=" * 60)
    from .folder_workers import FolderWorker

    folder = tempfile.mkdtemp(prefix="anatok_smoke_")
    for i in range(3):
        with open(os.path.join(folder, f"f{i}.txt"), "w",
                  encoding="utf-8") as f:
            f.write(f"folder file number {i} " * (10 + i))

    fw = FolderWorker(folder, operation="tokenize")
    summary = fw.process()
    assert "error" not in summary
    assert summary["total_files"] == 3, summary
    print(f"  files={summary['total_files']} errors={summary['errors']}")

    print()
    print("=" * 60)
    print("SMOKE 6: quick_tokenize / tokenize_with_hf / memory_report")
    print("=" * 60)
    from .managers import quick_tokenize, memory_report
    from .tokenizer import tokenize_with_hf

    q = quick_tokenize("Hello tokenizer world")
    assert q["tokens"] and q["tokens"][0] != "<id_0>", \
        f"reverse vocab broken: {q['tokens'][:5]}"
    print(f"  quick_tokenize tokens: {q['tokens'][:6]}...")

    t = tokenize_with_hf("offsets please", return_offsets=True)
    assert "offset_mapping" in t and \
        len(t["offset_mapping"]) == len(t["input_ids"])
    print(f"  offsets aligned: {len(t['offset_mapping'])} entries")

    r = memory_report()
    assert "library" in r["ctypes_wrapper"]
    print(f"  memory_report ok: peak={r['peak_usage_mb']:.2f}MB")

    print()
    print("ALL SMOKE TESTS PASSED")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
