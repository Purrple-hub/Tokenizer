"""Manager modules for the Tokenizer project.

Orchestrates tokenization, compression, deduplication and memory tracking.

Design goals (v2):
- Constant-memory streaming: files are processed block by block; full token
  lists are never materialized.
- Slim results: counts, ratios and a small preview are returned instead of
  entire token/id/text payloads.
- Shared singletons: HF tokenizers and managers are reused across
  operations instead of being reloaded per request.
- Real memory reporting via the process-RSS MemoryManager.
"""
import os
import time
import hashlib
from typing import Any, Dict, List, Optional

from .utils import validate_path, format_bytes, iter_file_blocks
from .tokenizer import (
    get_hf_tokenizer,
    count_file_tokens_hf,
    iter_word_group_tokens,
    merge_tokens,
    detokenize,
    text_tokenize,
    byte_based_tokenize,
    tokenize_with_hf,
    benchmark_tokenization,
)
from .compressors import (
    compress_data,
    simple_compress,
    lz77_compress,
    huffman_compress,
    zlib_compress,
)
from .decompressors import decompress_data, zlib_decompress
from .dedup import token_stream_optimize, remove_duplicate_tokens
from .viewers import TokenViewer, CompressionViewer
from .memory_manager import get_memory_manager, MemoryChunk, memory_pressure_test
from .ctypes_wrapper import CTypesWrapper, load_c_library, get_ctypes_wrapper

COMPRESSED_RETAIN_LIMIT = 32 * 1024 * 1024

LEGACY_INPUT_LIMIT = 256 * 1024 * 1024

UNIQUE_TOKEN_CAP = 2_000_000

def _package_version() -> str:
    """Best-effort anatok version string for result metadata."""
    try:
        from importlib.metadata import version
        return version("anatok")
    except Exception:
        try:
            from . import __version__
            return __version__
        except Exception:
            return "unknown"

class TokenizationManager:
    """Manages tokenization operations with HF Tokenizers and real-time
    memory tracking. Files are streamed; results carry statistics plus a
    bounded preview instead of full token lists."""

    def __init__(self, hf_model: str = "bert-base-uncased"):
        self.hf_model = hf_model
        self.viewer = TokenViewer()
        self.hf_tokenizer = get_hf_tokenizer(hf_model)
        self.memory_manager = get_memory_manager()
        self.ctypes = get_ctypes_wrapper()
        self._initialized = False

    def _ensure_initialized(self):
        """Start memory monitoring once (cheap, idempotent)."""
        if not self._initialized:
            self.memory_manager.initialize()
            self._initialized = True

    def process_path(self, path: str, compress: bool = False,
                     decompress: bool = False, use_hf: bool = True,
                     should_abort=None, progress_cb=None,
                     export: bool = True,
                     results_dir: Optional[str] = None) -> dict:
        """Stream-process a file or folder through the tokenization pipeline.

        Args:
            path: File or folder path.
            compress: Whether to compress after tokenizing (zlib).
            decompress: Whether to verify roundtrip after compressing.
            use_hf: Use HuggingFace tokenizer when True, custom grouping
                tokenizer otherwise.
            should_abort: Optional callable returning True to stop early;
                the result dict then has aborted=True.
            progress_cb: Optional callable invoked as progress_cb(bytes_done)
                during tokenization.
            export: When True (default), write the full token stream as an
                Arrow IPC file (.pyarrow) plus JSON and Markdown summaries
                into the results directory.
            results_dir: Optional results directory override (defaults to
                $ANATOK_RESULTS_DIR or ./results).

        Returns:
            Dictionary with counts, ratios, a small token preview, memory
            stats and written artifact paths. Full token/id lists are
            intentionally not kept in memory.
        """
        if not validate_path(path):
            return {"error": f"Path not found: {path}"}
        if os.path.isdir(path):
            return {"error": "Use FolderWorker for directories", "path": path}

        self._ensure_initialized()
        started = time.perf_counter()

        arrow_writer = None
        if export:
            from .results import TokenArrowWriter, get_results_dir, result_stem
            rdir = get_results_dir(results_dir)
            stem = result_stem(path)
            arrow_writer = TokenArrowWriter(
                os.path.join(rdir, stem + ".pyarrow"),
                source_path=path,
                metadata={
                    "anatok_version": _package_version(),
                    "method": "hf_tokenizer" if use_hf else "custom_tokenizer",
                    "model": self.hf_model,
                })

        def encoding_cb(enc):
            arrow_writer.add_encoding(path, enc)

        def group_cb(group):
            arrow_writer.add_group(path, group)

        try:
            if use_hf:
                stats = count_file_tokens_hf(
                    path, model=self.hf_model,
                    preview_limit=self.PREVIEW_LIMIT,
                    should_abort=should_abort,
                    progress_cb=progress_cb,
                    encoding_cb=encoding_cb if arrow_writer else None)
                method = "hf_tokenizer"
                vocab_size = stats["vocab_size"]
                total = stats["total_tokens"]
                unique = stats["unique_tokens"]
                capped = stats["unique_capped"]
                aborted = stats["aborted"]
                preview = stats["preview"]
            else:
                total, unique, capped, aborted, preview = \
                    self._custom_stream_stats(
                        path, should_abort, progress_cb,
                        group_cb if arrow_writer else None)
                method = "custom_tokenizer"
                vocab_size = max(unique, 100)
        except OSError as e:
            return {"error": f"Cannot read file: {e}", "path": path}
        finally:
            if arrow_writer is not None:
                arrow_writer.close()

        elapsed = time.perf_counter() - started
        result = {
            "method": method,
            "vocab_size": vocab_size,
            "file": path,
            "file_size": file_bytes(path),
            "original_tokens": total,
            "optimized_tokens": unique,
            "duplicates_removed": total - unique,
            "compression_ratio": (total - unique) / total if total else 0.0,
            "unique_is_lower_bound": capped,
            "tokens": preview,
            "preview_tokens": preview,
            "aborted": aborted,
            "elapsed_seconds": round(elapsed, 3),
        }
        self.viewer.set_tokens(preview)

        if compress:
            comp_manager = get_compression_manager()
            cres = comp_manager.compress_path_streaming(
                path, method="zlib", should_abort=should_abort)
            if "error" in cres:
                result["compressed_error"] = cres["error"]
            else:
                entry = {
                    "method": "zlib",
                    "original_size": cres["original_size"],
                    "compressed_size": cres["compressed_size"],
                    "ratio": cres["ratio"],
                    "sha256": cres["sha256"],
                    "retained": cres["data"] is not None,
                    "memory_peak_mb":
                        cres.get("peak_rss_bytes", 0) / (1024 * 1024),
                }
                if cres["data"] is not None:
                    entry["data"] = cres["data"]
                result["compressed"] = entry

                if decompress:
                    verification = comp_manager.decompress_verify(
                        cres["data"] if cres["data"] is not None else b"",
                        method="zlib",
                        expected_sha256=cres["sha256"],
                        data_available=cres["data"] is not None)
                    result["roundtrip_ok"] = verification.get("verified")
                    if not verification.get("verified", False):
                        result["roundtrip_note"] = verification.get("note")

        mem_stats = self.memory_manager.get_stats()
        result["memory"] = {
            "rss_mb": mem_stats["rss_bytes"] / (1024 * 1024),
            "pool_usage_mb": mem_stats["rss_bytes"] / (1024 * 1024),
            "peak_usage_mb": mem_stats["peak_usage"] / (1024 * 1024),
            "memory_efficiency": mem_stats["memory_efficiency"],
            "chunks_active": mem_stats["active_chunks"],
        }

        if export:
            from .results import export_result_files
            result["exports"] = export_result_files(
                result, results_dir=rdir, stem=stem)
        return result

    PREVIEW_LIMIT = 500

    def _custom_stream_stats(self, path: str, should_abort=None,
                             progress_cb=None, group_cb=None):
        """Stream word-group tokens, tracking counts without full retention."""
        total = 0
        unique = set()
        capped = False
        preview: List[str] = []
        aborted = False

        for group in iter_word_group_tokens(path, should_abort=should_abort,
                                            progress_cb=progress_cb,
                                            group_cb=group_cb):
            total += 1
            if not capped:
                unique.add(group)
                if len(unique) > UNIQUE_TOKEN_CAP:
                    capped = True
                    unique.clear()
                    unique.add("__capped__")
            if len(preview) < self.PREVIEW_LIMIT:
                preview.append(group)

        if should_abort is not None and should_abort():
            aborted = True

        return total, len(unique), capped, aborted, preview

    def _log_error(self, message: str):
        """Log an error message."""
        print(f"[TokenizationManager] ERROR: {message}")

    def compress_with_memory_mgmt(self, text: str, method: str = "zlib") -> dict:
        """Compress text with before/after memory snapshots."""
        self._ensure_initialized()

        before_stats = self.memory_manager.get_stats()
        compressed = compress_data(text, method)
        after_stats = self.memory_manager.get_stats()

        original_size = len(text.encode("utf-8"))
        return {
            "compressed_data": compressed,
            "compressed_size": len(compressed),
            "original_size": original_size,
            "compression_ratio": len(compressed) / original_size if original_size else 0,
            "memory_before": {
                "used_mb": before_stats["rss_bytes"] / (1024 * 1024),
                "peak_mb": before_stats["peak_usage"] / (1024 * 1024),
            },
            "memory_after": {
                "used_mb": after_stats["rss_bytes"] / (1024 * 1024),
                "peak_mb": after_stats["peak_usage"] / (1024 * 1024),
            },
            "memory_delta_mb": (
                after_stats["rss_bytes"] - before_stats["rss_bytes"])
            / (1024 * 1024),
        }

    def benchmark_all(self, texts: List[str], iterations: int = 5) -> dict:
        """Benchmark tokenization methods and attach memory stats."""
        results = benchmark_tokenization(texts, iterations)
        results['memory_manager_stats'] = self.memory_manager.get_stats()
        return results

    def load_c_library(self, library_path: str) -> bool:
        """Load a C shared library for low-level operations."""
        return load_c_library_bool(library_path)

def file_bytes(path: str) -> int:
    """File size in bytes (0 when unavailable)."""
    try:
        return os.path.getsize(path)
    except OSError:
        return 0

class CompressionManager:
    """Manages compression/decompression with streaming and memory tracking.

    zlib supports true block-by-block streaming. Legacy methods
    (lz77/simple/huffman) operate on whole buffers and are size-guarded.
    """

    def __init__(self):
        self.memory_manager = get_memory_manager()
        self._initialized = False

    def _ensure_initialized(self):
        if not self._initialized:
            self.memory_manager.initialize()
            self._initialized = True

    def compress_path_streaming(self, path: str, method: str = "zlib",
                                should_abort=None,
                                progress_cb=None) -> dict:
        """Stream-compress a file, returning sizes/hash and optionally data.

        The compressed payload is only included when it does not exceed
        COMPRESSED_RETAIN_LIMIT; larger outputs are represented by sizes
        alone so huge files cannot balloon memory.

        Args:
            path: File to compress.
            method: Only "zlib" streams; legacy methods buffer the input.
            should_abort: Callable returning True to stop early.
            progress_cb: Called as progress_cb(bytes_done, bytes_total).

        Returns:
            Dict with original_size, compressed_size, ratio, sha256,
            data (or None), peak_rss_bytes and optional error.
        """
        self._ensure_initialized()
        if not validate_path(path):
            return {"error": f"Path not found: {path}"}

        total_size = file_bytes(path)

        if method == "zlib":
            import zlib

            hasher = hashlib.sha256()
            compressor = zlib.compressobj(6)
            done = 0
            out_parts = []
            out_len = 0
            retain = True
            aborted = False

            for block in iter_file_blocks(path):
                hasher.update(block)
                chunk = compressor.compress(block)
                if chunk:
                    out_len += len(chunk)
                    if retain:
                        out_parts.append(chunk)
                        if out_len > COMPRESSED_RETAIN_LIMIT:
                            out_parts.clear()
                            retain = False
                done += len(block)
                if progress_cb is not None:
                    progress_cb(done, total_size)
                if should_abort is not None and should_abort():
                    aborted = True
                    break

            if aborted:
                return {"error": "aborted", "aborted": True,
                        "original_size": done}

            tail = compressor.flush()
            out_len += len(tail)
            if retain and tail:
                out_parts.append(tail)
                if out_len > COMPRESSED_RETAIN_LIMIT:
                    out_parts.clear()
                    retain = False

            self.memory_manager.sample()
            return {
                "method": "zlib",
                "original_size": done,
                "compressed_size": out_len,
                "ratio": out_len / done if done else 0,
                "sha256": hasher.hexdigest(),
                "data": b"".join(out_parts) if retain else None,
                "peak_rss_bytes": self.memory_manager.get_peak_usage(),
                "aborted": False,
            }

        if total_size > LEGACY_INPUT_LIMIT:
            return {"error":
                    f"Method '{method}' requires buffering {total_size} "
                    f"bytes; use zlib for large files"}

        from .utils import read_file_content
        content = read_file_content(path)
        try:
            compressed = self._compress_bytes_legacy(content, method)
        except ValueError as e:
            return {"error": str(e)}

        self.memory_manager.sample()
        return {
            "method": method,
            "original_size": len(content),
            "compressed_size": len(compressed),
            "ratio": len(compressed) / len(content) if content else 0,
            "sha256": hashlib.sha256(content).hexdigest(),
            "data": compressed if len(compressed) <= COMPRESSED_RETAIN_LIMIT else None,
            "peak_rss_bytes": self.memory_manager.get_peak_usage(),
            "aborted": False,
        }

    @staticmethod
    def _compress_bytes_legacy(data: bytes, method: str) -> bytes:
        """Legacy whole-buffer compression dispatch (bytes in, bytes out)."""
        if method == "simple":
            return simple_compress(data)
        elif method == "lz77":
            return lz77_compress(data)
        elif method == "huffman":
            return huffman_compress(data)["compressed_bytes"]
        raise ValueError(f"Unknown compression method: {method}")

    def decompress_verify(self, compressed: bytes, method: str = "zlib",
                          expected_sha256: Optional[str] = None,
                          sample_chars: int = 4000,
                          data_available: bool = True) -> dict:
        """Stream-decompress while hashing, keeping only a small sample.

        For zlib this never materializes the full output: bytes are
        streamed through a decompressobj, hashed and counted, so even
        extreme compression ratios cannot balloon memory.

        When the compressed payload was dropped (too large), verification
        reports verified=None with an explanatory note instead of guessing.

        Args:
            compressed: Compressed payload (may be empty when
                data_available is False).
            method: Compression method used.
            expected_sha256: Hash of the original uncompressed content.
            sample_chars: Leading characters to include in the result.
            data_available: False when the payload was not retained.

        Returns:
            Dict with verified (True/False/None), decompressed_size,
            optional sample text and note.
        """
        if not data_available:
            return {"verified": None,
                    "note": "payload too large to retain; "
                            "sizes were still verified"}
        try:
            if method == "zlib":
                import zlib

                decompressor = zlib.decompressobj()
                hasher = hashlib.sha256()
                size = 0
                remaining = max(sample_chars, 0)
                sample_parts: List[str] = []

                def account(chunk: bytes):
                    nonlocal size, remaining
                    hasher.update(chunk)
                    size += len(chunk)
                    if remaining > 0 and chunk:
                        take = min(remaining, len(chunk))
                        sample_parts.append(
                            chunk[:take].decode("utf-8", errors="replace"))
                        remaining -= take

                view = memoryview(compressed)
                block = 1024 * 1024
                for i in range(0, len(view), block):
                    account(decompressor.decompress(view[i:i + block]))
                account(decompressor.flush())

                verified = True
                if expected_sha256 is not None:
                    verified = hasher.hexdigest() == expected_sha256
                return {"verified": verified,
                        "decompressed_size": size,
                        "sample": "".join(sample_parts)}

            raw = decompress_data(compressed, method)
            text = raw.decode("utf-8", errors="replace")
            verified = True
            if expected_sha256 is not None:
                verified = \
                    hashlib.sha256(raw).hexdigest() == expected_sha256
            return {"verified": verified,
                    "decompressed_size": len(raw),
                    "sample": text[:sample_chars]}
        except Exception as e:
            return {"verified": False,
                    "note": f"decompression failed: {e}"}

    def compress_file(self, path: str, method: str = "zlib",
                      retain_data: bool = False) -> dict:
        """Compress a file with memory tracking (streaming for zlib).

        Unlike the previous version this never returns the original file
        contents, keeping memory proportional to output rather than input.

        Args:
            path: File to compress.
            method: Compression method.
            retain_data: Include the compressed payload when within limits.

        Returns:
            Summary dictionary with sizes, ratio and memory stats.
        """
        result = self.compress_path_streaming(path, method=method)
        if "error" in result and not result.get("aborted"):
            return result
        if result.get("aborted"):
            return {"error": "aborted"}
        summary = {
            "original_path": path,
            "original_size": result["original_size"],
            "compressed_size": result["compressed_size"],
            "compression_ratio": result["ratio"],
            "method": result["method"],
            "sha256": result["sha256"],
            "formatted_original": format_bytes(result["original_size"]),
            "formatted_compressed": format_bytes(result["compressed_size"]),
            "memory_peak_mb": result["peak_rss_bytes"] / (1024 * 1024),
        }
        if retain_data:
            summary["data"] = result["data"]
        return summary

    def compress_bytes(self, data: bytes, method: str = "zlib") -> dict:
        """Compress raw bytes with memory tracking (no file I/O).

        Args:
            data: Raw bytes to compress.
            method: "zlib", "simple", "lz77" or "huffman".

        Returns:
            Result dictionary with compressed data and stats.
        """
        self._ensure_initialized()

        if method == "zlib":
            compressed = zlib_compress(data)
        else:
            compressed = self._compress_bytes_legacy(data, method)

        self.memory_manager.sample()
        stats = self.memory_manager.get_stats()
        original_size = len(data)
        return {
            "original_size": original_size,
            "compressed_size": len(compressed),
            "compression_ratio":
                len(compressed) / original_size if original_size else 0,
            "method": method,
            "data": compressed,
            "memory_after": {
                "used_mb": stats["rss_bytes"] / (1024 * 1024),
                "peak_mb": stats["peak_usage"] / (1024 * 1024),
            },
        }

    def decompress_bytes(self, data: bytes, method: str = "zlib") -> dict:
        """Decompress raw bytes with memory tracking (no file I/O).

        Args:
            data: Compressed bytes (dict for huffman).
            method: Compression method used.

        Returns:
            Result dictionary with decompressed data and size.
        """
        self._ensure_initialized()
        decompressed = decompress_data(data, method)
        return {
            "data": decompressed,
            "decompressed_size": len(decompressed),
            "method": method,
        }

def load_c_library_bool(library_path: str) -> bool:
    """Load a C library and report success as a bool."""
    wrapper = load_c_library(library_path)
    return wrapper is not None

_tokenization_manager: Optional[TokenizationManager] = None
_compression_manager: Optional[CompressionManager] = None

def get_tokenization_manager(hf_model: str = "bert-base-uncased") -> TokenizationManager:
    """Get the global TokenizationManager instance."""
    global _tokenization_manager
    if _tokenization_manager is None:
        _tokenization_manager = TokenizationManager(hf_model)
    return _tokenization_manager

def get_compression_manager() -> CompressionManager:
    """Get the global CompressionManager instance."""
    global _compression_manager
    if _compression_manager is None:
        _compression_manager = CompressionManager()
    return _compression_manager

def quick_tokenize(text: str, model: str = "bert-base-uncased") -> dict:
    """Quick tokenization convenience function for short texts.

    Args:
        text: Text to tokenize.
        model: HF model name.

    Returns:
        Tokenization result dictionary.
    """
    tokenizer = get_hf_tokenizer(model)
    encoding = tokenizer.encode_full(text)

    return {
        "input_ids": encoding.ids,
        "tokens": encoding.tokens,
        "vocab_size": tokenizer.get_vocab_size(),
        "text": text,
    }

def memory_report() -> dict:
    """Get comprehensive memory report from all managers.

    Returns:
        Memory report dictionary.
    """
    mm = get_memory_manager()
    cw = get_ctypes_wrapper()

    return {
        "memory_manager": mm.get_stats(),
        "peak_usage_mb": mm.get_peak_usage() / (1024 * 1024),
        "current_rss_mb": mm.get_current_usage() / (1024 * 1024),
        "ctypes_wrapper": {
            "loaded": cw._loaded,
            "library": str(getattr(cw, "lib", None)),
        },
    }
