"""Folder worker module for processing multiple files in a directory.

Runs tokenization/compression on all files within a folder path using
shared cached managers, streaming I/O and per-file size caps.
"""

import os
import threading
from typing import Callable, Dict, List, Optional

from .managers import get_tokenization_manager, get_compression_manager
from .utils import validate_path

DEFAULT_MAX_FILE_SIZE = 512 * 1024 * 1024

class FolderWorker:
    """Process all files in a folder (tokenize or compress each file)."""

    SUPPORTED_OPERATIONS = ("tokenize", "compress")

    def __init__(self, path: str, operation: str = "tokenize",
                 method: str = "zlib",
                 max_file_size: int = DEFAULT_MAX_FILE_SIZE,
                 hf_model: str = "bert-base-uncased",
                 use_hf: bool = True):
        if operation not in self.SUPPORTED_OPERATIONS:
            raise ValueError(
                f"Unknown operation '{operation}'. "
                f"Expected one of {self.SUPPORTED_OPERATIONS}"
            )
        self.path = path
        self.operation = operation
        self.method = method
        self.max_file_size = max_file_size
        self.hf_model = hf_model
        self.use_hf = use_hf
        self.results: List[Dict] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    def stop(self):
        """Request the folder worker to stop after the current file."""
        self._stop_event.set()

    def stopped(self) -> bool:
        """Check whether stop was requested."""
        return self._stop_event.is_set()

    def _list_files(self) -> List[str]:
        """List immediate regular files under path."""
        if not os.path.isdir(self.path):
            return []
        return [
            os.path.join(self.path, f)
            for f in sorted(os.listdir(self.path))
            if os.path.isfile(os.path.join(self.path, f))
        ]

    def process(self,
                progress_callback: Optional[Callable[[int, int], None]] = None,
                complete_callback: Optional[Callable[[List[Dict]], None]] = None,
                file_progress_callback: Optional[
                    Callable[[str, int, int], None]] = None) -> Dict:
        """Process all files sequentially with streaming and abort support.

        Args:
            progress_callback: Called with (current, total) after each file.
            complete_callback: Called with summary when done.
            file_progress_callback: Called with (path, done_bytes,
                total_bytes) during per-file streaming operations.

        Returns:
            Summary dictionary with per-file summaries.
        """
        if not validate_path(self.path):
            return {"error": f"Path not found: {self.path}"}

        files = self._list_files()
        total = len(files)
        self.results = []

        tokenize_manager = get_tokenization_manager(self.hf_model) \
            if self.operation == "tokenize" else None
        compress_manager = get_compression_manager() \
            if self.operation == "compress" else None

        for idx, filepath in enumerate(files):
            if self.stopped():
                break

            try:
                size = os.path.getsize(filepath)
                if size > self.max_file_size:
                    result = {
                        "file": filepath, "skipped": True,
                        "reason": f"file exceeds cap "
                                  f"({size // (1024*1024)}MB)",
                    }
                elif self.operation == "tokenize":
                    result = tokenize_manager.process_path(
                        filepath, should_abort=self.stopped,
                        use_hf=self.use_hf)
                    result["file"] = filepath
                else:
                    cres = compress_manager.compress_path_streaming(
                        filepath, method=self.method,
                        should_abort=self.stopped,
                        progress_cb=(
                            lambda d, t, p=filepath:
                            file_progress_callback(p, d, t))
                        if file_progress_callback else None)
                    result = {
                        "file": filepath,
                        "method": self.method,
                        "original_size": cres.get("original_size", 0),
                        "compressed_size": cres.get("compressed_size", 0),
                        "ratio": cres.get("ratio", 0),
                    }
                    if "error" in cres:
                        result["error"] = cres["error"]
            except Exception as e:
                result = {"file": filepath, "error": str(e)}

            with self._lock:
                self.results.append(result)

            if progress_callback:
                progress_callback(idx + 1, total)

        summary = {
            "path": self.path,
            "operation": self.operation,
            "total_files": total,
            "processed": sum(1 for r in self.results if not r.get("skipped")),
            "skipped": sum(1 for r in self.results if r.get("skipped")),
            "errors": sum(1 for r in self.results if "error" in r),
            "results": self.results,
        }

        if complete_callback:
            complete_callback(summary)

        return summary
