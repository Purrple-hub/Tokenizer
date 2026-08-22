"""Worker modules for the Tokenizer project.

Background workers for tokenization, compression, and operations
that must not block a GUI thread.

Integrates HuggingFace Tokenizers, ctypes libraries and real-process
memory tracking. All workers share cached managers/tokenizers instead of
constructing fresh instances, and support cooperative cancellation.
"""

import threading
import queue
import time
import sys
from typing import Optional, Callable, Any, Dict, List

from .managers import TokenizationManager, CompressionManager
from .managers import get_tokenization_manager, get_compression_manager
from .memory_manager import get_memory_manager, memory_pressure_test
from .ctypes_wrapper import CTypesWrapper, get_ctypes_wrapper, load_c_library
from .tokenizer import HFTokenizerWrapper, text_tokenize, byte_based_tokenize, detokenize, merge_tokens, get_hf_tokenizer
from .utils import validate_path

class BaseWorker(threading.Thread):
    """Base worker class with signal-like callback pattern.

    Uses the shared cached HF tokenizer so worker creation never triggers
    model loads, and exposes cooperative stop support.
    """

    def __init__(self, callback=None, error_callback=None, use_hf: bool = True):
        super().__init__(daemon=True)
        self.callback = callback
        self.error_callback = error_callback
        self.use_hf = use_hf
        self._result = None
        self._error = None
        self._stop_event = threading.Event()
        self._memory_manager = get_memory_manager()
        self._ctypes = get_ctypes_wrapper()
        self._hf_tokenizer: Optional[HFTokenizerWrapper] = \
            get_hf_tokenizer() if use_hf else None

    def stop(self):
        """Request the worker to stop."""
        self._stop_event.set()

    def stopped(self):
        """Check if stop has been requested."""
        return self._stop_event.is_set()

    def _safe_callback(self, *args, **kwargs):
        """Safely invoke the callback."""
        if self.callback:
            try:
                self.callback(*args, **kwargs)
            except Exception as e:
                if self.error_callback:
                    self.error_callback(str(e))

    def _fail(self, message: str):
        """Record and report an error."""
        self._error = str(message)
        self._log(f"{message}", level="error")
        if self.error_callback:
            try:
                self.error_callback(self._error)
            except Exception:
                pass

    def _log(self, message: str, level: str = "info"):
        """Internal logging."""
        print(f"[Worker] {level.upper()}: {message}")

class TokenizationWorker(BaseWorker):
    """Worker for tokenization operations in background thread."""

    def __init__(self, path: str, callback=None, error_callback=None,
                 use_hf: bool = True):
        super().__init__(callback=callback, error_callback=error_callback,
                         use_hf=use_hf)
        self.path = path
        self.manager = get_tokenization_manager(
            self._hf_tokenizer.model_name if self._hf_tokenizer
            else "bert-base-uncased")

    def run(self):
        """Run tokenization in background thread with memory tracking."""
        try:
            self._memory_manager.initialize()

            result = self.manager.process_path(
                self.path,
                compress=False,
                decompress=False,
                use_hf=self.use_hf,
                should_abort=self.stopped,
            )

            self._result = result
            self._safe_callback("tokenization_complete", result)

        except Exception as e:
            self._safe_callback("error_occurred", str(e))

class CompressionWorker(BaseWorker):
    """Worker for compression operations in background thread (bytes API)."""

    def __init__(self, data: bytes, method: str = "zlib", callback=None,
                 error_callback=None):
        super().__init__(callback=callback, error_callback=error_callback,
                         use_hf=False)
        self.data = data
        self.method = method
        self.manager = get_compression_manager()

    def run(self):
        """Run compression in background thread with memory tracking."""
        try:
            self._memory_manager.initialize()

            result = self.manager.compress_bytes(self.data, self.method)

            self._result = result
            self._safe_callback("compression_complete", result)

        except Exception as e:
            self._safe_callback("error_occurred", str(e))

class DecompressionWorker(BaseWorker):
    """Worker for decompression operations in background thread (bytes API)."""

    def __init__(self, data, method: str = "zlib", callback=None,
                 error_callback=None):
        super().__init__(callback=callback, error_callback=error_callback,
                         use_hf=False)
        self.data = data
        self.method = method
        self.manager = get_compression_manager()

    def run(self):
        """Run decompression in background thread."""
        try:
            self._memory_manager.initialize()

            result = self.manager.decompress_bytes(self.data, self.method)

            self._result = result
            self._safe_callback("decompression_complete", result)

        except Exception as e:
            self._safe_callback("error_occurred", str(e))

class CTypesWorker(BaseWorker):
    """Worker for ctypes library operations in background thread."""

    def __init__(self, library_path: Optional[str], callback=None,
                 error_callback=None):
        super().__init__(callback=callback, error_callback=error_callback,
                         use_hf=False)
        self.library_path = library_path
        self.load_result = load_c_library(library_path) \
            if library_path else None

    def run(self):
        """Run ctypes library operations in background thread."""
        try:
            if self.library_path and not self.load_result:
                self._result = {"library_loaded": False,
                                "path": self.library_path}
                self._safe_callback("ctypes_complete", self._result)
                return

            self._ctypes = get_ctypes_wrapper()
            if self._ctypes and self._ctypes.lib:
                output_buf = self._ctypes.create_buffer(4096)

            self._result = {"library_loaded": True, "path": self.library_path}
            self._safe_callback("ctypes_complete", self._result)

        except Exception as e:
            self._safe_callback("error_occurred", str(e))

class MemoryPressureTestWorker(BaseWorker):
    """Worker for memory pressure testing in background thread."""

    def __init__(self, iterations: int = 100, callback=None,
                 error_callback=None):
        super().__init__(callback=callback, error_callback=error_callback,
                         use_hf=False)
        self.iterations = iterations

    def run(self):
        """Run memory pressure test in background thread."""
        try:
            self._memory_manager.initialize()

            result = memory_pressure_test(self.iterations)

            self._result = result
            self._safe_callback("memory_test_complete", result)

        except Exception as e:
            self._safe_callback("error_occurred", str(e))
