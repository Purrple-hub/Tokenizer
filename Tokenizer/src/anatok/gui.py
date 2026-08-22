"""GUI module for the Tokenizer project using PyQt6.

All heavy work (file I/O, tokenization, compression) runs on worker
threads; the UI thread never reads files or loads models. Operations
report real byte-level progress and can be cancelled mid-stream.
"""

import os
import sys
import threading

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QFileDialog, QTextEdit, QProgressBar, QGroupBox,
                             QMessageBox, QTabWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from .managers import get_tokenization_manager, get_compression_manager, \
    COMPRESSED_RETAIN_LIMIT
from .utils import format_bytes
from .viewers import TokenViewer, CompressionViewer
from .error_warehouse import log_error

class TokenizationWorkerWrapper(QThread):
    """Background thread for tokenization with Qt signals."""

    done = pyqtSignal(dict)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, int)

    def __init__(self, path: str, use_hf: bool = True):
        super().__init__()
        self.path = path
        self.use_hf = use_hf
        self._stop_event = threading.Event()

    def cancel(self):
        """Request cancellation; takes effect between stream chunks."""
        self._stop_event.set()

    def run(self):
        """Run tokenization off the UI thread."""
        try:
            manager = get_tokenization_manager()
            total_size = os.path.getsize(self.path)

            def report(done_bytes: int):
                if total_size:
                    self.progress.emit(done_bytes, total_size)

            result = manager.process_path(
                self.path,
                compress=False,
                decompress=False,
                use_hf=self.use_hf,
                should_abort=self._stop_event.is_set,
                progress_cb=report,
            )
            self.done.emit(result)
        except Exception as e:
            self.failed.emit(str(e))

class PathCompressionWorker(QThread):
    """Background thread that streams a file into zlib compression."""

    done = pyqtSignal(dict)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, int)

    def __init__(self, path: str, method: str = "zlib"):
        super().__init__()
        self.path = path
        self.method = method
        self._stop_event = threading.Event()

    def cancel(self):
        """Request cancellation."""
        self._stop_event.set()

    def run(self):
        """Run streaming compression off the UI thread."""
        try:
            manager = get_compression_manager()
            result = manager.compress_path_streaming(
                self.path,
                method=self.method,
                should_abort=self._stop_event.is_set,
                progress_cb=lambda d, t: self.progress.emit(d, t),
            )
            self.done.emit(result)
        except Exception as e:
            self.failed.emit(str(e))

class DecompressionWorkerWrapper(QThread):
    """Background thread for decompression; reports a bounded sample."""

    done = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, data, method: str = "zlib"):
        super().__init__()
        self.data = data
        self.method = method

    def run(self):
        """Run streaming decompression off the UI thread."""
        try:
            manager = get_compression_manager()
            verification = manager.decompress_verify(
                self.data, method=self.method, sample_chars=4000,
                data_available=True)
            self.done.emit(verification)
        except Exception as e:
            self.failed.emit(str(e))

class TokenizerGUI(QMainWindow):
    """Main GUI window for the Tokenizer application."""

    MAX_TOKENS_SHOWN = 500
    SAMPLE_TEXT_LIMIT = 4000

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tokenizer - AI-Optimized Tokenization Tool")
        self.setMinimumSize(800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self._create_input_section(main_layout)
        self._create_operation_section(main_layout)
        self._create_results_section(main_layout)

        self.current_result = None
        self.compressed_entry = None
        self.viewer = TokenViewer()
        self.compression_viewer = CompressionViewer()
        self._worker: QThread | None = None

    def _create_input_section(self, parent):
        """Create the file path input section."""
        input_group = QGroupBox("File/Folder Path")
        input_layout = QHBoxLayout()

        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Enter file or folder path...")

        self.browse_file_btn = QPushButton("Browse File")
        self.browse_file_btn.clicked.connect(self._browse_file)

        self.browse_folder_btn = QPushButton("Browse Folder")
        self.browse_folder_btn.clicked.connect(self._browse_folder)

        input_layout.addWidget(self.path_input, 1)
        input_layout.addWidget(self.browse_file_btn)
        input_layout.addWidget(self.browse_folder_btn)
        input_group.setLayout(input_layout)
        parent.addWidget(input_group)

    def _browse_file(self):
        """Open a file browse dialog."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select File", "", "All Files (*);;Text Files (*.txt)"
        )
        if path:
            self.path_input.setText(path)
            self._set_operation_state(True)

    def _browse_folder(self):
        """Open a folder browse dialog."""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.path_input.setText(folder_path)
            self._set_operation_state(True)

    def _create_operation_section(self, parent):
        """Create the operation control section."""
        op_group = QGroupBox("Operations")
        op_layout = QHBoxLayout()

        self.tokenize_btn = QPushButton("Tokenize (HF)")
        self.tokenize_btn.clicked.connect(self._on_tokenize)

        self.tokenize_local_btn = QPushButton("Tokenize (Fast)")
        self.tokenize_local_btn.clicked.connect(self._on_tokenize_fast)

        self.compress_btn = QPushButton("Compress")
        self.compress_btn.clicked.connect(self._on_compress)

        self.decompress_btn = QPushButton("Decompress")
        self.decompress_btn.clicked.connect(self._on_decompress)

        for btn in (self.tokenize_btn, self.tokenize_local_btn,
                    self.compress_btn, self.decompress_btn):
            op_layout.addWidget(btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._on_cancel)
        op_layout.addWidget(self.cancel_btn)

        op_group.setLayout(op_layout)
        parent.addWidget(op_group)

    def _create_results_section(self, parent):
        """Create the results display section."""
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()

        self.tab_widget = QTabWidget()

        self.token_text = QTextEdit()
        self.token_text.setReadOnly(True)
        self.tab_widget.addTab(self.token_text, "Tokens")

        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        self.ratio_label = QLabel("Compression ratio: N/A")
        self.size_label = QLabel("Size: N/A")
        self.memory_label = QLabel("Memory: N/A")
        for lbl in (self.ratio_label, self.size_label, self.memory_label):
            lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            stats_layout.addWidget(lbl)
        stats_layout.addStretch(1)
        self.tab_widget.addTab(stats_widget, "Stats")

        results_layout.addWidget(self.tab_widget)
        results_group.setLayout(results_layout)
        parent.addWidget(results_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        parent.addWidget(self.progress_bar)

    def process_path(self, path: str, compress: bool = False,
                     decompress: bool = False):
        """Pre-load a path into the GUI input field."""
        self.path_input.setText(path)
        self._set_operation_state(True)

    def _current_path(self) -> str:
        return self.path_input.text().strip()

    def _begin_operation(self, cancellable: bool = True,
                         determinate: bool = False):
        """Disable buttons, show progress bar and Cancel."""
        self._set_operation_state(False)
        self.cancel_btn.setVisible(cancellable)
        self.progress_bar.setVisible(True)
        if determinate:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
        else:
            self.progress_bar.setRange(0, 0)

    def _finish_operation(self):
        """Re-enable buttons and hide progress bar."""
        self._set_operation_state(True)
        self.cancel_btn.setVisible(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.reset()
        self.cancel_btn.setText("Cancel")
        if self._worker is not None:
            try:
                if not self._worker.isRunning():
                    self._worker = None
            except RuntimeError:
                self._worker = None

    def _start_worker(self, worker: QThread):
        """Wire up and launch a worker, replacing any previous one."""
        self._dispose_worker()
        self._worker = worker
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _dispose_worker(self):
        """Cleanly discard a previous worker if one lingers."""
        worker = self._worker
        if worker is None:
            return
        self._worker = None
        try:
            if worker.isRunning():
                worker.wait(2000)
            worker.deleteLater()
        except RuntimeError:
            pass

    def _path_is_valid(self) -> bool:
        path = self._current_path()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Warning", f"Path not found: {path}")
            return False
        return os.path.isfile(path)

    def _is_current_worker(self, worker) -> bool:
        """Guard against stale completions from replaced workers."""
        return worker is not None and worker is self._worker

    def _on_tokenize(self):
        """Handle Tokenize (HF) button click."""
        if not self._path_is_valid():
            return
        self._begin_operation(determinate=True)
        self.token_text.setPlainText("Tokenizing...")
        worker = TokenizationWorkerWrapper(self._current_path(), use_hf=True)
        worker.progress.connect(self._on_progress)
        worker.done.connect(self._on_tokenization_complete)
        worker.failed.connect(self._on_operation_error)
        self._start_worker(worker)

    def _on_tokenize_fast(self):
        """Handle Tokenize (Fast/local) button click."""
        if not self._path_is_valid():
            return
        self._begin_operation(determinate=True)
        self.token_text.setPlainText("Tokenizing (custom tokenizer)...")
        worker = TokenizationWorkerWrapper(self._current_path(), use_hf=False)
        worker.progress.connect(self._on_progress)
        worker.done.connect(self._on_tokenization_complete)
        worker.failed.connect(self._on_operation_error)
        self._start_worker(worker)

    def _on_compress(self):
        """Handle Compress button click -- streams entirely off-thread."""
        if not self._path_is_valid():
            return
        self._begin_operation(determinate=True)
        self.token_text.setPlainText("Compressing...")
        worker = PathCompressionWorker(self._current_path(), "zlib")
        worker.progress.connect(self._on_progress)
        worker.done.connect(self._on_compression_complete)
        worker.failed.connect(self._on_operation_error)
        self._start_worker(worker)

    def _on_decompress(self):
        """Handle Decompress button click."""
        if not self.compressed_entry or \
                self.compressed_entry.get("data") is None:
            QMessageBox.information(
                self, "Info",
                "No compressed data available.\n"
                "Run Compress first (very large outputs are not kept "
                "in memory)."
            )
            return

        self._begin_operation(cancellable=False)
        self.token_text.setPlainText("Decompressing...")
        worker = DecompressionWorkerWrapper(
            self.compressed_entry["data"],
            self.compressed_entry.get("method", "zlib"))
        worker.done.connect(self._on_decompression_complete)
        worker.failed.connect(self._on_operation_error)
        self._start_worker(worker)

    def _on_cancel(self):
        """Request the running operation to stop."""
        if self._worker is not None:
            self._worker.cancel()
            self.cancel_btn.setEnabled(False)
            self.cancel_btn.setText("Cancelling...")
            self.token_text.append("\nCancelling after current chunk...")

    def _on_progress(self, done_bytes: int, total_bytes: int):
        """Update determinate progress from byte counts."""
        if total_bytes > 0:
            self.progress_bar.setValue(min(int(done_bytes * 100 / total_bytes), 100))

    def _on_tokenization_complete(self, result: dict):
        """Handle tokenization completion."""
        if self.sender() is not None and \
                not self._is_current_worker(self.sender()):
            return
        self._finish_operation()

        if "error" in result:
            QMessageBox.warning(self, "Warning", result["error"])
            return

        self.current_result = result
        tokens = result.get("preview_tokens", [])

        self.viewer.set_tokens(tokens)
        self.token_text.setPlainText(
            "\n".join(str(t) for t in tokens[:self.MAX_TOKENS_SHOWN])
        )
        note = ""
        if len(tokens) >= self.MAX_TOKENS_SHOWN or \
                result.get("original_tokens", 0) > len(tokens):
            note = (f"\n... (preview of {len(tokens)} tokens; "
                    f"{result.get('original_tokens', 0):,} total)")
        if note:
            self.token_text.append(note)
        if result.get("aborted"):
            self.token_text.append("(operation cancelled)")

        self.ratio_label.setText(
            f"Tokens: {result.get('original_tokens', 0):,} "
            f"(unique: {result.get('optimized_tokens', 0):,}, "
            f"duplicates removed: {result.get('duplicates_removed', 0):,})"
            + (" [unique count is a lower bound]"
               if result.get("unique_is_lower_bound") else "")
        )
        mem = result.get("memory", {})
        self.memory_label.setText(
            f"Method: {result.get('method', 'unknown')} | "
            f"RSS: {mem.get('rss_mb', 0):.1f} MB | "
            f"Peak RSS: {mem.get('peak_usage_mb', 0):.1f} MB | "
            f"Took: {result.get('elapsed_seconds', 0)}s"
        )
        self.tab_widget.setCurrentIndex(1)

    def _on_compression_complete(self, result: dict):
        """Handle compression completion."""
        if self.sender() is not None and \
                not self._is_current_worker(self.sender()):
            return
        self._finish_operation()

        if result.get("aborted"):
            self.token_text.setPlainText("Compression cancelled.")
            return
        if "error" in result:
            QMessageBox.warning(self, "Warning", result["error"])
            return

        original_size = result.get("original_size", 0)
        compressed_size = result.get("compressed_size", 0)
        ratio = result.get("ratio", 0)

        self.compressed_entry = {
            "data": result.get("data"),
            "method": result.get("method", "zlib"),
        }
        self.current_result = {
            "original_size": original_size,
            "compressed_size": compressed_size,
            "ratio": ratio,
            "sha256": result.get("sha256"),
        }

        self.compression_viewer.set_stats(original_size, compressed_size)
        self.ratio_label.setText(f"Compression ratio: {ratio:.2%}")
        self.size_label.setText(
            f"Original: {format_bytes(original_size)} | "
            f"Compressed: {format_bytes(compressed_size)}"
        )
        retained_note = "" if result.get("data") is not None else \
            "\n(output too large to keep in RAM; sizes shown are final)"
        self.token_text.setPlainText(
            f"Compressed {format_bytes(original_size)} -> "
            f"{format_bytes(compressed_size)} ({ratio:.2%}).{retained_note}\n"
            f"Click 'Decompress' to verify roundtrip."
        )
        self.tab_widget.setCurrentIndex(1)

    def _on_decompression_complete(self, result: dict):
        """Handle decompression completion."""
        if self.sender() is not None and \
                not self._is_current_worker(self.sender()):
            return
        self._finish_operation()

        size = result.get("decompressed_size", 0)
        verified = result.get("verified")
        status = ("roundtrip VERIFIED" if verified
                  else "roundtrip FAILED" if verified is False
                  else "not verified")
        sample = result.get("sample", "")
        self.token_text.setPlainText(
            f"Decompressed ({format_bytes(size)}) - {status}:\n\n"
            f"{sample[:self.SAMPLE_TEXT_LIMIT]}"
        )

    def _on_operation_error(self, error_msg: str):
        """Handle operation error."""
        log_error("error", error_msg, "gui_operation")
        self._finish_operation()
        QMessageBox.critical(self, "Error", f"An error occurred: {error_msg}")

    def _set_operation_state(self, enabled: bool):
        """Enable/disable operation buttons."""
        for btn in (self.tokenize_btn, self.tokenize_local_btn,
                    self.compress_btn, self.decompress_btn):
            btn.setEnabled(enabled)
        self.cancel_btn.setEnabled(True)
        if enabled:
            self.cancel_btn.setText("Cancel")

def launch_gui(initial_path: str = None):
    """Launch the GUI application, optionally pre-loading a path.

    Args:
        initial_path: Optional file/folder path to preload.
    """
    app = QApplication(sys.argv)
    window = TokenizerGUI()
    if initial_path:
        window.process_path(initial_path)
    window.show()
    return app.exec()

def main():
    """Run the GUI application."""
    sys.exit(launch_gui())

if __name__ == "__main__":
    main()
