"""Error warehouse for anatok.

Centralized error handling and logging. All errors from throughout the
application are caught and logged to this module to prevent unhandled
exceptions and maintain GUI responsiveness.

Logs are written to a platform-appropriate application-data directory
(never the current working directory):

- Windows: ``%LOCALAPPDATA%/anatok/logs/tokenizer_errors.log``
- macOS:   ``~/Library/Logs/anatok/tokenizer_errors.log``
- Linux:   ``$XDG_STATE_HOME/anatok/tokenizer_errors.log``
           (default ``~/.local/state/anatok/``)

Override with the ``ANATOK_LOG_DIR`` environment variable.
"""
import logging
import os
import sys
import tempfile
from datetime import datetime

LOG_FILENAME = "tokenizer_errors.log"

def default_log_dir() -> str:
    """Return the platform-appropriate anatok log directory."""
    override = os.environ.get("ANATOK_LOG_DIR")
    if override:
        return override
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(
            os.path.expanduser("~"), "AppData", "Local")
        return os.path.join(base, "anatok", "logs")
    if sys.platform == "darwin":
        return os.path.join(os.path.expanduser("~"), "Library", "Logs",
                            "anatok")
    state_home = os.environ.get("XDG_STATE_HOME") or os.path.join(
        os.path.expanduser("~"), ".local", "state")
    return os.path.join(state_home, "anatok")

def default_log_file() -> str:
    """Return a writable platform-appropriate log file path."""
    directory = default_log_dir()
    try:
        os.makedirs(directory, exist_ok=True)
        probe = os.path.join(directory, LOG_FILENAME)
        with open(probe, "a", encoding="utf-8"):
            pass
        return probe
    except OSError:
        return os.path.join(tempfile.gettempdir(), f"anatok_{LOG_FILENAME}")

class ErrorWarehouse:
    """Central error logging and handling utility."""

    def __init__(self, log_file=None):
        self.log_file = log_file or default_log_file()
        self._setup_logger()
        self.error_counts = {}
        self.operation_counts = {}

    def _setup_logger(self):
        """Configure the logging system (idempotent: no duplicate handlers)."""
        logger = logging.getLogger("TokenizerError")
        logger.setLevel(logging.DEBUG)

        if not logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.WARNING)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            try:
                file_handler = logging.FileHandler(self.log_file,
                                                   encoding="utf-8")
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except OSError as e:
                logger.warning("File logging disabled (%s); "
                               "console-only mode.", e)

        self.logger = logger

    def log(self, level: str, message: str, operation: str = "unknown"):
        """Log an error or informational message.

        Args:
            level: Log level ("info", "warning", "error", "critical").
            message: The error message.
            operation: The operation where the error occurred.
        """
        timestamp = datetime.now().isoformat()
        entry = f"[{timestamp}] [{level.upper()}] [{operation}] {message}"
        self.logger.log(getattr(logging, level.upper(), logging.INFO), entry)

        key = f"{operation}:{level}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        self.operation_counts[operation] = self.operation_counts.get(operation, 0) + 1

    def info(self, message: str, operation: str = "unknown"):
        """Log an informational message."""
        self.log("info", message, operation)

    def warning(self, message: str, operation: str = "unknown"):
        """Log a warning."""
        self.log("warning", message, operation)

    def error(self, message: str, operation: str = "unknown"):
        """Log an error."""
        self.log("error", message, operation)

    def critical(self, message: str, operation: str = "unknown"):
        """Log a critical error."""
        self.log("critical", message, operation)

    def get_stats(self) -> dict:
        """Get error statistics dictionary."""
        return {
            "error_counts": dict(self.error_counts),
            "operation_counts": dict(self.operation_counts),
            "log_file": self.log_file,
        }

    def clear_stats(self):
        """Clear accumulated statistics."""
        self.error_counts.clear()
        self.operation_counts.clear()

error_warehouse = ErrorWarehouse()

def log_error(level: str, message: str, operation: str = "unknown"):
    """Convenience function to log errors globally."""
    error_warehouse.log(level, message, operation)

def log_info(message: str, operation: str = "unknown"):
    """Convenience function to log info globally."""
    error_warehouse.info(message, operation)

def log_warning(message: str, operation: str = "unknown"):
    """Convenience function to log warnings globally."""
    error_warehouse.warning(message, operation)

def log_critical(message: str, operation: str = "unknown"):
    """Convenience function to log critical errors globally."""
    error_warehouse.critical(message, operation)
