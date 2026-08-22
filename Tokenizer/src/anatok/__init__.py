"""anatok -- fast, memory-bounded AI text tokenization toolkit.

Streaming HuggingFace tokenization with compression, deduplication,
real process-memory tracking and Arrow (.pyarrow) results export.

Submodules are imported lazily by the entry points, so ``import anatok``
stays cheap. GUI support (PyQt6) is an optional extra: ``anatok[gui]``.
"""
from typing import TYPE_CHECKING

__version__ = "0.1.0"

__all__ = [
    "main",
    "gui_main",
    "get_tokenization_manager",
    "get_compression_manager",
    "quick_tokenize",
    "__version__",
]

if TYPE_CHECKING:
    from .main import gui_main, main
    from .managers import (
        get_compression_manager,
        get_tokenization_manager,
        quick_tokenize,
    )

def __getattr__(name):
    """PEP 562 lazy attribute access for the public convenience API."""
    if name == "main":
        from .main import main
        return main
    if name == "gui_main":
        from .main import gui_main
        return gui_main
    if name in ("get_tokenization_manager", "get_compression_manager",
                "quick_tokenize"):
        import importlib

        module = importlib.import_module(".managers", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
