"""Viewer modules for displaying tokenization results.

Displays token information, compression stats, and optimization results.
"""

class TokenViewer:
    """Viewer for token display."""

    def __init__(self):
        self.tokens = []
        self.idx = 0

    def set_tokens(self, tokens: list) -> None:
        """Set the token list to display."""
        self.tokens = tokens
        self.idx = 0

    def get_current(self) -> str:
        """Get the current token being displayed."""
        if not self.tokens or self.idx >= len(self.tokens):
            return ""
        return str(self.tokens[self.idx])

    def next_token(self) -> str:
        """Move to next token and return it."""
        if self.tokens and self.idx < len(self.tokens):
            self.idx += 1
            return self.get_current()
        return ""

    def prev_token(self) -> str:
        """Move to previous token and return it."""
        if self.tokens and self.idx > 0:
            self.idx -= 1
            return self.get_current()
        return ""

class CompressionViewer:
    """Viewer for compression/decompression statistics."""

    def __init__(self):
        self.original_size = 0
        self.compressed_size = 0
        self.ratio = 0

    def set_stats(self, original: int, compressed: int) -> None:
        """Set compression statistics."""
        self.original_size = original
        self.compressed_size = compressed
        if original > 0:
            self.ratio = compressed / original
        else:
            self.ratio = 0

    def get_ratio_str(self) -> str:
        """Get compression ratio as formatted string."""
        return f"{self.ratio:.2%}"

    def get_original_str(self) -> str:
        """Get original size formatted."""
        return f"{self.original_size}"

    def get_compressed_str(self) -> str:
        """Get compressed size formatted."""
        return f"{self.compressed_size}"
