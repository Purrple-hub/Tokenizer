"""Tokenization modules for the Tokenizer project.

HuggingFace Tokenizers integration plus classic fallback algorithms,
optimized for AI workloads.
"""

import os
import threading
from typing import Iterator, Optional, List, Dict, Any

from tokenizers import Tokenizer, Encoding

from .utils import iter_file_blocks

class HFTokenizerWrapper:
    """Wrapper for HuggingFace Tokenizers library."""

    def __init__(self, model_name: str = "bert-base-uncased",
                 tokenizer_type: str = "bert",
                 use_fast: bool = True):
        self.model_name = model_name
        self.tokenizer_type = tokenizer_type
        self.use_fast = use_fast
        self._tokenizer: Optional[Tokenizer] = None
        self._lock = threading.Lock()
        self._load_tokenizer()

    def _load_tokenizer(self):
        """Load HuggingFace tokenizer model, falling back to a small local BPE."""
        try:
            self._tokenizer = Tokenizer.from_pretrained(self.model_name)
        except Exception:
            self._create_basic_tokenizer()

    def _create_basic_tokenizer(self):
        """Create a basic offline BPE tokenizer as fallback."""
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel

        self._tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self._tokenizer.pre_tokenizer = ByteLevel()
        trainer = BpeTrainer(
            vocab_size=10000,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        )
        corpus = [
            "hello world test training data",
            "tokenizer project fast ai optimization",
            "the quick brown fox jumps over the lazy dog",
        ]
        self._tokenizer.train_from_iterator(corpus, trainer)

    @property
    def is_fallback(self) -> bool:
        """True if the pretrained model could not be loaded."""
        return self._tokenizer is not None and self.model_name not in str(
            getattr(self._tokenizer, "model", "")
        )

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        with self._lock:
            if self._tokenizer is None:
                self._load_tokenizer()
            return self._tokenizer.encode(text).ids

    def encode_full(self, text: str):
        """Encode text and return the full Encoding object (ids, tokens, offsets)."""
        with self._lock:
            if self._tokenizer is None:
                self._load_tokenizer()
            return self._tokenizer.encode(text)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode multiple texts to token IDs."""
        with self._lock:
            if self._tokenizer is None:
                self._load_tokenizer()
            return [e.ids for e in self._tokenizer.encode_batch(texts)]

    def encode_batch_full(self, texts: List[str]) -> List[Encoding]:
        """Encode multiple texts and return full Encoding objects."""
        with self._lock:
            if self._tokenizer is None:
                self._load_tokenizer()
            return list(self._tokenizer.encode_batch(texts))

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        with self._lock:
            if self._tokenizer is None:
                self._load_tokenizer()
            return self._tokenizer.decode(token_ids)

    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary mapping (token string -> id)."""
        with self._lock:
            if self._tokenizer is None:
                self._load_tokenizer()
            return self._tokenizer.get_vocab()

    def id_to_token(self, token_id: int) -> str:
        """Convert a single token id back to its token string."""
        with self._lock:
            if self._tokenizer is None:
                self._load_tokenizer()
            token = self._tokenizer.id_to_token(token_id)
            return token if token is not None else f"<id_{token_id}>"

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        with self._lock:
            if self._tokenizer is None:
                self._load_tokenizer()
            return self._tokenizer.get_vocab_size()

_hf_tokenizer: Optional[HFTokenizerWrapper] = None
_hf_lock = threading.Lock()
_tokenizer_cache: Dict[str, HFTokenizerWrapper] = {}

def get_hf_tokenizer(model_name: str = "bert-base-uncased") -> HFTokenizerWrapper:
    """Get a cached HF tokenizer instance (one per model name)."""
    with _hf_lock:
        if model_name not in _tokenizer_cache:
            _tokenizer_cache[model_name] = HFTokenizerWrapper(model_name)
        return _tokenizer_cache[model_name]

HF_SUB_CHUNK_SIZE = 256 * 1024

class TextChunkStreamer:
    """Decode byte blocks into UTF-8 pieces of ~target_size characters.

    Splits preferentially at newlines, then spaces, so words are not cut
    in half across piece boundaries. Binary content falls back to hard
    cuts. Memory use stays bounded regardless of input file size.
    """

    def __init__(self, target_size: int = HF_SUB_CHUNK_SIZE):
        self.target_size = max(int(target_size), 4096)
        self._buf = ""
        self.bytes_fed = 0

    def feed(self, block: bytes) -> List[str]:
        """Feed one bytes block; returns completed text pieces."""
        self.bytes_fed += len(block)
        buf = self._buf + block.decode("utf-8", errors="replace")
        pieces: List[str] = []
        target = self.target_size
        while len(buf) >= target:
            cut = buf.rfind("\n", 0, target)
            if cut < 0:
                cut = buf.rfind(" ", 0, target)
            if cut < 0:
                cut = target - 1
            pieces.append(buf[:cut + 1])
            buf = buf[cut + 1:]
        self._buf = buf
        return pieces

    def finish(self) -> List[str]:
        """Return any trailing buffered content."""
        rest = self._buf
        self._buf = ""
        return [rest] if rest else []

def stream_encode_file(path: str,
                       model: Optional[str] = None,
                       should_abort=None,
                       progress_cb=None,
                       encoding_cb=None) -> Iterator[Encoding]:
    """Stream a file through an HF tokenizer, yielding Encoding objects.

    Memory usage is bounded by chunk + batch size rather than file size.

    Args:
        path: File to tokenize.
        model: HF model name (defaults to bert-base-uncased).
        should_abort: Optional callable returning True to stop early.
        progress_cb: Optional callable invoked as progress_cb(bytes_done).
        encoding_cb: Optional callable invoked with each Encoding before it
            is yielded (used for streaming results export).

    Yields:
        tokenizers.Encoding objects for each text chunk.
    """
    wrapper = get_hf_tokenizer(model) if model else get_hf_tokenizer()
    streamer = TextChunkStreamer()

    pending: List[str] = []
    plen = 0

    def flush():
        nonlocal pending, plen
        if pending:
            for enc in wrapper.encode_batch_full(pending):
                if encoding_cb is not None:
                    try:
                        encoding_cb(enc)
                    except Exception:
                        pass
                yield enc
            pending = []
            plen = 0

    for block in iter_file_blocks(path):
        if progress_cb is not None:
            try:
                progress_cb(streamer.bytes_fed + len(block))
            except Exception:
                pass
        aborted = should_abort is not None and should_abort()
        if aborted:
            return
        for piece in streamer.feed(block):
            pending.append(piece)
            plen += len(piece)
            if plen >= HF_SUB_CHUNK_SIZE * 4:
                yield from flush()
    for piece in streamer.finish():
        pending.append(piece)
    yield from flush()

def iter_word_group_tokens(path: str, group_size: int = 256,
                           should_abort=None,
                           progress_cb=None,
                           group_cb=None) -> Iterator[str]:
    """Stream a file as word-group tokens (custom tokenizer semantics).

    Mirrors text_tokenize()'s grouping but processes incrementally so only
    one group plus the current read block is held in memory.

    Args:
        path: File to tokenize.
        group_size: Words per emitted token group.
        should_abort: Optional callable returning True to stop early.
        progress_cb: Optional callable invoked as progress_cb(bytes_done).
        group_cb: Optional callable invoked with each emitted group string
            (used for streaming results export).

    Yields:
        Grouped token strings.
    """
    streamer = TextChunkStreamer()
    word_buf: List[str] = []

    def drain(final: bool):
        while len(word_buf) >= group_size:
            group = " ".join(word_buf[:group_size])
            del word_buf[:group_size]
            if group_cb is not None:
                try:
                    group_cb(group)
                except Exception:
                    pass
            yield group
        if final and word_buf:
            group = " ".join(word_buf)
            word_buf.clear()
            if group_cb is not None:
                try:
                    group_cb(group)
                except Exception:
                    pass
            yield group

    for block in iter_file_blocks(path):
        if should_abort is not None and should_abort():
            return
        if progress_cb is not None:
            try:
                progress_cb(streamer.bytes_fed + len(block))
            except Exception:
                pass
        for piece in streamer.feed(block):
            word_buf.extend(piece.split())
            yield from drain(False)
    for piece in streamer.finish():
        word_buf.extend(piece.split())
        yield from drain(False)
    yield from drain(True)

def count_file_tokens_hf(path: str,
                         model: Optional[str] = None,
                         preview_limit: int = 500,
                         unique_cap: int = 2_000_000,
                         should_abort=None,
                         progress_cb=None,
                         encoding_cb=None) -> Dict[str, Any]:
    """Tokenize a file in streaming mode and return aggregate statistics.

    Never materializes the full token/id lists: unique tokens are tracked
    in a set bounded by the vocabulary (or unique_cap), and only the first
    preview_limit token strings are retained for display.

    Args:
        path: File to tokenize.
        model: HF model name.
        preview_limit: Number of leading token strings to keep.
        unique_cap: Upper bound on the unique-token set.
        should_abort: Optional callable returning True to stop early.
        progress_cb: Optional callable invoked as progress_cb(bytes_done).
        encoding_cb: Optional callable invoked with each Encoding (used for
            streaming results export).

    Returns:
        Dict with total_tokens, unique_tokens, unique_capped, preview,
        aborted and vocab_size.
    """
    total = 0
    unique: set = set()
    capped = False
    preview: List[str] = []
    aborted = False

    for encoding in stream_encode_file(
            path, model=model, should_abort=should_abort,
            progress_cb=progress_cb, encoding_cb=encoding_cb):
        total += len(encoding.ids)
        if not capped:
            unique.update(encoding.tokens)
            if len(unique) > unique_cap:
                capped = True
                unique.clear()
                unique.add("__capped__")
        if len(preview) < preview_limit:
            preview.extend(encoding.tokens[:preview_limit - len(preview)])
        if should_abort is not None and should_abort():
            aborted = True
            break

    wrapper = get_hf_tokenizer(model) if model else get_hf_tokenizer()
    return {
        "total_tokens": total,
        "unique_tokens": len(unique),
        "unique_capped": capped,
        "preview": preview,
        "aborted": aborted,
        "vocab_size": wrapper.get_vocab_size(),
    }

def tokenize_with_hf(text: str, model: str = "bert-base-uncased",
                     return_offsets: bool = False) -> Dict[str, Any]:
    """Convenience function to tokenize using HuggingFace.

    Args:
        text: Input text to tokenize.
        model: HF model name or path.
        return_offsets: Whether to include character offset mappings.

    Returns:
        Dictionary with input_ids, tokens, vocab_size and optional offsets.
    """
    wrapper = get_hf_tokenizer(model)
    encoding = wrapper.encode_full(text)

    result = {
        'input_ids': encoding.ids,
        'tokens': encoding.tokens,
        'vocab_size': wrapper.get_vocab_size(),
        'model': model,
    }

    if return_offsets:
        result['offset_mapping'] = list(encoding.offsets)

    return result

def benchmark_tokenization(texts: List[str], iterations: int = 10) -> Dict[str, Any]:
    """Benchmark different tokenization methods.

    Args:
        texts: List of texts to tokenize.
        iterations: Number of iterations per method.

    Returns:
        Benchmark results dictionary keyed by method name.
    """
    import time

    results = {}
    n_items = max(len(texts), 1)

    hf_start = time.perf_counter()
    for _ in range(iterations):
        for text in texts:
            get_hf_tokenizer().encode(text)
    hf_time = time.perf_counter() - hf_start
    results['hf_tokenizer'] = {
        'total_time': hf_time,
        'per_call': hf_time / (iterations * n_items),
    }

    byte_start = time.perf_counter()
    for _ in range(iterations):
        for text in texts:
            byte_based_tokenize(text.encode('utf-8'))
    byte_time = time.perf_counter() - byte_start
    results['byte_based'] = {
        'total_time': byte_time,
        'per_call': byte_time / (iterations * n_items),
    }

    custom_start = time.perf_counter()
    for _ in range(iterations):
        for text in texts:
            text_tokenize(text)
    custom_time = time.perf_counter() - custom_start
    results['custom'] = {
        'total_time': custom_time,
        'per_call': custom_time / (iterations * n_items),
    }

    return results

def byte_based_tokenize(data: bytes, max_tokens: int = None) -> list:
    """Tokenize bytes into fixed-size chunks."""
    tokens = []
    chunk_size = 256
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        tokens.append(chunk)
        if max_tokens and len(tokens) >= max_tokens:
            break
    return tokens

def text_tokenize(text: str, max_tokens: int = None) -> list:
    """Tokenize text into groups of up to 256 words."""
    words = text.split()
    tokens = []
    for i in range(0, len(words), 256):
        chunk = words[i:i + 256]
        tokens.append(" ".join(chunk))
        if max_tokens and len(tokens) >= max_tokens:
            break
    return tokens

def tokenize_file(filepath: str, max_tokens: int = None) -> dict:
    """Tokenize a file's content using constant-memory streaming.

    Args:
        filepath: Path to the file.
        max_tokens: Stop after this many tokens (optional).

    Returns:
        Dict with a token sample, total count and file metadata. Only the
        first ~50 tokens are retained regardless of file size.
    """
    from .utils import format_bytes

    size = os.path.getsize(filepath)
    sample: List[str] = []
    total = 0
    stopped = False

    for encoding in stream_encode_file(filepath):
        total += len(encoding.ids)
        if len(sample) < 50:
            sample.extend(encoding.tokens[:50 - len(sample)])
        if max_tokens and total >= max_tokens:
            stopped = True
            break

    return {
        'tokens': sample,
        'token_count': total,
        'file_size': size,
        'formatted_size': format_bytes(size),
        'filepath': filepath,
        'truncated': stopped or max_tokens is not None,
    }

def merge_tokens(token_lists: list) -> list:
    """Merge multiple token lists into one."""
    merged = []
    for tl in token_lists:
        merged.extend(tl)
    return merged

def detokenize(tokens: list) -> str:
    """Convert token list back to text string."""
    return " ".join(str(t) for t in tokens)
