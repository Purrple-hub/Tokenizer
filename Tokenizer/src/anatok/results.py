"""Results export for anatok.

Every tokenization run can persist its full output so it can be inspected
later without re-tokenizing:

- ``<name>_<timestamp>.pyarrow`` -- the complete token stream as an Apache
  Arrow IPC (feather v2) file, written batch-by-batch while tokenizing so
  memory stays constant regardless of file size. Schema metadata carries
  run provenance.
- ``<name>_<timestamp>.json`` -- machine-readable summary of the run.
- ``<name>_<timestamp>.md`` -- human-readable report.

The results directory defaults to ``./results`` and can be overridden via
the ``ANATOK_RESULTS_DIR`` environment variable or an explicit argument.

Reading a result back::

    import pyarrow as pa          # or: import pyarrow.feather as feather
    table = feather.read_table("results/report_20260823-120000.pyarrow")
    table.to_pandas().head()      # columns: doc, seq, token_id, token
"""
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

RESULTS_DIR_ENV = "ANATOK_RESULTS_DIR"
DEFAULT_RESULTS_DIRNAME = "results"
ARROW_SUFFIX = ".pyarrow"

def get_results_dir(results_dir: Optional[str] = None) -> str:
    """Resolve (and create) the results directory.

    Precedence: explicit argument > ANATOK_RESULTS_DIR env var > ./results.
    """
    base = results_dir or os.environ.get(RESULTS_DIR_ENV) \
        or os.path.join(os.getcwd(), DEFAULT_RESULTS_DIRNAME)
    path = os.path.abspath(base)
    os.makedirs(path, exist_ok=True)
    return path

def result_stem(source_path: str) -> str:
    """Build a unique-per-run file stem from a source path."""
    name = os.path.splitext(os.path.basename(source_path))[0] or "input"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    return f"{name}_{stamp}"

class TokenArrowWriter:
    """Stream tokens into an Arrow IPC file with constant memory use.

    Rows are buffered in small Python lists and flushed to Arrow record
    batches every BATCH_ROWS rows, so neither the token list nor the Arrow
    buffers ever grow proportionally to the input size.
    """

    BATCH_ROWS = 8192

    def __init__(self, filepath: str, source_path: str = "",
                 metadata: Optional[Dict[str, Any]] = None):
        import pyarrow as pa

        self.filepath = filepath
        self.row_count = 0
        self._pa = pa

        meta = {
            "created": datetime.now().isoformat(),
            "source_path": os.path.abspath(source_path) if source_path else "",
            "generator": "anatok",
        }
        if metadata:
            meta.update({k: str(v) for k, v in metadata.items()})

        schema = pa.schema([
            ("doc", pa.string()),
            ("seq", pa.int64()),
            ("token_id", pa.int64()),
            ("token", pa.string()),
        ], metadata={b"anatok_meta": json.dumps(meta).encode("utf-8")})
        self._schema = schema

        parent = os.path.dirname(os.path.abspath(filepath))
        os.makedirs(parent, exist_ok=True)

        self._sink = open(filepath, "wb")
        try:
            self._writer = pa.ipc.new_file(self._sink, schema)
        except Exception:
            self._sink.close()
            raise

        self._docs: list = []
        self._seqs: list = []
        self._ids: list = []
        self._tokens: list = []
        self._closed = False

    def add_row(self, doc: str, token_id: Optional[int], token: str):
        """Buffer one token row."""
        self._docs.append(doc)
        self._seqs.append(self.row_count)
        self._ids.append(token_id)
        self._tokens.append(token)
        self.row_count += 1
        if len(self._tokens) >= self.BATCH_ROWS:
            self.flush()

    def add_encoding(self, doc: str, encoding) -> int:
        """Add all tokens of a tokenizers.Encoding object."""
        ids = getattr(encoding, "ids", None) or []
        tokens = getattr(encoding, "tokens", None) or []
        for tid, tok in zip(ids, tokens):
            self.add_row(doc, tid, tok)
        return len(tokens)

    def add_group(self, doc: str, group: str):
        """Add one custom-tokenizer word-group (token id unknown -> null)."""
        self.add_row(doc, None, group)

    def add_tokens(self, doc: str, tokens, ids=None):
        """Add plain token strings with optional matching id list."""
        for i, tok in enumerate(tokens):
            self.add_row(doc, ids[i] if ids is not None else None, str(tok))

    def flush(self):
        """Write buffered rows as one record batch."""
        if not self._tokens:
            return
        pa = self._pa
        batch = pa.record_batch(
            [
                pa.array(self._docs, type=pa.string()),
                pa.array(self._seqs, type=pa.int64()),
                pa.array(self._ids, type=pa.int64()),
                pa.array(self._tokens, type=pa.string()),
            ],
            schema=self._schema,
        )
        self._writer.write_batch(batch)
        self._docs.clear()
        self._seqs.clear()
        self._ids.clear()
        self._tokens.clear()

    def close(self):
        """Flush pending rows and finalize the Arrow file footer."""
        if self._closed:
            return
        self._closed = True
        try:
            self.flush()
            self._writer.close()
        finally:
            self._sink.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

def read_arrow_result(filepath: str):
    """Read a .pyarrow result back into a pyarrow.Table (helper for tests)."""
    import pyarrow.feather as feather
    return feather.read_table(filepath)

_DROP_KEYS = ("data",)

def _clean_for_export(result: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in result.items() if k not in _DROP_KEYS}

def export_json_summary(result: Dict[str, Any], filepath: str) -> bool:
    """Write the machine-readable summary next to the arrow file."""
    payload = {
        "anatok_version": _anatok_version(),
        "generated": datetime.now().isoformat(),
        "result": _clean_for_export(result),
    }
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        return True
    except Exception:
        return False

def format_markdown_report(result: Dict[str, Any]) -> str:
    """Render a human-readable Markdown report for a tokenization run."""
    lines = ["# anatok Tokenization Report", ""]
    lines.append(f"- **File:** `{result.get('file', 'N/A')}`")
    lines.append(f"- **Method:** {result.get('method', 'unknown')}")
    lines.append(f"- **Model:** {result.get('model', '-')}")
    lines.append(f"- **File size:** "
                 f"{result.get('file_size', 0):,} bytes")
    lines.append("")
    lines.append("## Tokens")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Total tokens | {result.get('original_tokens', 0):,} |")
    lines.append(f"| Unique tokens | {result.get('optimized_tokens', 0):,} |")
    lines.append(f"| Duplicates removed | "
                 f"{result.get('duplicates_removed', 0):,} |")
    ratio = result.get("compression_ratio", 0) or 0
    lines.append(f"| Dedup ratio | {ratio:.2%} |")
    if result.get("unique_is_lower_bound"):
        lines.append("| Note | unique count is a lower bound (capped) |")
    if result.get("aborted"):
        lines.append("| Status | ABORTED (partial results) |")
    lines.append(f"| Elapsed | {result.get('elapsed_seconds', 0)} s |")

    comp = result.get("compressed")
    if comp:
        lines += ["", "## Compression (zlib)", ""]
        lines.append(f"- Original: {comp.get('original_size', 0):,} bytes")
        lines.append(f"- Compressed: {comp.get('compressed_size', 0):,} bytes")
        cratio = comp.get("ratio", 0) or 0
        lines.append(f"- Ratio: {cratio:.2%}")
        if "roundtrip_ok" in result:
            ok = result.get("roundtrip_ok")
            lines.append(f"- Roundtrip verified: "
                         f"{'yes' if ok else ('no' if ok is False else 'n/a')}")
            note = result.get("roundtrip_note")
            if note:
                lines.append(f"- Roundtrip note: {note}")

    mem = result.get("memory")
    if mem:
        lines += ["", "## Memory", ""]
        lines.append(f"- Peak RSS: {mem.get('peak_usage_mb', 0):.1f} MB")
        lines.append(f"- Final RSS: {mem.get('rss_mb', 0):.1f} MB")

    preview = result.get("preview_tokens") or []
    if preview:
        shown = preview[:50]
        text = " ".join(str(t) for t in shown)
        if len(text) > 1200:
            text = text[:1200] + "..."
        lines += ["", "## Token Preview", "", "```text", text,
                  "```", ""]
        if len(preview) > len(shown):
            lines.append(f"(showing first {len(shown)} of "
                         f"{len(preview)} preview tokens)")
    exports = result.get("exports")
    if isinstance(exports, dict):
        lines += ["", "## Artifacts", ""]
        for kind, p in exports.items():
            if p:
                lines.append(f"- `{kind}`: `{p}`")
    lines.append("")
    return "\n".join(lines)

def export_markdown_report(result: Dict[str, Any], filepath: str) -> bool:
    """Write the human-readable report next to the arrow file."""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(format_markdown_report(result))
        return True
    except Exception:
        return False

def export_result_files(result: Dict[str, Any],
                        results_dir: Optional[str] = None,
                        stem: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Write the JSON + Markdown sidecars for a completed run.

    The .pyarrow file itself is streamed by TokenArrowWriter during
    processing; its expected path is derived from the same stem.

    Returns:
        Mapping {"arrow": path_or_None, "json": ..., "md": ...}.
    """
    rdir = get_results_dir(results_dir)
    src = str(result.get("file") or "")
    stem = stem or result_stem(src or "run")
    arrow_path = os.path.join(rdir, stem + ARROW_SUFFIX)
    json_path = os.path.join(rdir, stem + ".json")
    md_path = os.path.join(rdir, stem + ".md")

    exports: Dict[str, Optional[str]] = {
        "arrow": arrow_path if os.path.exists(arrow_path) else None,
    }

    enriched = _clean_for_export(result)
    enriched["exports"] = {
        **exports,
        "json": json_path,
        "md": md_path,
    }

    exports["json"] = json_path \
        if export_json_summary(enriched, json_path) else None
    exports["md"] = md_path \
        if export_markdown_report(enriched, md_path) else None
    return exports

def export_folder_summary(summary: Dict[str, Any],
                          results_dir: Optional[str] = None,
                          stem: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Write JSON + Markdown summaries for a folder-wide run."""
    rdir = get_results_dir(results_dir)
    base = stem or \
        datetime.now().strftime("folder_%Y%m%d-%H%M%S")
    json_path = os.path.join(rdir, base + ".json")
    md_path = os.path.join(rdir, base + ".md")

    ok_json = False
    try:
        slim = dict(summary)
        slim["results"] = [_clean_for_export(r) for r in
                           summary.get("results", [])]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"anatok_version": _anatok_version(),
                       "generated": datetime.now().isoformat(),
                       "summary": slim}, f, indent=2, default=str)
        ok_json = True
    except Exception:
        pass

    lines = ["# anatok Folder Run Summary", ""]
    lines.append(f"- **Path:** `{summary.get('path', '')}`")
    lines.append(f"- **Operation:** {summary.get('operation', '')}")
    lines.append(f"- **Files:** {summary.get('total_files', 0)} total, "
                 f"{summary.get('processed', 0)} processed, "
                 f"{summary.get('skipped', 0)} skipped, "
                 f"{summary.get('errors', 0)} errors")
    lines += ["", "| File | Tokens | Unique | Ratio |", "|---|---:|---:|---:|"]
    for r in summary.get("results", []):
        name = os.path.basename(str(r.get("file", "?")))
        if "error" in r:
            lines.append(f"| {name} | ERROR | - | - |")
        elif r.get("skipped"):
            lines.append(f"| {name} | skipped | - | - |")
        else:
            lines.append(
                f"| {name} | {r.get('original_tokens', 0):,} "
                f"| {r.get('optimized_tokens', 0):,} "
                f"| {(r.get('compression_ratio') or 0):.2%} |")
    lines.append("")
    ok_md = False
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        ok_md = True
    except Exception:
        pass

    return {"json": json_path if ok_json else None,
            "md": md_path if ok_md else None}

def _anatok_version() -> str:
    try:
        from importlib.metadata import version
        return version("anatok")
    except Exception:
        try:
            from . import __version__
            return __version__
        except Exception:
            return "unknown"
