# ANATOK — Full Project Dossier

**Repository:** [Purrple-hub/Tokenizer](https://github.com/Purrple-hub/Tokenizer)  
**Commit:** `bb80911a54d55629de9791e5583d0e2024acf5a8` (Aug 22, 2026)  
**Audit Date:** 2026-08-23

---

## README (Executive Summary)

### What This Is

**Anatok** is a memory-bounded text tokenization toolkit with streaming, compression, deduplication, and Arrow (.pyarrow) results export. It delegates actual tokenization to HuggingFace's `tokenizers` library (bert-base-uncased by default), with an offline fallback that trains a tiny BPE on three hardcoded sentences when the hub is unreachable.

**What it is NOT:** A from-scratch BPE implementation to study. The repo name "Tokenizer" is misleading advertising — this is a systems-plumbing education: streaming, chunk-boundary management, cooperative cancellation, singleton lifecycle control, and cross-platform memory introspection.

### Key Features

- **Constant-memory streaming:** Files processed in ~4 MB blocks, never materializing the full token stream in RAM
- **Output-size-based compression retention:** Keeps compressed payload only if OUTPUT fits under 32 MB — not the input
- **Multiple compression backends:** zlib (default), LZ77 v2 (corrected), Huffman with incremental bit packing, byte-pair RLE
- **Real-RSS memory tracking:** psutil → Win32 GetProcessMemoryInfo → Unix getrusage fallback chain
- **PyQt6 GUI** with all heavy operations off the UI thread, cooperative cancellation, and stale-worker guards
- **Results export:** Token stream as Arrow IPC (.pyarrow), plus JSON and Markdown sidecars

### Installation

```bash
pip install "anatok[gui]"  # with GUI
# or
pip install anatok         # CLI only
```

Runtime dependencies: `tokenizers`, `psutil`, `pyarrow`

### Quick Start

```bash
anatok path/to/file.txt          # CLI
anatok-gui                       # GUI
```

### The Honest Truth

This is a **prototype with an unusually good skeleton and an unusually dishonest surface layer.** The streaming architecture is real and works. Files are genuinely processed in constant memory. Cancellation genuinely takes effect between chunks. Arrow results export genuinely streams batch-by-batch.

But the project has historically spent its energy polishing the tokenization pipeline while shipping broken plumbing around it. Several bugs were live until the recent restructuring: phantom imports, log files written to CWD, CLI flags silently discarded, a test module that did literally nothing, and an `is_fallback` property that has never once returned a correct answer.

**Verdict:** The bones are worth keeping. Several organs need transplanting.

---

## RESEARCH PAPER (Full Dossier)

*The following is the complete research dossier from the 2026-08-23 audit, reproduced verbatim with light Markdown formatting.*

---

### 0. TL;DR VERDICT

This is a prototype with an unusually good skeleton and an unusually dishonest surface layer.

The streaming architecture is real and it works. Files are genuinely processed in constant memory. Cancellation genuinely takes effect between chunks. The Arrow results export genuinely streams batch-by-batch without ever materializing the full token stream. Most hobby projects never get the memory model right at all; this one did, repeatedly, on purpose.

But the project has historically spent its energy polishing the tokenization pipeline while shipping broken plumbing around it. Consider the greatest hits, all of which were live until the recent restructuring:

- A phantom import (`from Projects.Tokenizer.utils import ensure_dir`) that made the extras package unimportable from anywhere except one exact parent directory.
- A log file written into whatever directory you happened to run from.
- Two CLI flags (`-c` / `-d`) parsed by argparse and then silently discarded.
- A test module whose documented entry point (`python -m testing`) did literally nothing, because nobody had added a `__main__` guard.
- An `is_fallback` property that — verified live, five minutes before this sentence was written — reports `True` even when the real BERT tokenizer loaded successfully. In practice it has never once returned a correct answer.

**Where things stand now:** proper importable package (`anatok`), two console commands (`anatok`, `anatok-gui`), 55 passing pytest tests including a full write-read-verify Arrow roundtrip, platform-correct log locations, and a `results/` directory that receives `.pyarrow` + `.json` + `.md` artifacts per run.

**Verdict:** The bones are worth keeping. Several organs need transplanting.

---

### 1. WHAT THIS PROJECT ACTUALLY IS (AND IS NOT)

The repo name says "Tokenizer". That is misleading advertising.

Actual tokenization is delegated almost entirely to HuggingFace's `tokenizers` library (Rust underneath), defaulting to `bert-base-uncased`, with an offline fallback that trains a tiny throwaway BPE on three hardcoded sentences when the hub is unreachable.

**What is actually yours, hand-written, in this codebase:**

- A **constant-memory streaming harness:** `iter_file_blocks` (4 MB) → `TextChunkStreamer` (~256 KB pieces, splits at newlines/spaces) → `encode_batch_full` in groups of ~1 MB of text.
- An **orchestration layer** (`managers.py`): `TokenizationManager` pipeline + `CompressionManager` with streaming zlib and output-size-based retention.
- **Thread workers:** plain `threading.Thread` variants (`workers.py`) for headless use; `QThread` wrappers (`gui.py`) with signals for the GUI.
- **Legacy compression algorithms:** zlib wrapper, hand-written LZ77 ("v2" wire format), Huffman with incremental bit packing, byte-pair RLE.
- **Token-stream dedup utilities** (`dedup.py`).
- **Real-RSS memory tracking** (`memory_manager.py`): psutil → Win32 GetProcessMemoryInfo → Unix getrusage fallback chain, plus a background sampler thread.
- A **PyQt6 GUI** that keeps every heavy operation off the UI thread.
- The **results exporter** (`results.py`): complete token stream as Arrow IPC, plus JSON and Markdown sidecars.

If you came looking for a from-scratch BPE implementation to study, lower your expectations accordingly. What is here instead is a decent systems-plumbing education: streaming, chunk-boundary management, cooperative cancellation, singleton lifecycle control, and cross-platform memory introspection. That is a perfectly respectable thing for a project to be — as long as you know that is what it is.

---

### 2. HOW TO READ THIS CODEBASE IN TWENTY MINUTES

Read in this order. Each file assumes the previous one.

1. **`utils.py`** — Block reader (`iter_file_blocks`), hashing, byte formatting, path validation. Tiny. The block reader is the heartbeat of the whole design.
2. **`tokenizer.py`** — `HFTokenizerWrapper` (thread-safe, cached per model), `TextChunkStreamer` (the chunk-boundary machinery), `stream_encode_file` / `iter_word_group_tokens` / `count_file_tokens_hf` (the three streaming entry points), plus classic fallbacks.
3. **`results.py`** — `TokenArrowWriter` (streaming Arrow IPC export), JSON/Markdown sidecars, results-dir resolution.
4. **`managers.py`** — `TokenizationManager.process_path` — **THE pipeline.** Everything user-facing funnels through here. `CompressionManager`: streaming zlib + legacy size-guarded paths + `decompress_verify`.
5. **`workers.py`** — `threading.Thread` wrappers with callback events and stop-events. GUI-agnostic on purpose.
6. **`folder_workers.py`** — `FolderWorker`: sequential per-file batch runs with a 512 MB skip cap and abort support.
7. **`compressors.py` / `decompressors.py`** — zlib default; corrected LZ77 v2; Huffman dict payload {compressed_bytes, code_table, padding}; RLE pairs (count, byte). Decoders are defensive: corrupt streams stop instead of crashing.
8. **`memory_manager.py`** — Real RSS sampling + logical allocation bookkeeping. The docstring honestly admits the old version was a fake 100 MB "pool" that stored nothing. Respect for that honesty.
9. **`gui.py`** — `QMainWindow` + `QThread` workers, stale-completion guards, byte-level progress, `deleteLater` discipline.
10. **`error_warehouse.py`** — Central logging. File handler goes to the platform app-data dir now, not your CWD.
11. **everything else** — viewers (trivial state holders), interactions (REPL), ctypes_wrapper (mostly scaffolding), extras (base64/session helpers), testing/smoke_test (legacy suites).

**Data flow of the main pipeline (`TokenizationManager.process_path`):**

```
file on disk
    ↓ iter_file_blocks (4 MB blocks) → progress_cb(bytes)
    ↓
TextChunkStreamer.feed (decodes to text, splits at ~256 KB,
    preferring newline > space > hard cut)
    ↓ pending buffer (~1 MB) → encode_batch_full → Encoding objects
        |→ encoding_cb → TokenArrowWriter (flush every 8192 rows to disk)
        running stats: total count, unique set (capped at 2M),
        preview list (first 500 token strings ONLY)
    ↓
result dict: counts + ratios + preview + memory snapshot
    +-- optional: zlib streaming compress (+ roundtrip verify)
    +-- exports: _.pyarrow / .json / .md in results/
```

**Notice what never happens:** a full list of tokens is never held in RAM. That is the entire personality of this codebase, and it is a good personality to have.

---

### 3. WHAT IS GENUINELY GOOD (CREDIT WHERE DUE)

Being brutal cuts both ways. These things are legitimately well done:

- **The constant-memory discipline is real, not marketing.** Verified end-to-end: a run over a repeated-sentence file peaked around 45-55 MB RSS regardless of input size, because nothing downstream of the block reader ever sees more than a few MB.
- **Output-size-based compression retention** (`managers.CompressionManager`). The compressed payload is kept only if the OUTPUT fits under 32 MB — not the input. A highly compressible 10 GB log therefore stays fully verifiable while an incompressible 40 MB blob gets dropped gracefully with sizes still reported. That is a genuinely thoughtful decision most people get wrong in the naive direction.
- **LZ77 v2 wire format is correct**, including overlapping-match copy semantics (`decompressors.lz77_decompress` copies byte-by-byte as the output grows, which is exactly what LZ77 requires and what naive implementations get wrong). The format also fixed the old bug where literal bytes had their high bit destroyed by the flag byte.
- **Defensive decoders everywhere.** Corrupt LZ77/RLE/Huffman input makes the decoder stop cleanly rather than throw or loop forever. `decompress_verify` hashes while streaming and reports `verified=None` ("cannot know") when the payload was dropped, instead of guessing. Distinguishing "no" from "unknown" is rarer than it should be in the wild.
- **Idempotent logger setup** (`error_warehouse` checks `logger.handlers` before adding), so constructing `ErrorWarehouse` ten times does not produce ten duplicate log lines. Small thing. Everyone forgets it once.
- **The GUI's stale-worker guard** (`_is_current_worker`) prevents a cancelled operation's late completion signal from clobbering a newer operation's results, and `_dispose_worker` handles the `RuntimeError` race when Qt has already destroyed the C++ object. Someone got burned by that race before, and then wrote the fix down. Good.
- **Cooperative cancellation is threaded through every layer consistently:** `should_abort` callables flow from GUI button → QThread wrapper → manager → stream generator → between-chunk checks. No layer fakes it.
- **The pytest suite (55 tests)** covers real integration, including: arrow write-read-verify with row-count equality against the live run's token count, null-token_id handling for the custom path, CLI subprocess tests, and folder summary export. This is not decoration; it caught four real bugs during the restructuring.

---

### 4. THE FLAW CATALOG

**Severity scale:**
- **CRITICAL** — wrong results, data loss, or crash under normal use
- **HIGH** — breaks or misleads under realistic (not exotic) conditions
- **MEDIUM** — debt that WILL bite someone eventually
- **LOW** — hygiene, cosmetics, papercuts

#### 4.1 Bugs and landmines

**BUG #1 — `is_fallback` has never returned a correct answer [HIGH]**

`tokenizer.py`, `HFTokenizerWrapper.is_fallback`:

```python
return self._tokenizer is not None and self.model_name not in str(
    getattr(self._tokenizer, "model", "")
)
```

The theory: if the pretrained model loaded, its repr will contain the model name. The reality: `Tokenizer.from_pretrained("bert-base-uncased")` yields a WordPiece object whose repr looks like:

```
WordPiece(unk_token="[UNK]", continuing_subword_prefix="##", max_input_chars_per_word=100, ...)
```

No hub id anywhere. So the name is never "in" it and the property returns `True` unconditionally — for a successful load AND for the fallback alike.

Verified live during this audit:
```python
>>> w = HFTokenizerWrapper("bert-base-uncased")  # real BERT loaded
>>> w.is_fallback
True
```

**Impact today:** nothing consumes the property, so this is a lying gauge on a dashboard nobody reads. The moment you build a UI indicator or a metrics export on top of it, your data is quietly wrong.

**Fix:** record which branch of `_load_tokenizer` succeeded in a boolean set at load time. Two lines.

---

**BUG #2 — UTF-8 corruption at every 4 MB block boundary [HIGH]**

`TextChunkStreamer.feed` does:
```python
buf = self._buf + block.decode("utf-8", errors="replace")
```

A multi-byte UTF-8 sequence that straddles a 4 MB block boundary gets split, and each orphaned byte decodes to U+FFFD REPLACEMENT CHARACTER. Every single block boundary in non-ASCII text silently mangles up to three bytes.

For pure token counting this is nearly invisible; for any future feature that cares about text fidelity (offsets, detokenization, dedup of exact strings) it is data corruption with no error raised.

**Fix:** use `codecs.getincrementaldecoder("utf-8")(errors="replace")` and feed it bytes; it holds incomplete tail bytes across calls by design. Roughly a four-line change. This is the bug most worth fixing first.

---

**BUG #3 — "custom" tokenization still loads the HF stack [MEDIUM]**

`TokenizationWorker(use_hf=False)` correctly skips caching an HF wrapper on itself... and then constructs `get_tokenization_manager()`, whose `__init__` ALWAYS calls `get_hf_tokenizer(hf_model)`. Net effect: choosing the fast local path still pays for constructing the full BERT wrapper (online: vocab download attempt + cache hit; offline: trains the mini fallback BPE). The custom path works, but it is not actually independent of the HF stack.

---

**BUG #4 — fabricated `vocab_size` on the custom path [LOW]**

`managers.process_path`: `vocab_size = max(unique, 100)`. There is no vocabulary in the word-group method; this invents a number for display. Harmless-looking, but it teaches consumers to distrust every other number in the result dict. Report null instead of inventing.

---

**BUG #5 — "compression_ratio" means dedup ratio [MEDIUM]**

In `process_path`'s result dict, `compression_ratio = duplicates_removed / total_tokens`. It has nothing to do with compression; the zlib numbers live under `result["compressed"]["ratio"]`. Two different ratios with overlapping names in one payload is how dashboards end up plotting the wrong thing with total confidence. Rename one of them (`dedup_ratio` is right there).

---

**BUG #6 — Arrow files are unreadable if the process dies mid-run [MEDIUM]**

Arrow IPC needs its footer written by `close()` to be valid. Crash, Ctrl-C at the wrong moment, or power loss leaves a `.pyarrow` that `feather` refuses to open, sitting next to perfectly readable `.json`/`.md` sidecars that claim the run completed.

**Fix pattern:** write to `.pyarrow.parttmp` then atomic `os.replace()` on successful close.

---

**BUG #7 — `results/` grows forever [MEDIUM]**

Every run writes three artifacts; nothing prunes. Run `anatok` nightly over a corpus for a year and `results/` becomes its own big-data problem. Needs a retention policy (keep last N / age-based) or at least a documented manual cleanup story.

---

**BUG #8 — folder mode ignored `--fast` and `--model` [fixed]**

Caught during this audit: `process_folder_cli` built `FolderWorker` without forwarding the flags, so directory runs always used `bert-base-uncased` via HF regardless of CLI arguments. Fixed in `folder_workers.py` + `main.py` during this audit (constructor now takes `hf_model`/`use_hf`).

Documented here because it is instructive: two entry points (file vs folder) drifting apart is exactly how flag-handling bugs are born. One pipeline, many fronts — keep the frontends dumb.

---

**BUG #9 — second `ErrorWarehouse` ignores its `log_file` argument [LOW]**

`_setup_logger` attaches handlers once per process (idempotent by design), but that also means a second `ErrorWarehouse(log_file="other.log")` silently logs to the FIRST instance's file while reporting `self.log_file` as `other.log`. `get_stats()` would lie about where logs go. Minor, but it is the kind of thing that costs someone an hour at 2 AM.

---

#### 4.2 Design decisions that will bite later

**D-1. Print-based logging all over `workers.py` and `ctypes_wrapper.py`** (`print(f"[Worker] {level}...")`) while `error_warehouse` exists. Two logging systems in one codebase means neither is trustworthy. Pick one; it should be `error_warehouse` (or better, plain `logging` with the warehouse as a handler).

**D-2. `WARNING+` console spam by design.** The warehouse prints every warning to stdout, so library consumers piping output get their stdout polluted by HF-hub rate-limit warnings and friends. Console handler should be stderr, or opt-in.

**D-3. The tokenizer lock serializes everything.** `HFTokenizerWrapper` guards every encode with `self._lock`. The Rust engine releases the GIL and parallelizes beautifully — and then this lock funnels all threads through one door. Correct for safety, terrible for throughput if you ever run concurrent workers. Fine for now; document it as a known ceiling.

**D-4. `memory_manager` bookkeeping is O(n).** `release()` linear-scans the chunk list; `get_stats()` sums under lock. With thousands of logical allocations this turns into real time spent inside a mutex. Currently bounded by polite usage; not bounded by code.

**D-5. Background sampler thread wakes 4x/second forever after `initialize()`.** Cheap, but it means every process that touches a manager carries a permanent timer. There is an `atexit` shutdown hook, which is good. Just know it is there when you wonder why a script will not exit instantly under `-X importtime` profiling.

**D-6. `ctypes_wrapper` is mostly scaffolding wearing a lab coat.** `set_tokenizer_` callback accepts raw integer "pointers" from Python callers — an API shape that is a memory-corruption invitation the day anyone points it at a real library. `memmove`/`set_string_attribute` wrappers add nothing over raw ctypes. Either give it a real native companion library or shrink it to `load_library` + `create_buffer`.

**D-7. Legacy compression paths double-buffer.** `compress_path_streaming` for lz77/simple/huffman reads the WHOLE file into memory (guarded at 256 MB), then builds the compressed bytearray next to it. Worst case ~2× input in RAM, plus `simple_compress` can DOUBLE size again for non-repetitive data. The guards prevent disasters, not inefficiency.

**D-8. LZ77's hash-chain dict is unbounded in key count.** Each distinct 3-byte sequence gets a positions list (capped at 128 entries each). Feed it high-entropy binary and you can approach ~16M distinct keys — tens of MB of dict overhead to compress data that will not compress anyway.

**D-9. Huffman decode walks bit-by-bit.** `huffman_decompress` consumes one bit at a time with a dict lookup per bit. Pedagogically lovely, production-wise slow (table-driven decoding is the standard fix). Fine for a legacy opt-in path; do not benchmark it against zlib.

**D-10. `utils.get_file_info` reads entire files into RAM to hash them**, and `interactions.py` calls it on user-supplied paths. Type a 20 GB file into interactive mode and watch RSS become a problem statement. Hash should stream through `iter_file_blocks` like everything else in this codebase.

---

#### 4.3 Performance reality check

Measured during this audit (Python 3.10, Windows, bert-base-uncased via hub):

- **3.8 KB demo text, HF path:** ~3.8K tokens, ~45 MB RSS peak, well under a second end-to-end including export of three artifacts.
- **Same file, custom path:** 12 word-group tokens (the file is one repeated sentence — the grouper is doing its job), roundtrip zlib verify passed.
- **Arrow export:** 3,802 rows → 295 KB `.pyarrow`. Strings dominate; ids-only export would be far smaller if you ever need leaner files.

Where the real time goes on big inputs: `encode_batch_full` (Rust, fast), then `TextChunkStreamer`'s Python string surgery, then unique-set bookkeeping. The architecture will comfortably eat multi-GB text files in bounded RAM; throughput is respectable but this is not an optimized batch tokenizer and does not pretend to be.

---

#### 4.4 Packaging / repo hygiene

**Fixed during restructuring** (listed because they explain why the layout looks like this now):

- Flat modules named `utils`/`workers`/`managers`/`testing` — collision bait with any other package on PyPI or in site-packages. Now namespaced under `src/anatok` with relative imports throughout (~30 import edges rewritten).
- `extras`' phantom absolute import (`Projects.Tokenizer.utils`). Now `..utils`.
- Log file in CWD → platform app-data dir (`ANATOK_LOG_DIR` overrides).
- Dead `-c`/`-d` flags → wired to `process_path(compress/decompress)` with `"-d requires -c"` validation.
- `testing.py` had no `__main__` guard; README documented running it anyway, and also referenced `test_gui_offscreen.py` which never existed, and cited filenames with wrong casing (`Dedup.py` vs `dedup.py`). The old README was confidently wrong about its own project — it has been removed at your request; this document inherits its truthful duties.
- `LISENCE` misspelled, stray committed `tokenizer_errors.log`, `__pycache__` tracked, no `.gitignore`. All resolved.

**Still open:**

- **No CI config** (`.github/workflows` absent). For a GitHub-first repo this is the single highest-value missing artifact: `pytest` + `ruff` across 3.10-3.12 on ubuntu/windows would lock in everything this audit just fixed.
- No `mypy`/`ruff` gate wired despite `ruff` config living in `pyproject.toml`.
- Typing is partial and inconsistent (mix of `typing.Optional` legacy and PEP 604 in `gui.py` — note: `self._worker: QThread | None` is evaluated at runtime, which is exactly why `requires-python` is >=3.10).
- Version is hand-synced between `pyproject.toml` and `__init__.py`. One source of truth wanted.

---

#### 4.5 Test suite honesty report

Two generations coexist:

- **`tests/` (pytest, 55 tests):** current source of truth. Integration-flavored, includes CLI subprocess runs and Arrow roundtrips. Weaknesses: no coverage measurement wired, no property-based tests (`hypothesis` would earn its keep on compressors/decompressors roundtrips), GUI untested by design (headless Qt testing was referenced by the old README but never existed).
- **`anatok/testing.py` + `smoke_test.py` (legacy assert-count style):** kept for parity. Honest flaw of that style: failures are counted prints without tracebacks, so when something fails you get "FAIL: something" and a prayer. It survives as a smoke ritual (`python -m anatok.smoke_test`), not as the test system.

**Not covered anywhere yet:** abort-mid-stream behavior (`should_abort` firing between chunks), non-UTF-8/binary inputs to the streaming tokenizer (where BUG #2 lives), concurrent workers hammering one manager, Windows long-path edge cases.

---

#### 4.6 Security and robustness notes

- **Good news first:** no `eval`/`exec`/`pickle` of external data, no SQL, no network writes, JSON-only persistence. Attack surface is small and honest.
- `ctypes` CDLL can load arbitrary native libraries — that is its entire job, but treat `library_path` as privileged input forever.
- HuggingFace hub downloads are a standard supply-chain trust decision. Pin/audit like any dependency. Offline mode degrades gracefully to the mini-BPE fallback, which is nice for reproducibility and slightly sad for accuracy — know which mode you ran.
- `results/*.json` embeds absolute filesystem paths (source file location). If you ever share result artifacts, you are sharing directory structures. Worth a sanitize option before results leave your machine.

---

### 5. MODULE SCORECARD

Graded against the project's own stated goal: memory-safe tokenization tooling.

| Module | Grade | Notes |
|--------|-------|-------|
| `utils.py` | **A-** | Tiny, correct, does its job. Loses points for `get_file_info`'s whole-file hash (D-10). |
| `tokenizer.py` | **B** | The streaming heart. Genuinely good chunking design; loses for `is_fallback` (BUG #1) and UTF-8 boundaries (BUG #2). |
| `results.py` | **B+** | New code, clean seams, streaming export that respects the house memory religion. Needs atomic finalize + retention policy. |
| `managers.py` | **B** | The pipeline works end to end with cancellation, previews, and honest size reporting. Loses for the ratio naming mess (BUG #5) and fabricated vocab size (BUG #4). |
| `workers.py` | **C+** | Functional, cancellable, but `print()`-logging and the hidden HF dependency on the "fast" path. |
| `folder_workers.py` | **B-** | Simple, correct, sequential-only; flags plumbing just fixed. Parallelism is an open opportunity. |
| `compressors.py` | **B** | zlib path solid; legacy paths are honest about being legacy. LZ77 v2 format is actually correct. |
| `decompressors.py` | **A-** | Defensive, symmetric, stops on corruption. The best-behaved module in the repo. |
| `dedup.py` | **B** | Small, clear, does what it claims. |
| `memory_manager.py` | **B-** | Real RSS tracking with a graceful fallback chain is genuinely nice; O(n) bookkeeping and a forever sampler thread are real costs. |
| `ctypes_wrapper.py` | **D** | Scaffolding in production clothing. Unsafe API shape (raw int pointers) waiting for a victim. |
| `viewers.py` | **B** | Trivial, and honestly so. |
| `interactions.py` | **C** | Works; inherits D-10's RAM spike on big files; no integration with managers' nicer reports. |
| `gui.py` | **B+** | The threading discipline here (stale guards, deleteLater, byte-progress, cancel) is better than most commercial Qt code I have read. |
| `error_warehouse.py` | **B** | Idempotent, platform-aware now; dual-instance gotcha remains (BUG #9), stdout spam by design. |
| `testing.py` / `smoke_test.py` | **C** | Kept as ritual; superseded by `tests/`. |
| `extras/` | **C+** | Session save/load and base64 helpers are fine; `compression.py` duplicates `compressors.zlib` for no reason anyone can name. |

---

### 6. HISTORY THIS REPO REMEMBERS (context you will not get from names)

- `memory_manager`'s docstring confesses the old version reserved 100 MB+ "pool" memory and stored nothing — a fake allocator from an earlier design phase. The current module is the apology.
- LZ77's docstring documents its own past bug (flag byte destroying the high bit of literals). The v2 format exists because of it. This is what healthy archaeology looks like inside code comments.
- `COMPRESSED_RETAIN_LIMIT=32MB` and `LEGACY_INPUT_LIMIT=256MB` encode hard-won lessons about GUI memory death; `UNIQUE_TOKEN_CAP=2M` with a `"__capped__"` sentinel string is a pragmatic hack whose smell was acknowledged by naming the flag `unique_is_lower_bound` in results.
- The word "fr" in the old `tokenizersfr.py` filename appears to be an accident that fossilized into identity. It is now `tokenizer.py`; the old name survives only in this paragraph.
- `check_project.py`'s "risky references" list (`_manager`, `compress_file_from_bytes`, `get_offset_mapping`) is a scar table from bugs that once shipped. Keep it; static scans that remember history are cheap insurance.
- The demo artifacts currently sitting in `results/` were produced during this audit's verification runs. They double as format documentation — open one of each alongside section 2.

---

### 7. PRIORITIZED FIX ROADMAP

**P0 — correctness (do these before anything else)** 

1. Fix UTF-8 block-boundary decoding with an incremental decoder. ~4 lines.
2. Replace or delete `is_fallback` (record the load branch explicitly). ~3 lines.
3. Rename `result["compression_ratio"]` → `dedup_ratio` (keep old key one release for compatibility).

**P1 — robustness** 

4. Atomic Arrow finalize: write `.parttmp`, `os.replace()` after close().
5. Results retention policy (env-configurable keep-last-N, default maybe 200 runs) or document manual cleanup.
6. Stream the SHA-256 in `utils.get_file_info` / `interactions`.
7. Skip HF construction when `use_hf=False` (make the fast path actually fast).

**P2 — performance** 

8. Cap LZ77's trigram dict size for high-entropy inputs.
9. Table-driven Huffman decode if the legacy path matters to you.
10. Consider per-thread tokenizer instances instead of the global lock when concurrent throughput becomes a requirement.

**P3 — hygiene / growth** 

11. GitHub Actions: `pytest` + `ruff` on 3.10/3.11/3.12 × ubuntu/windows. Highest value-per-hour item on this list for a GitHub-first repo.
12. Unify logging on `error_warehouse` (kill `print()` logging); console handler to stderr.
13. Wire `mypy` (or drop the pretense) and add `hypothesis` roundtrip tests for all four codecs.
14. Single-source the version number.

**Verdict on salvage vs rewrite: SALVAGE, clearly.**  The streaming core, the manager seams, and the worker discipline are the hard parts and they are done right. Everything on this roadmap is plumbing-level effort measured in hours, not weeks.

---

### 8. GLOSSARY OF THINGS WITH WEIRD NAMES

| Term | Meaning |
|------|---------|
| `TextChunkStreamer` | Decodes bytes → text pieces at ~256 KB, splitting at newline > space > hard cut. Holds partial chunks across feeds. |
| `encoding_cb`/`group_cb` | Hooks letting managers mirror the live token stream into `TokenArrowWriter` without re-tokenizing. |
| `COMPRESSED_RETAIN_LIMIT` | Max compressed OUTPUT kept in RAM (32 MB). Output-based, deliberately. |
| `__capped__` | Sentinel stuffed into the unique-set when the 2M cap trips; paired with `unique_is_lower_bound=True`. |
| `decompress_verify` | Streaming decompress + SHA-256 compare; `verified` can be `True`/`False`/`None` (None = payload wasn't retained). |
| `stale-worker guard` | `gui.py`'s defense against cancelled operations delivering late completion signals. |
| `risky references` | `check_project.py`'s scar table of historical bug names. |
| `.pyarrow` | Arrow IPC (feather v2) file holding the complete run: columns `doc`, `seq`, `token_id`, `token` + provenance JSON in schema metadata. Readable via `pyarrow.feather`. |

---

## Appendix A: LICENSE

```
MIT License

Copyright (c) 2026 anatok contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Appendix B: pyproject.toml (abridged)

```toml
[build-system]
requires = ["hatchling>=1.24"]
build-backend = "hatchling.build"

[project]
name = "anatok"
version = "0.1.0"
description = "Fast, memory-bounded AI text tokenization toolkit with compression, deduplication and Arrow (.pyarrow) results export."
requires-python = ">=3.10"
license = { file = "LICENSE" }
authors = [ { name = "anatok contributors" } ]
keywords = ["tokenizer", "tokenization", "bpe", "huggingface", "nlp", "compression", "lz77", "huffman", "arrow", "streaming"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Text Processing :: Linguistic",
    "Topic :: System :: Archiving :: Compression",
]
dependencies = [
    "tokenizers>=0.15,<1.0",
    "psutil>=5.9",
    "pyarrow>=14.0",
]

[project.optional-dependencies]
gui = [ "PyQt6>=6.6" ]
dev = [ "pytest>=8.0", "pytest-cov>=4.1", "ruff>=0.4", "build>=1.2" ]

[project.scripts]
anatok = "anatok.main:main"
anatok-gui = "anatok.main:gui_main"

[tool.hatch.build.targets.wheel]
packages = ["src/anatok"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP"]
ignore = ["E501"]
```

---

## Appendix C: requirements.txt

# Runtime dependencies (see pyproject.toml for authoritative, version-bounded pins)
tokenizers
psutil
pyarrow
# Optional GUI extra: pip install "anatok[gui]"
PyQt6


---

## Appendix D: check_project.py (excerpt)

A full project health check script that verifies syntax and scans for undefined names across all source files. It compiles every Python file in `src/anatok/`, `src/anatok/extras/`, and `tests/`, then performs AST-based undefined-name analysis. The script includes a "risky references" list — a scar table from bugs that once shipped.

---

*End of document.*
