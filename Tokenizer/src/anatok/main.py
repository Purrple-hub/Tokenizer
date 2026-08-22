"""Main entry point for anatok.

A fast, AI-optimized tokenization tool with GUI, compression, deduplication
and full Arrow (.pyarrow) results export.

Usage:
    anatok                      # show help
    anatok path/to/file         # tokenize a file (CLI report + results/)
    anatok path/to/folder       # tokenize every file in a folder
    anatok file -c              # also compress (zlib) after tokenizing
    anatok file -c -d           # ... and verify the decompress roundtrip
    anatok --interactive        # interactive CLI mode
    anatok --gui [path]         # launch the PyQt6 GUI
"""
import argparse
import os
import sys

from .error_warehouse import ErrorWarehouse

def run_gui(path=None):
    """Launch the PyQt6 GUI application, optionally pre-loading a path."""
    try:
        from .gui import launch_gui
    except ImportError as e:
        print(
            "The GUI requires PyQt6, which is not installed.\n"
            "Install it with:  pip install 'anatok[gui]'\n"
            f"(import error: {e})"
        )
        return 2
    return launch_gui(initial_path=path)

def gui_main():
    """Console-script entry point for ``anatok-gui``."""
    sys.exit(run_gui())

def process_file_cli(path, error_handler, args):
    """Tokenize a single file and print a CLI report."""
    from .managers import get_tokenization_manager
    from .utils import format_bytes

    manager = get_tokenization_manager(
        args.model or "bert-base-uncased")
    result = manager.process_path(
        path,
        compress=args.compress,
        decompress=args.decompress,
        use_hf=not args.fast,
        export=not args.no_export,
        results_dir=args.results_dir,
    )

    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)

    print("=" * 50)
    print("Tokenization Report")
    print("=" * 50)
    print(f"File:              {path}")
    print(f"Method:            {result.get('method', 'unknown')}")
    print(f"Tokens:            {result.get('original_tokens', 0)}")
    print(f"Unique tokens:     {result.get('optimized_tokens', 0)}")
    print(f"Duplicates removed:{result.get('duplicates_removed', 0)}")
    comp = result.get("compressed")
    if comp:
        print(f"Compressed size:   {comp.get('compressed_size', 0)} bytes "
              f"({(comp.get('ratio') or 0):.2%})")
        if "roundtrip_ok" in result:
            ok = result.get("roundtrip_ok")
            status = ("VERIFIED" if ok
                      else "FAILED" if ok is False else "not retained")
            print(f"Roundtrip:         {status}")
    mem = result.get("memory", {})
    if mem:
        print(f"Memory used:       {mem.get('pool_usage_mb', 0):.2f} MB")
        print(f"Peak memory:       {mem.get('peak_usage_mb', 0):.2f} MB")

    exports = result.get("exports") or {}
    written = {k: v for k, v in exports.items() if v}
    if written:
        print()
        print("Results saved:")
        for kind, p in sorted(written.items()):
            print(f"  [{kind}] {p} ({format_bytes(os.path.getsize(p))})")

def process_folder_cli(path, error_handler, args):
    """Tokenize all files in a folder and print a CLI report."""
    from .folder_workers import FolderWorker
    from .results import export_folder_summary

    worker = FolderWorker(path, operation="tokenize",
                          max_file_size=args.max_file_size,
                          hf_model=args.model or "bert-base-uncased",
                          use_hf=not args.fast)
    summary = worker.process()

    total_tokens = sum(r.get("original_tokens", 0)
                       for r in summary.get("results", [])
                       if "error" not in r and not r.get("skipped"))
    for r in summary.get("results", []):
        name = os.path.basename(str(r.get("file", "?")))
        if "error" in r:
            error_handler.log("error", f"{r['file']}: {r['error']}")
            print(f"  {name}: ERROR ({r['error']})")
        elif r.get("skipped"):
            print(f"  {name}: skipped ({r.get('reason', '')})")
        else:
            print(f"  {name}: {r.get('original_tokens', 0)} tokens")

    print("=" * 50)
    print(f"Files processed:   {summary.get('processed', 0)}")
    print(f"Total tokens:      {total_tokens}")
    print(f"Errors:            {summary.get('errors', 0)}")

    exports = {}
    if not args.no_export:
        exports = export_folder_summary(
            summary, results_dir=args.results_dir,
            stem=os.path.basename(path.rstrip("\\/")) + "_folder")
        written = {k: v for k, v in exports.items() if v}
        if written:
            print()
            print("Results saved:")
            for kind, p in sorted(written.items()):
                print(f"  [{kind}] {p}")

def build_parser() -> argparse.ArgumentParser:
    """Construct the anatok argument parser."""
    parser = argparse.ArgumentParser(
        prog="anatok",
        description="Fast AI-optimized tokenization tool with compression, "
                    "deduplication and Arrow (.pyarrow) results export",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="File or folder path to tokenize",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the PyQt6 graphical interface",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--compress", "-c",
        action="store_true",
        help="Compress the input (zlib) after tokenizing",
    )
    parser.add_argument(
        "--decompress", "-d",
        action="store_true",
        help="Verify the decompression roundtrip after compressing "
             "(requires -c)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use the built-in word-group tokenizer instead of HuggingFace",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="HuggingFace tokenizer model (default: bert-base-uncased)",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory for exported results "
             "(default: $ANATOK_RESULTS_DIR or ./results)",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Do not write .pyarrow/.json/.md result files",
    )
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=512 * 1024 * 1024,
        help="Skip files larger than this many bytes in folder mode",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )
    return parser

def _get_version() -> str:
    from . import __version__
    return __version__

def main():
    """Entry point for the anatok application."""
    error_handler = ErrorWarehouse()
    error_handler.log("info", "anatok application starting")

    parser = build_parser()
    args = parser.parse_args()

    if args.decompress and not args.compress:
        parser.error("-d/--decompress requires -c/--compress")

    if args.gui:
        sys.exit(run_gui(args.path))

    if args.interactive:
        from .interactions import interactive_mode
        interactive_mode()
        return

    if args.path:
        path = args.path
        if not os.path.exists(path):
            error_handler.log("error", f"Path not found: {path}")
            print(f"Error: Path not found: {path}")
            sys.exit(1)

        if os.path.isdir(path):
            process_folder_cli(path, error_handler, args)
        else:
            process_file_cli(path, error_handler, args)
        return

    print("No arguments given. Use --gui for graphical mode, --interactive")
    print("for interactive CLI, or pass a file/folder path.")
    print()
    parser.print_help()

if __name__ == "__main__":
    main()
