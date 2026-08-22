"""Full project health check: syntax, imports, warnings, undefined names.

Usage:  python check_project.py   (from the repository root)
"""
import ast
import glob
import io
import os
import py_compile
import shutil
import sys
import tempfile
import warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src", "anatok")

files = sorted(glob.glob(os.path.join(SRC, "*.py"))) + \
    sorted(glob.glob(os.path.join(SRC, "extras", "*.py"))) + \
    sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py")))

_tmpdir = tempfile.mkdtemp(prefix="anatok_check_")

print("=" * 60)
print("PHASE 1: Syntax compile check")
print("=" * 60)
syntax_errors = []
for f in files:
    try:
        cfile = os.path.join(_tmpdir, os.path.basename(f) + ".pyc")
        py_compile.compile(f, doraise=True, cfile=cfile)
        print(f"  OK      {os.path.relpath(f, ROOT)}")
    except py_compile.PyCompileError as e:
        syntax_errors.append(f)
        print(f"  SYNTAX  {os.path.relpath(f, ROOT)}: {e}")

print()
print("=" * 60)
print("PHASE 2: AST undefined-name scan")
print("=" * 60)

def get_defined_names(tree):
    names = set(dir(__builtins__)) if isinstance(__builtins__, dict) \
        else set(dir(__builtins__))
    stdlib = {"os", "sys", "ctypes", "gc", "time", "json", "glob", "struct",
              "queue", "threading", "argparse", "hashlib", "heapq", "logging",
              "tempfile", "base64", "zlib", "ast", "io", "importlib",
              "warnings", "contextlib"}
    names |= stdlib
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == "*":
                    continue
                names.add(a.asname or a.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.For,)) and isinstance(node.target,
                                                         ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.comprehension) and \
                isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.withitem) and node.optional_vars and \
                isinstance(node.optional_vars, ast.Name):
            names.add(node.optional_vars.id)
    return names

def find_used_names(tree):
    used = set()
    attrs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            attrs.add(node.attr)
    return used, attrs

for f in files:
    if f.endswith("check_project.py"):
        continue
    try:
        tree = ast.parse(open(f, encoding="utf-8").read())
        defined = get_defined_names(tree)
        used, attrs = find_used_names(tree)
        missing = {n for n in used if n not in defined}
        missing -= {"self", "cls", "__name__", "__file__", "__doc__",
                    "__main__", "super"}
        if missing:
            print(f"  WARN    {os.path.relpath(f, ROOT)}: "
                  f"possibly-undefined names: {sorted(missing)}")
        else:
            print(f"  OK      {os.path.relpath(f, ROOT)}")
    except SyntaxError as e:
        print(f"  SYNTAX  {os.path.relpath(f, ROOT)}: {e}")

print()
print("=" * 60)
print("PHASE 3: Import test (all modules, all warnings shown)")
print("=" * 60)

sys.path.insert(0, os.path.join(ROOT, "src"))

MODULES = ["anatok.utils", "anatok.error_warehouse", "anatok.memory_manager",
           "anatok.ctypes_wrapper", "anatok.compressors",
           "anatok.decompressors", "anatok.dedup", "anatok.viewers",
           "anatok.tokenizer", "anatok.managers", "anatok.workers",
           "anatok.folder_workers", "anatok.results", "anatok.interactions",
           "anatok.testing", "anatok.main", "anatok.extras",
           "anatok.extras.compression"]
MODULES.append("anatok.gui")

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    results = {}
    for m in MODULES:
        buf_err = io.StringIO()
        buf_out = io.StringIO()
        mod_ok = True
        err_txt = ""
        old = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = buf_out, buf_err
        try:
            __import__(m)
        except Exception as e:
            mod_ok = False
            err_txt = f"{type(e).__name__}: {e}"
        finally:
            sys.stdout, sys.stderr = old
        results[m] = (mod_ok, err_txt, buf_err.getvalue(),
                      buf_out.getvalue())

for m, (ok, err, errbuf, outbuf) in results.items():
    status = "OK     " if ok else "FAIL   "
    print(f"  {status} import {m}" + (f"  -> {err}" if not ok else ""))
    if ok and errbuf.strip():
        for line in errbuf.strip().splitlines():
            print(f"           stderr: {line}")
    if ok and outbuf.strip():
        for line in outbuf.strip().splitlines()[:5]:
            print(f"           stdout: {line}")

for w in caught:
    print(f"  WARNING [{w.category.__name__}] {w.message} "
          f"({w.filename}:{w.lineno})")

print()
print("=" * 60)
print("PHASE 4: Cross-module attribute checks (calls into other modules)")
print("=" * 60)
risky = [
    ("managers.py", "compress_file_from_bytes"),
    ("workers.py", "_manager"),
    ("workers.py", "compress_file_from_bytes"),
    ("gui.py", "compress_file_from_bytes"),
    ("tokenizer.py", "get_offset_mapping"),
]
for fname, needle in risky:
    path = os.path.join(SRC, fname)
    if os.path.exists(path):
        content = open(path, encoding="utf-8").read()
        count = content.count(needle)
        flag = "FOUND " if count else "clean "
        print(f"  {flag} {fname}: '{needle}' x{count}")

print()
shutil.rmtree(_tmpdir, ignore_errors=True)
print("DONE")
