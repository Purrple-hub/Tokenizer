import json
import subprocess
import sys

from anatok.folder_workers import FolderWorker

def test_cli_tokenizes_and_exports(sample_file, tmp_path):
    results_dir = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, "-m", "anatok", sample_file,
         "--results-dir", str(results_dir)],
        capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr + proc.stdout

    files = sorted(p.name for p in results_dir.iterdir())
    assert any(f.endswith(".pyarrow") for f in files)
    assert any(f.endswith(".json") for f in files)
    assert any(f.endswith(".md") for f in files)

    summary = json.load(open(
        next(results_dir.glob("*.json")), encoding="utf-8"))
    assert summary["result"]["original_tokens"] > 0

def test_cli_compress_decompress_flags(sample_file, tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "anatok", sample_file, "-c", "-d",
         "--no-export"],
        capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "Roundtrip:         VERIFIED" in proc.stdout
    assert "Compressed size:" in proc.stdout

def test_cli_decompress_requires_compress():
    proc = subprocess.run(
        [sys.executable, "-m", "anatok", "-d"],
        capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0
    assert "requires -c" in (proc.stderr + proc.stdout)

def test_cli_missing_path_fails(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "anatok", str(tmp_path / "nope.txt")],
        capture_output=True, text=True, timeout=60)
    assert proc.returncode == 1
    assert "Path not found" in proc.stdout

def test_cli_version():
    from anatok import __version__
    proc = subprocess.run(
        [sys.executable, "-m", "anatok", "--version"],
        capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0
    assert __version__ in proc.stdout

def test_folder_worker_and_summary_export(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    for i in range(3):
        (folder / f"f{i}.txt").write_text(f"folder file {i} content " * 20,
                                          encoding="utf-8")
    fw = FolderWorker(str(folder), operation="tokenize")
    summary = fw.process()

    assert summary["total_files"] == 3
    assert summary["errors"] == 0
    for r in summary["results"]:
        assert r["original_tokens"] > 0
        assert r.get("exports", {}).get("arrow")

def test_folder_worker_invalid_path():
    fw = FolderWorker("Z:/definitely/not/here", operation="tokenize")
    summary = fw.process()
    assert "error" in summary
