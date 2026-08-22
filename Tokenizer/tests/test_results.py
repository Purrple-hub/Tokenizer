import json

import pyarrow.feather as feather

from anatok.managers import get_tokenization_manager
from anatok.results import (
    TokenArrowWriter,
    export_folder_summary,
    export_result_files,
    format_markdown_report,
)

def test_arrow_writer_roundtrip(tmp_path):
    path = tmp_path / "out.pyarrow"
    with TokenArrowWriter(str(path), source_path="x.txt",
                          metadata={"model": "test"}) as w:
        for i in range(100):
            w.add_row("x.txt", i, f"tok_{i}")
    table = feather.read_table(str(path))
    assert table.num_rows == 100
    assert table.column_names == ["doc", "seq", "token_id", "token"]
    meta = json.loads(table.schema.metadata[b"anatok_meta"])
    assert meta["source_path"].endswith("x.txt")
    assert meta["model"] == "test"

def test_arrow_writer_batches_flush(tmp_path):
    path = tmp_path / "batches.pyarrow"
    with TokenArrowWriter(str(path), source_path="") as w:
        assert w.BATCH_ROWS == 8192
        for i in range(w.BATCH_ROWS * 2 + 5):
            w.add_row("", i, "t")
        assert w.row_count == w.BATCH_ROWS * 2 + 5
    table = feather.read_table(str(path))
    assert table.num_rows == w.BATCH_ROWS * 2 + 5

def test_hf_run_exports_full_stream(sample_file, tmp_path):
    tm = get_tokenization_manager()
    result = tm.process_path(sample_file, use_hf=True,
                             results_dir=str(tmp_path / "res"))
    exports = result.get("exports")
    assert exports and exports["arrow"]

    table = feather.read_table(exports["arrow"])
    assert table.num_rows == result["original_tokens"]
    docs = set(table.column("doc").to_pylist())
    assert docs == {sample_file}
    ids = table.column("token_id").to_pylist()
    assert all(i is not None and i >= 0 for i in ids)

    summary = json.load(open(exports["json"], encoding="utf-8"))
    assert summary["result"]["original_tokens"] == \
        result["original_tokens"]

    md = open(exports["md"], encoding="utf-8").read()
    assert "# anatok Tokenization Report" in md
    assert "## Artifacts" in md

def test_custom_run_exports_groups_with_null_ids(sample_file, tmp_path):
    tm = get_tokenization_manager()
    result = tm.process_path(sample_file, use_hf=False,
                             results_dir=str(tmp_path / "res"))
    exports = result.get("exports")
    assert exports and exports["arrow"]
    table = feather.read_table(exports["arrow"])
    assert table.num_rows == result["original_tokens"]
    assert all(v is None for v in table.column("token_id").to_pylist())

def test_export_result_files_sidecars(tmp_path):
    result = {"file": "a.txt", "method": "hf_tokenizer",
              "original_tokens": 12, "optimized_tokens": 10,
              "duplicates_removed": 2, "compression_ratio": 2 / 12,
              "preview_tokens": ["hello", "world"]}
    exports = export_result_files(result, results_dir=str(tmp_path),
                                  stem="manual")
    assert exports["arrow"] is None
    loaded = json.load(open(exports["json"], encoding="utf-8"))
    assert loaded["result"]["original_tokens"] == 12
    md_text = open(exports["md"], encoding="utf-8").read()
    assert "Total tokens | 12" in md_text

def test_export_folder_summary(tmp_path):
    summary = {
        "path": str(tmp_path), "operation": "tokenize",
        "total_files": 2, "processed": 2, "skipped": 0, "errors": 0,
        "results": [
            {"file": str(tmp_path / "a.txt"), "original_tokens": 30,
             "optimized_tokens": 20, "compression_ratio": 1 / 3},
            {"file": str(tmp_path / "skip.bin"), "skipped": True},
        ],
    }
    out = export_folder_summary(summary, results_dir=str(tmp_path),
                                stem="folder_run")
    loaded = json.load(open(out["json"], encoding="utf-8"))
    assert loaded["summary"]["total_files"] == 2
    md_text = open(out["md"], encoding="utf-8").read()
    assert "| a.txt | 30 | 20 | 33.33% |" in md_text
    assert "skipped" in md_text

def test_markdown_report_handles_compression_block():
    result = {"file": "z.bin", "method": "hf_tokenizer",
              "original_tokens": 5, "compressed":
                  {"original_size": 100, "compressed_size": 40,
                   "ratio": 0.4},
              "roundtrip_ok": True, "memory": {"peak_usage_mb": 1.5}}
    text = format_markdown_report(result)
    assert "## Compression (zlib)" in text
    assert "Roundtrip verified: yes" in text
