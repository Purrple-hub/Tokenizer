from anatok.utils import (
    compute_hash,
    count_tokens,
    ensure_dir,
    format_bytes,
    iter_file_blocks,
    truncate_text,
    validate_path,
)

def test_compute_hash():
    h = compute_hash(b"hello")
    assert isinstance(h, str) and len(h) == 64

def test_validate_path(tmp_path):
    assert validate_path(str(tmp_path))
    assert not validate_path(str(tmp_path / "missing_xyz"))

def test_format_bytes():
    assert "KB" in format_bytes(1024)
    assert "MB" in format_bytes(1024 * 1024)

def test_count_tokens():
    assert count_tokens("hello world test") == 3

def test_truncate_text():
    assert len(truncate_text("a" * 1000, 50)) <= 53
    assert truncate_text("short", 100) == "short"

def test_iter_file_blocks(tmp_path):
    p = tmp_path / "blocks.bin"
    data = bytes(range(256)) * 100
    p.write_bytes(data)
    got = b"".join(iter_file_blocks(str(p), block_size=512))
    assert got == data

def test_ensure_dir(tmp_path):
    target = str(tmp_path / "a" / "b" / "file.txt")
    ensure_dir(target)
    assert (tmp_path / "a" / "b").is_dir()
