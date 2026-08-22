import pytest

@pytest.fixture(autouse=True)
def _isolated_results(tmp_path, monkeypatch):
    """Keep every test's exported artifacts inside its own tmp dir."""
    monkeypatch.setenv("ANATOK_RESULTS_DIR", str(tmp_path / "results"))

@pytest.fixture()
def sample_file(tmp_path):
    """A small deterministic text file."""
    p = tmp_path / "sample.txt"
    p.write_text("The quick brown fox jumps over the lazy dog. " * 20,
                 encoding="utf-8")
    return str(p)
