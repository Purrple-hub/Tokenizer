import threading

from anatok.workers import (
    CompressionWorker,
    DecompressionWorker,
    MemoryPressureTestWorker,
    TokenizationWorker,
)

def _run(worker, timeout=30):
    done = {}
    worker.callback = lambda ev, r: done.__setitem__("done", (ev, r))
    worker.error_callback = lambda e: done.__setitem__("error", e)
    worker.start()
    worker.join(timeout)
    assert not worker.is_alive(), "worker timed out"
    return done

def test_tokenization_worker(sample_file):
    done = _run(TokenizationWorker(sample_file, use_hf=False))
    assert "done" in done, done.get("error")
    ev, result = done["done"]
    assert ev == "tokenization_complete"

def test_compression_worker_roundtrip():
    done = _run(CompressionWorker(b"abcabcabcabc", method="lz77"))
    assert "done" in done, done.get("error")
    _, result = done["done"]
    assert result["compressed_size"] > 0

    done2 = _run(DecompressionWorker(result["data"], method="lz77"))
    _, decoded = done2["done"]
    assert decoded["data"] == b"abcabcabcabc"

def test_memory_pressure_worker():
    done = _run(MemoryPressureTestWorker(iterations=10))
    assert "done" in done
    ev, result = done["done"]
    assert ev == "memory_test_complete"
    assert result["iterations"] == 10

def test_worker_stop_event_is_thread_safe():
    from anatok.workers import BaseWorker

    w = BaseWorker(use_hf=False)
    w.stop()
    assert w.stopped()
    assert isinstance(w._stop_event, threading.Event)
