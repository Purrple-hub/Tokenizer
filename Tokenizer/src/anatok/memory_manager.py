"""Real-process memory tracking for the Tokenizer project.

Replaces the former fixed RAM "pool" allocator (which reserved 100MB+
without ever storing useful data) with lightweight bookkeeping plus true
process-memory sampling:

- Current/peak RSS tracking via psutil (if installed) or the Windows
  GetProcessMemoryInfo API, falling back gracefully elsewhere.
- Optional low-frequency background sampler thread for accurate peaks.
- Allocation bookkeeping (sizes/counts only -- no memory is reserved).
- Public API kept backward compatible with the previous module.
"""

import atexit
import logging
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def _read_process_memory() -> tuple:
    """Return (rss_bytes, virtual_bytes) for the current process.

    Tries psutil first, then the Windows performance counter API,
    then Unix getrusage. Returns (0, 0) when nothing is available.
    """
    try:
        import psutil

        mem = psutil.Process().memory_info()
        return mem.rss, mem.vms
    except Exception:
        pass

    try:
        import ctypes

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        psapi = ctypes.WinDLL("psapi")
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
            return int(pmc.WorkingSetSize), int(pmc.PagefileUsage)
    except Exception:
        pass

    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        return int(rss), 0
    except Exception:
        pass

    return 0, 0

class MemoryChunk:
    """Bookkeeping record for a logical allocation (no memory reserved)."""

    __slots__ = ("address", "size", "offset", "allocated_time",
                 "ref_count", "last_accessed", "is_active")

    def __init__(self, address: int, size: int, offset: int):
        self.address = address
        self.size = size
        self.offset = offset
        self.allocated_time = time.time()
        self.ref_count = 1
        self.last_accessed = self.allocated_time
        self.is_active = True

class MemoryManager:
    """Process-memory monitor with lightweight allocation bookkeeping.

    Tracks real resident memory (RSS) of the running process and keeps
    counters for logical allocations issued through allocate()/release().
    No memory pool is reserved; the addresses handed out are synthetic
    identifiers used purely for accounting.
    """

    SAMPLE_INTERVAL = 0.25

    def __init__(self, pool_size: int = 0, gc_threshold: float = 0.8,
                 auto_compact: bool = True):
        self.pool_size = 0
        self.gc_threshold = gc_threshold
        self.auto_compact = auto_compact

        self._chunks: List[MemoryChunk] = []
        self._next_address = 1
        self._tracked_total_allocated = 0
        self._tracked_peak = 0
        self._allocation_count = 0
        self._deallocation_count = 0

        self._lock = threading.RLock()
        self._initialized = False

        self._rss = 0
        self._vms = 0
        self._peak_rss = 0
        self._sampler: Optional[threading.Thread] = None
        self._stop_sampler = threading.Event()

        self._log("MemoryManager ready (real-RSS tracking mode)")

    @staticmethod
    def _log(message: str, level: str = "info"):
        logger.log(getattr(logging, level.upper(), logging.INFO), message)

    def initialize(self) -> bool:
        """Start memory monitoring. Cheap and idempotent."""
        with self._lock:
            if self._initialized:
                return True
            self.sample()
            self._start_sampler()
            self._initialized = True
            self._log(f"Monitoring started. Initial RSS: {self._rss // 1024}KB")
            return True

    def shutdown(self):
        """Stop the background sampler thread."""
        self._stop_sampler.set()
        sampler = getattr(self, "_sampler", None)
        if sampler is not None and sampler.is_alive():
            sampler.join(timeout=1.0)
        with self._lock:
            self._sampler = None
            self._initialized = False

    def _start_sampler(self):
        if self._sampler is not None and self._sampler.is_alive():
            return
        self._stop_sampler.clear()

        def _run():
            while not self._stop_sampler.wait(self.SAMPLE_INTERVAL):
                self.sample()

        self._sampler = threading.Thread(
            target=_run, name="MemorySampler", daemon=True)
        self._sampler.start()

    def sample(self) -> int:
        """Refresh current RSS immediately. Returns current RSS in bytes."""
        rss, vms = _read_process_memory()
        with self._lock:
            self._rss = rss
            self._vms = vms
            if rss > self._peak_rss:
                self._peak_rss = rss
        return rss

    def get_current_usage(self) -> int:
        """Current resident memory of the process in bytes."""
        return self._rss

    def get_peak_usage(self) -> int:
        """Peak resident memory observed (bytes)."""
        with self._lock:
            return self._peak_rss

    def allocate(self, size: int, align: int = 8) -> Optional[int]:
        """Register a logical allocation. Returns a synthetic address.

        No memory is reserved; this only records size/count statistics so
        existing callers keep working.
        """
        if not self._initialized:
            self.initialize()

        if size <= 0:
            return None

        aligned = (size + align - 1) // align * align
        with self._lock:
            addr = self._next_address
            self._next_address += aligned
            chunk = MemoryChunk(addr, aligned, addr)
            self._chunks.append(chunk)
            self._allocation_count += 1
            self._tracked_total_allocated += aligned
            active = sum(c.size for c in self._chunks if c.is_active)
            if active > self._tracked_peak:
                self._tracked_peak = active
        return addr

    def release(self, address: int) -> bool:
        """Release a previously registered logical allocation."""
        with self._lock:
            for chunk in self._chunks:
                if chunk.address == address and chunk.is_active:
                    chunk.is_active = False
                    self._deallocation_count += 1
                    self._prune_if_needed()
                    return True
        self._log(f"Attempted to release unknown address {address}",
                  level="warning")
        return False

    def _prune_if_needed(self):
        """Drop inactive records when they dominate the list (amortized O(1))."""
        if len(self._chunks) > 64 and \
                sum(1 for c in self._chunks if not c.is_active) > len(self._chunks) // 2:
            self._chunks = [c for c in self._chunks if c.is_active]

    def allocate_chunk(self, size: int) -> Optional[Dict[str, Any]]:
        """Compatibility wrapper around allocate()."""
        addr = self.allocate(size)
        if addr is None:
            return None
        with self._lock:
            chunk_id = max(len(self._chunks) - 1, 0)
        return {
            "address": addr,
            "size": size,
            "offset": addr,
            "chunk_id": chunk_id,
            "timestamp": time.time(),
        }

    def release_chunk(self, chunk_info: Dict[str, Any]) -> bool:
        """Release a chunk created by allocate_chunk()."""
        addr = (chunk_info or {}).get("address")
        if addr is None:
            return False
        return self.release(addr)

    def trigger_gc(self):
        """Run Python's garbage collector and refresh RSS sampling."""
        import gc

        gc.collect()
        self.sample()

    _trigger_gc_if_needed = trigger_gc
    _perform_garbage_collection = trigger_gc

    def get_stats(self) -> Dict[str, Any]:
        """Return combined real-memory and bookkeeping statistics."""
        with self._lock:
            active = sum(c.size for c in self._chunks if c.is_active)
            inactive = len(self._chunks) - \
                sum(1 for c in self._chunks if c.is_active)
            tracked_peak = self._tracked_peak
            rss = self._rss
            peak = self._peak_rss

        capacity = tracked_peak if tracked_peak > 0 else 1
        return {
            "rss_bytes": rss,
            "current_rss": rss,
            "virtual_bytes": self._vms,
            "peak_rss_bytes": peak,
            "pool_size": tracked_peak,
            "pool_offset": active,
            "actual_used": active,
            "free": max(tracked_peak - active, 0),
            "pool_ratio": active / capacity,
            "total_allocated": self._tracked_total_allocated,
            "total_freed": self._tracked_total_allocated - active,
            "allocation_count": self._allocation_count,
            "deallocation_count": self._deallocation_count,
            "active_chunks": len(self._chunks) - inactive,
            "inactive_chunks": inactive,
            "peak_usage": peak,
            "memory_efficiency": (active / tracked_peak * 100.0)
            if tracked_peak > 0 else 100.0,
        }

    def reset(self):
        """Clear bookkeeping state (monitoring keeps running)."""
        with self._lock:
            self._chunks.clear()
            self._tracked_total_allocated = 0
            self._tracked_peak = 0
            self._allocation_count = 0
            self._deallocation_count = 0
        self._log("Bookkeeping reset")

    @staticmethod
    def memory_pressure_test(iterations: int = 100) -> Dict[str, Any]:
        """Exercise allocate/release paths while measuring real RSS."""
        mm = get_memory_manager()
        mm.initialize()

        stats_before = mm.get_stats()
        start = time.perf_counter()
        addresses: List[int] = []

        for i in range(iterations):
            size = 1024 * ((i % 10) + 1)
            addr = mm.allocate(size)
            if addr is not None:
                addresses.append(addr)

        for addr in addresses:
            mm.release(addr)

        mm.sample()
        elapsed = time.perf_counter() - start
        stats_after = mm.get_stats()

        return {
            "iterations": iterations,
            "elapsed_time": elapsed,
            "allocations_successful": len(addresses),
            "avg_alloc_time": elapsed / max(len(addresses), 1),
            "memory_leaked": stats_after["pool_offset"]
            - stats_before["pool_offset"],
            "pool_efficiency": stats_after["memory_efficiency"],
            "rss_before": stats_before["rss_bytes"],
            "rss_after": stats_after["rss_bytes"],
            "final_stats": stats_after,
        }

_memory_manager = MemoryManager()

def get_memory_manager() -> MemoryManager:
    """Get the global MemoryManager instance."""
    return _memory_manager

atexit.register(_memory_manager.shutdown)

def create_token_stream_chunk(token_count: int,
                              tokens_per_byte: int = 1,
                              estimated_token_size: int = 4) -> Dict[str, Any]:
    """Create a logical chunk sized for a token stream (bookkeeping only)."""
    required_size = token_count * estimated_token_size // tokens_per_byte
    return get_memory_manager().allocate_chunk(max(required_size, 256))

def release_token_stream_chunk(chunk_info: Dict[str, Any]):
    """Release a chunk created by create_token_stream_chunk()."""
    get_memory_manager().release_chunk(chunk_info)

def memory_pressure_test(iterations: int = 100) -> Dict[str, Any]:
    """Convenience wrapper around MemoryManager.memory_pressure_test."""
    return MemoryManager.memory_pressure_test(iterations)
