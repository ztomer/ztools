"""
Signal handling utilities for CLI scripts.
Provides graceful Ctrl+C / SIGTERM handling with cleanup callbacks.
"""
import signal
import sys
import threading
from typing import Callable, List, Optional

_cleanup_callbacks: List[Callable] = []
_shutdown_requested = False
_shutdown_lock = threading.Lock()


def register_cleanup(callback: Callable) -> None:
    """Register a cleanup callback to run on shutdown."""
    _cleanup_callbacks.append(callback)


def _run_cleanup() -> None:
    """Run all registered cleanup callbacks."""
    for callback in _cleanup_callbacks:
        try:
            callback()
        except Exception:
            pass


def _signal_handler(signum: int, frame: Optional[object]) -> None:
    global _shutdown_requested
    with _shutdown_lock:
        if _shutdown_requested:
            sys.exit(130)
        _shutdown_requested = True
    print("\n=== Interrupted — shutting down ===", flush=True)
    _run_cleanup()
    sys.exit(130)


def setup_signals() -> None:
    """Install SIGINT/SIGTERM handlers. Call once at startup."""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


def is_shutdown_requested() -> bool:
    """Check if shutdown was requested."""
    return _shutdown_requested


class GracefulShutdown:
    """Context manager for graceful shutdown with auto-registration."""
    
    def __init__(self, cleanup_fn: Callable = None):
        self.cleanup_fn = cleanup_fn
    
    def __enter__(self):
        if self.cleanup_fn:
            register_cleanup(self.cleanup_fn)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup_fn:
            try:
                _cleanup_callbacks.remove(self.cleanup_fn)
            except ValueError:
                pass
