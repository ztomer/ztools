"""
Signal handling utilities for CLI scripts.
Provides graceful Ctrl+C / SIGTERM handling with cleanup callbacks.
"""

import os
import signal
import sys
import threading
from typing import Callable, List, Optional

from lib.tui import WARN

_cleanup_callbacks: List[Callable] = []
_shutdown_requested = False
_shutdown_lock = threading.Lock()
_drain_on_interrupt = False

EXIT_INTERRUPTED = 130


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
        already_requested = _shutdown_requested
        _shutdown_requested = True

    if already_requested:
        # Second interrupt means "stop now" — skip cleanup, which is exactly
        # what may be wedged (e.g. a browser that will not close).
        print(f"{WARN} Forced exit", flush=True)
        os._exit(EXIT_INTERRUPTED)

    if _drain_on_interrupt:
        # Return without exiting so long-running loops can observe
        # is_shutdown_requested(), unwind, and keep the work done so far.
        print(
            f"{WARN} Interrupted — finishing up (Ctrl+C again to force quit)",
            flush=True,
        )
        return

    print(f"{WARN} Interrupted — shutting down", flush=True)
    _run_cleanup()
    sys.exit(EXIT_INTERRUPTED)


def setup_signals(drain: bool = False) -> None:
    """Install SIGINT/SIGTERM handlers. Call once at startup.

    drain=False (default): the first interrupt runs cleanups and exits 130.
    drain=True: the first interrupt only sets the shutdown flag, so a loop
    polling is_shutdown_requested() can stop early and still return partial
    results. A second interrupt always force-exits.
    """
    global _drain_on_interrupt
    _drain_on_interrupt = drain
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


def is_shutdown_requested() -> bool:
    """Check if shutdown was requested."""
    return _shutdown_requested


def reset_signal_state() -> None:
    """Clear shutdown flag, drain mode, and cleanups. For tests."""
    global _shutdown_requested, _drain_on_interrupt
    with _shutdown_lock:
        _shutdown_requested = False
    _drain_on_interrupt = False
    _cleanup_callbacks.clear()


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
