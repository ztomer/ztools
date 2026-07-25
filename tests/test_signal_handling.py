"""Tests for lib.signal_handling — drain mode and cleanup ordering.

The interesting property is the difference between the two modes: the default
hard-exits on the first interrupt, drain mode sets a flag and returns so a
long-running loop can unwind and keep its partial results.
"""

import signal
from unittest.mock import MagicMock

import pytest

import lib.signal_handling as sig


@pytest.fixture(autouse=True)
def clean_signal_state():
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)
    sig.reset_signal_state()
    yield
    sig.reset_signal_state()
    signal.signal(signal.SIGINT, original_sigint)
    signal.signal(signal.SIGTERM, original_sigterm)


class TestDrainMode:
    def test_drain_first_interrupt_sets_flag_without_exiting(self, capsys):
        sig.setup_signals(drain=True)
        assert sig.is_shutdown_requested() is False

        # Must not raise SystemExit — that is the whole point of drain mode.
        sig._signal_handler(signal.SIGINT, None)

        assert sig.is_shutdown_requested() is True
        assert "Ctrl+C again" in capsys.readouterr().out

    def test_drain_does_not_run_cleanups_on_first_interrupt(self):
        """Cleanups belong to the caller's own unwind path in drain mode."""
        cleanup = MagicMock()
        sig.setup_signals(drain=True)
        sig.register_cleanup(cleanup)

        sig._signal_handler(signal.SIGINT, None)

        cleanup.assert_not_called()

    def test_default_mode_runs_cleanups_and_exits(self):
        cleanup = MagicMock()
        sig.setup_signals(drain=False)
        sig.register_cleanup(cleanup)

        with pytest.raises(SystemExit) as excinfo:
            sig._signal_handler(signal.SIGINT, None)

        assert excinfo.value.code == 130
        cleanup.assert_called_once()

    def test_second_interrupt_force_exits_without_cleanup(self, monkeypatch):
        """Force-quit must not wait on a cleanup that may itself be wedged."""
        cleanup = MagicMock()
        hard_exit = MagicMock()
        monkeypatch.setattr(sig.os, "_exit", hard_exit)
        sig.setup_signals(drain=True)
        sig.register_cleanup(cleanup)

        sig._signal_handler(signal.SIGINT, None)  # first: flag only
        sig._signal_handler(signal.SIGINT, None)  # second: force

        hard_exit.assert_called_once_with(130)
        cleanup.assert_not_called()

    def test_setup_signals_defaults_to_non_drain(self):
        sig.setup_signals()
        with pytest.raises(SystemExit):
            sig._signal_handler(signal.SIGTERM, None)

    def test_failing_cleanup_does_not_block_the_others(self):
        order = []
        sig.setup_signals(drain=False)
        sig.register_cleanup(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        sig.register_cleanup(lambda: order.append("second"))

        with pytest.raises(SystemExit):
            sig._signal_handler(signal.SIGINT, None)

        assert order == ["second"]


class TestGracefulShutdown:
    def test_registers_and_unregisters(self):
        cleanup = MagicMock()
        with sig.GracefulShutdown(cleanup):
            assert cleanup in sig._cleanup_callbacks
        assert cleanup not in sig._cleanup_callbacks
        # Exiting the block must not fire the cleanup — the caller does that.
        cleanup.assert_not_called()
