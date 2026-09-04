"""The stall watchdog, and the timeout's refusal to trust unclean samples.

Both exist because of one incident. qwen3.8-27b-mxfp8 was measured on a machine
whose compressor held 18.07GB, so `machine_is_uncontended()` correctly tagged
every sample unclean -- and it made no difference, because `_derived_timeout`
read the derived SCALAR and never asked. Decode came out at 0.1158 tok/s, so
`max_tokens / decode` alone was ~138,000s; capped at MAX_EVAL_TIMEOUT that still
bought a 2-hour per-task ceiling, and the run sat wedged for 83 minutes with zero
tasks completed and nothing tripped.
"""

from unittest.mock import MagicMock, patch

import pytest
from eval.samples import add_sample, clean_estimate
from eval.signals import _derived_timeout
from eval.watchdog import MODEL_STALL_SECONDS, check_stall


class TestCleanEstimate:
    def test_an_all_unclean_history_yields_no_estimate(self):
        """Not a number, and not zero -- None, so the caller falls back to its
        documented floor instead of doing arithmetic on a reading it distrusts."""
        caps = {}
        add_sample(caps, "decode_tokens_per_sec", 0.1158, clean=False)
        assert clean_estimate(caps, "decode_tokens_per_sec") is None

    def test_unclean_samples_do_not_drag_the_clean_median(self):
        caps = {}
        for v in (30.0, 31.0, 32.0):
            add_sample(caps, "decode_tokens_per_sec", v, clean=True)
        add_sample(caps, "decode_tokens_per_sec", 0.1158, clean=False)
        assert clean_estimate(caps, "decode_tokens_per_sec") == 31.0

    def test_a_missing_key_is_not_an_error(self):
        assert clean_estimate({}, "decode_tokens_per_sec") is None


class TestDerivedTimeoutIgnoresContendedMeasurements:
    @pytest.fixture
    def signals(self):
        def _mk(clean):
            caps = {}
            for key, val in (
                ("prefill_chars_per_sec", 132.2),
                ("decode_tokens_per_sec", 0.1158),
                ("cold_start_seconds", 74.2878),
            ):
                add_sample(caps, key, val, clean=clean)
            return {"m": {"_capabilities": caps}}
        return _mk

    def test_the_real_qwen38_numbers_produce_no_derived_timeout(self, signals):
        """The exact readings that bought a 2-hour ceiling. Unclean, so the
        timeout path must decline to use them at all."""
        with patch("eval.signals._load_eval_signals", return_value=signals(False)):
            assert _derived_timeout("m", prompt_chars=6000, max_tokens=16000) == 0

    @pytest.mark.parametrize(
        "dirty_key", ["prefill_chars_per_sec", "decode_tokens_per_sec", "cold_start_seconds"]
    )
    def test_any_single_unclean_term_disqualifies_the_derivation(self, dirty_key):
        """One term at a time, the other two clean.

        The all-three-unclean case above passes even if only ONE term consults
        clean_estimate, because the `not prefill or not decode or not cold_start`
        guard short-circuits on the first. A mutant that reverted decode to the raw
        scalar survived it. Each term needs its own fixture to be observable.
        """
        caps = {}
        for key, val in (
            ("prefill_chars_per_sec", 132.2),
            ("decode_tokens_per_sec", 0.1158),
            ("cold_start_seconds", 74.2878),
        ):
            add_sample(caps, key, val, clean=(key != dirty_key))
        with patch(
            "eval.signals._load_eval_signals", return_value={"m": {"_capabilities": caps}}
        ):
            assert _derived_timeout("m", prompt_chars=6000, max_tokens=16000) == 0

    def test_the_same_numbers_when_clean_still_derive_a_timeout(self, signals):
        """Guards against 'fixed' by breaking derivation outright: a genuinely
        slow model measured on a quiet box must still get its long timeout."""
        with patch("eval.signals._load_eval_signals", return_value=signals(True)):
            assert _derived_timeout("m", prompt_chars=6000, max_tokens=16000) > 0


class TestStallWatchdog:
    def test_a_slow_but_working_model_does_not_trip_it(self):
        """bonsai-27b-ternary-jang spent 866s on one task and completed all 30.
        A watchdog that fires on that is worse than none."""
        out = MagicMock()
        with patch("eval.watchdog.time.monotonic", return_value=866.0):
            assert check_stall("bonsai", last_completion=0.0, out=out) is False
        out.print.assert_not_called()

    def test_no_completion_past_the_limit_abandons_the_model(self):
        out = MagicMock()
        with (
            patch("eval.watchdog.time.monotonic", return_value=MODEL_STALL_SECONDS + 1),
            patch("eval.watchdog.restart_after_stall") as restart,
        ):
            assert check_stall("qwen3.8-27b-mxfp8", last_completion=0.0, out=out) is True
        printed = " ".join(str(c) for c in out.print.call_args_list)
        assert "NOT quality results" in printed, printed
        assert "qwen3.8-27b-mxfp8" in printed
        restart.assert_called_once()

    def test_it_does_not_depend_on_any_measured_rate(self):
        """The independence IS the fix. Whatever the capabilities say, 83 minutes
        with no completed task must abandon the model."""
        out = MagicMock()
        with (
            patch("eval.signals._load_eval_signals", return_value={"m": {"_capabilities": {
                "prefill_chars_per_sec": 132.2, "decode_tokens_per_sec": 0.1158,
                "cold_start_seconds": 74.2878}}}),
            patch("eval.watchdog.time.monotonic", return_value=83 * 60),
            patch("eval.watchdog.restart_after_stall"),
        ):
            assert check_stall("m", last_completion=0.0, out=out) is True


class TestRestartAfterStallIsNeverFatal:
    def test_a_failing_restart_is_reported_not_raised(self):
        """The watchdog runs mid-sweep; it must not take the sweep down with it."""
        from eval.watchdog import restart_after_stall

        out = MagicMock()
        with patch("eval.cli_runtime.restart_server", side_effect=Exception("boom")):
            restart_after_stall(out=out)
        assert "could not restart" in " ".join(str(c) for c in out.print.call_args_list)
