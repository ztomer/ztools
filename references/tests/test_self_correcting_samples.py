"""A performance measurement must be able to RECOVER from a bad reading.

The recorders kept the extreme observation -- slowest rate, longest cold start -- so a
timeout would hold on a bad run. Sound for a timeout, fatal for a measurement: the
extreme is exactly what a contended machine produces, and nothing could displace it.

A leaked plugin daemon held 31GB of this machine's 64 for a day. Everything measured
under it was wrong, and re-measuring on a healthy machine changed NOTHING, because the
worst value always won:

    nemotron-3.5-lightning-30b   recorded 0.68 tok/s   actually ~33 tok/s
    gemma-4-12b-it-mxfp8         recorded 309s cold    a 16GB model loads in ~33s

Those numbers reached conf/config.toml, docs/MODEL_QUIRKS.md and a default_model
choice. The only remedy was deleting the file by hand.
"""

from unittest.mock import patch

import pytest
from eval.samples import (
    SAMPLE_WINDOW,
    add_sample,
    estimate_from,
    machine_is_uncontended,
    migrate_scalar,
)


class TestTheEstimatorRecovers:
    """The property the old one lacked, stated directly."""

    def test_a_contaminated_reading_is_outvoted_by_clean_ones(self):
        caps = {"decode_tokens_per_sec": 0.68}      # measured during the leak
        migrate_scalar(caps, "decode_tokens_per_sec")
        assert caps["decode_tokens_per_sec"] == 0.68
        for rate in (33.0, 31.5, 34.2):
            add_sample(caps, "decode_tokens_per_sec", rate, clean=True)
        assert caps["decode_tokens_per_sec"] > 25, (
            "a bad reading is still winning; this is the defect, not a rounding issue"
        )

    def test_the_old_extreme_keeping_rule_would_not_have_recovered(self):
        """Calibration: show the previous behaviour on the same inputs, so the test
        above cannot pass for a trivial reason."""
        previous = 0.68
        for rate in (33.0, 31.5, 34.2):
            previous = min(previous, rate)          # what the recorder used to do
        assert previous == 0.68, "the old rule kept the worst value forever"

    def test_one_bad_sample_among_good_ones_does_not_move_the_estimate_much(self):
        caps = {}
        for rate in (30.0, 31.0, 32.0, 0.5, 33.0):
            add_sample(caps, "decode_tokens_per_sec", rate, clean=True)
        assert caps["decode_tokens_per_sec"] >= 30, "the median should absorb one outlier"

    def test_a_real_slowdown_is_still_tracked(self):
        """Recovery must not mean 'ignores change'. A machine that genuinely got
        slower has to be reflected, or the timeout stops covering the real run."""
        caps = {}
        for rate in (30.0, 31.0, 32.0):
            add_sample(caps, "decode_tokens_per_sec", rate, clean=True)
        for rate in (5.0, 5.2, 4.8, 5.1, 5.0):
            add_sample(caps, "decode_tokens_per_sec", rate, clean=True)
        assert caps["decode_tokens_per_sec"] < 10, "a sustained real change must land"


class TestCleanSamplesArePreferred:
    def test_clean_samples_outrank_dirty_ones(self):
        history = (
            [{"v": 0.5, "clean": False}] * 4
            + [{"v": 30.0, "clean": True}, {"v": 31.0, "clean": True}]
        )
        assert estimate_from(history) >= 30

    def test_with_no_clean_samples_it_still_answers(self):
        """A stale number beats no number, provided it can be displaced later."""
        assert estimate_from([{"v": 7.0, "clean": False}]) == 7.0

    def test_an_empty_history_is_zero_not_an_error(self):
        assert estimate_from([]) == 0.0

    def test_history_is_bounded(self):
        caps = {}
        for i in range(50):
            add_sample(caps, "k", float(i + 1), clean=True)
        assert len(caps["k_samples"]) <= SAMPLE_WINDOW * 2


class TestMigrationOfExistingScalars:
    def test_a_legacy_scalar_is_seeded_as_unclean(self):
        """Those on disk were recorded under the extreme-keeping rule and some during
        the leak, so they must not be trusted as clean baselines -- but discarding
        them would throw away the only reading some models have."""
        caps = {"cold_start_seconds": 309.0}
        migrate_scalar(caps, "cold_start_seconds")
        assert caps["cold_start_seconds_samples"][0]["clean"] is False
        assert caps["cold_start_seconds_samples"][0]["legacy"] is True

    def test_migration_happens_once(self):
        caps = {"cold_start_seconds": 309.0}
        migrate_scalar(caps, "cold_start_seconds")
        add_sample(caps, "cold_start_seconds", 12.0, clean=True)
        migrate_scalar(caps, "cold_start_seconds")
        assert sum(1 for s in caps["cold_start_seconds_samples"] if s.get("legacy")) == 1

    def test_a_missing_key_is_not_invented(self):
        caps = {}
        migrate_scalar(caps, "decode_tokens_per_sec")
        assert caps == {}


class TestContentionGating:
    """Gates on PRESSURE, not headroom. After a sweep the page cache legitimately
    holds tens of GB and 'available' drops to ~12GB on a healthy box."""

    def _with(self, swap_gb, compressor_pages):
        vm = f"Pages occupied by compressor: {compressor_pages}."
        return (
            patch("psutil.swap_memory", return_value=type("S", (), {"used": swap_gb * 1024**3})),
            patch("subprocess.run", return_value=type("R", (), {"stdout": vm})),
        )

    def test_the_leaked_state_is_rejected(self):
        """swap 12.88GB, compressor 29.3GB -- the machine during the leak."""
        swap, comp = self._with(12.88, int(29.3 * 1024**3 / 16384))
        with swap, comp:
            assert machine_is_uncontended() is False

    def test_the_healthy_state_is_accepted(self):
        """swap 1.43GB, compressor 5.1GB -- the same machine after a full sweep."""
        swap, comp = self._with(1.43, int(5.1 * 1024**3 / 16384))
        with swap, comp:
            assert machine_is_uncontended() is True

    def test_an_unreadable_probe_marks_the_sample_unverified(self):
        """Not clean, because we cannot say so -- and not dropped, because a sample
        we cannot verify is still evidence."""
        with patch("psutil.swap_memory", side_effect=OSError("nope")):
            assert machine_is_uncontended() is False


class TestTheTimeoutUsesEveryMeasuredQuantity:
    """All three terms are measured per model by `ev`. Only prefill was read back;
    decode and cold start used flat constants while the real values sat in
    conf/eval_signals.json."""

    def test_a_measured_decode_rate_is_used(self):
        from twitter.budget import _measured_or

        with patch("lib.model_caps.recorded_capability", return_value=2.0):
            assert _measured_or("m", "decode_tokens_per_sec", 8) == 2.0

    def test_an_unmeasured_model_keeps_the_pessimistic_fallback(self):
        from twitter.budget import _measured_or

        with patch("lib.model_caps.recorded_capability", return_value=None):
            assert _measured_or("m", "decode_tokens_per_sec", 8) == 8

    def test_a_slower_measured_model_gets_a_longer_timeout(self):
        from twitter.budget import _estimate_timeout

        prompt = "x" * 20000
        with patch("twitter.budget._measured_or", side_effect=lambda m, k, f: 2.0 if "decode" in k else 10.0):
            slow = _estimate_timeout(prompt, "slow-model")
        with patch("twitter.budget._measured_or", side_effect=lambda m, k, f: 40.0 if "decode" in k else 10.0):
            fast = _estimate_timeout(prompt, "fast-model")
        assert slow > fast, "a measured-slow model must get more time, not the same"

    @pytest.mark.parametrize("key", ["decode_tokens_per_sec", "cold_start_seconds"])
    def test_a_zero_or_negative_measurement_is_ignored(self, key):
        """A nonsense reading must not produce a nonsense timeout."""
        from twitter.budget import _measured_or

        with patch("lib.model_caps.recorded_capability", return_value=0):
            assert _measured_or("m", key, 99) == 99
