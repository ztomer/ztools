"""`Pages free` is not available memory, and a gate that thinks it is misfires.

A 64GB Mac with 7.2GB genuinely in use was reported as having "21.9GB available"
because `Pages free` was read as headroom. macOS keeps reclaimable memory in
`inactive` and `speculative`; the real figure was 45.6GB. These tests pin the
arithmetic so the misreading cannot come back.

Each was proven able to fail by reverting its guard first (rule #2).
"""

import pytest

# Imported by value: conftest's `deterministic_machine_contention` fixture is
# autouse and patches `eval.memory.is_thrashing` / `reclaimable_available_gb`
# so no test outcome depends on the developer's machine. Binding here, at import
# time, holds the real functions -- which is the point of this module's tests.
from eval.memory import (
    MAX_CLEAN_COMPRESSOR_GB,
    MAX_CLEAN_SWAP_GB,
    RATE_WINDOW_SECONDS,
    NotSupportedHere,
    compression_rate,
    is_thrashing,
    pressure,
    reclaimable_available_gb,
)

#: The real vm_stat shape, in pages, from the machine that produced the misreading.
#: 16384-byte pages: free 26.7GB, inactive 16.9GB, speculative 2.0GB, and
#: 16.7GB file-backed of which essentially all is already inactive.
IDLE_MAC = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                  1746343.
Pages active:                                 280706.
Pages inactive:                              1108553.
Pages speculative:                            127847.
Pages throttled:                                   0.
Pages wired down:                             192277.
Pages purgeable:                               28108.
Pages occupied by compressor:                      0.
File-backed pages:                           1096931.
"""

#: The same box just after a sweep: the page cache holds the previous model's
#: weights as ACTIVE file-backed pages, so psutil reports little available even
#: though the kernel would evict them for free. This is the state the headroom-only
#: gate refused to run on -- the one you most want readings from.
AFTER_A_SWEEP = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                   131072.
Pages active:                                2097152.
Pages inactive:                               131072.
Pages speculative:                             13107.
Pages wired down:                             192277.
Pages purgeable:                               13107.
Pages occupied by compressor:                 327680.
File-backed pages:                           1966080.
"""


def _fake_vm_stat(monkeypatch, text):
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: type("R", (), {"stdout": text})(),
    )


def _fake_psutil(monkeypatch, available_gb, swap_gb=0.0):
    G = 1024**3

    monkeypatch.setattr(
        "psutil.virtual_memory",
        lambda: type("V", (), {"available": available_gb * G})(),
    )
    monkeypatch.setattr(
        "psutil.swap_memory", lambda: type("S", (), {"used": swap_gb * G})()
    )


class TestReclaimableAvailable:
    def test_an_idle_box_reports_what_psutil_reports(self, monkeypatch):
        """Nothing file-backed is active here, so there is nothing to add."""
        _fake_vm_stat(monkeypatch, IDLE_MAC)
        _fake_psutil(monkeypatch, available_gb=45.6)
        # file-backed 16.7 - inactive 16.9 - speculative 2.0 -> clamped to 0,
        # plus 0.4GB purgeable.
        assert reclaimable_available_gb() == pytest.approx(46.0, abs=0.2)

    def test_page_cache_holding_model_weights_counts_as_reclaimable(self, monkeypatch):
        """The bug this module exists for.

        psutil says 4GB available. 30GB of file-backed pages are ACTIVE -- the
        last model's weights, evictable at no cost. A headroom gate reading only
        psutil would refuse a 20GB model on a machine that would run it fine.
        """
        _fake_vm_stat(monkeypatch, AFTER_A_SWEEP)
        _fake_psutil(monkeypatch, available_gb=4.0)
        reclaimable = reclaimable_available_gb()
        assert reclaimable > 25, f"page cache not counted: {reclaimable:.1f}GB"

    def test_the_estimate_never_exceeds_the_truth_direction(self, monkeypatch):
        """Over-subtracting is deliberate: understate, never overstate."""
        _fake_vm_stat(monkeypatch, IDLE_MAC)
        _fake_psutil(monkeypatch, available_gb=45.6)
        # 64GB machine: whatever we add, it cannot claim more than exists.
        assert reclaimable_available_gb() < 64

    def test_a_broken_vm_stat_raises_rather_than_guessing(self, monkeypatch):
        """It used to catch this and return psutil's figure alone.

        That turned "vm_stat is broken" into a number that looks fine and is
        simply wrong -- the exact failure this module exists to stop. Every
        quantity here comes from vm_stat; without it there is no answer to give.
        """
        _fake_psutil(monkeypatch, available_gb=45.6)
        monkeypatch.setattr(
            "subprocess.run", lambda *a, **k: (_ for _ in ()).throw(OSError("nope"))
        )
        with pytest.raises(OSError):
            reclaimable_available_gb()

    def test_a_non_macos_platform_is_a_hard_failure(self, monkeypatch):
        """House rule #3: an unsupported platform fails, it does not degrade.

        The eval path is macOS-only end to end -- osaurus, Metal, the GPU lock --
        so a Linux branch here would be dead code whose only effect is to turn a
        missing tool into a plausible number.
        """
        monkeypatch.setattr("sys.platform", "linux")
        with pytest.raises(NotSupportedHere, match="macOS"):
            reclaimable_available_gb()


class TestCompressionRate:
    """The rate itself, not a monkeypatched stand-in for it.

    Every is_thrashing test patches `compression_rate` wholesale, so none of them
    exercise its internals -- a mutation that read the GB-scaled counters instead
    of the raw ones survived the entire suite. Compressions is an EVENT count;
    scaling it by the page size turns a rate of thousands into 0.00008 and makes
    a thrashing box read clean.
    """

    def _counter_pair(self, monkeypatch, first, second):
        outputs = iter(
            [
                "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                f"Compressions:  {first}.\n",
                "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                f"Compressions:  {second}.\n",
            ]
        )
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **k: type("R", (), {"stdout": next(outputs)})(),
        )
        monkeypatch.setattr("time.sleep", lambda _s: None)

    def test_the_rate_is_raw_events_per_second(self, monkeypatch):
        """5000 compressions across the window is 5000/s, not 5000 x page size."""
        self._counter_pair(monkeypatch, 1_000_000, 1_005_000)
        assert compression_rate() == pytest.approx(5000.0 / RATE_WINDOW_SECONDS)

    def test_the_window_actually_divides(self, monkeypatch):
        """A window of 1.0 cannot see a missing division.

        The obvious test -- 5000 events, assert 5000/s -- passes whether or not
        the code divides by the window, because the window IS 1.0. Same shape as
        the `_names_match` fixture already recorded in BACKLOG: a case that is
        true under both the original and the mutant tests nothing. Use a window
        the division can be seen through.
        """
        monkeypatch.setattr("eval.memory.RATE_WINDOW_SECONDS", 2.0)
        self._counter_pair(monkeypatch, 1_000_000, 1_005_000)
        assert compression_rate() == pytest.approx(2500.0)

    def test_an_idle_machine_rates_zero(self, monkeypatch):
        """Measured on this box: Compressions moved by exactly 0 over 10 seconds."""
        self._counter_pair(monkeypatch, 229_828_396, 229_828_396)
        assert compression_rate() == 0.0

    def test_a_counter_that_went_backwards_is_clamped(self, monkeypatch):
        """Counters reset; a negative rate must not read as 'very idle'."""
        self._counter_pair(monkeypatch, 5_000, 100)
        assert compression_rate() == 0.0

    def test_a_missing_counter_is_cannot_tell(self, monkeypatch):
        self._counter_pair(monkeypatch, 1, 2)
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **k: type("R", (), {"stdout": "Pages free: 12.\n"})(),
        )
        monkeypatch.setattr("time.sleep", lambda _s: None)
        assert compression_rate() is None


class TestPressure:
    def test_a_quiet_machine_is_not_thrashing(self, monkeypatch):
        _fake_vm_stat(monkeypatch, IDLE_MAC)
        _fake_psutil(monkeypatch, available_gb=45.6, swap_gb=0.0)
        assert is_thrashing() is False

    def test_swap_above_the_threshold_is_thrashing(self, monkeypatch):
        _fake_vm_stat(monkeypatch, IDLE_MAC)
        _fake_psutil(monkeypatch, available_gb=45.6, swap_gb=MAX_CLEAN_SWAP_GB + 1)
        assert is_thrashing() is True

    def test_a_full_compressor_is_thrashing_even_when_the_counters_look_idle(
        self, monkeypatch
    ):
        """The guard stays OVER-cautious, deliberately, and this pins why.

        A rate-gated version of this check was written and then measured against
        a run known to be thrashing (compressor climbing 2.5 -> 28.4GB). The
        compression counter read ZERO in 76 of 83 one-second samples, and the
        compressor-level delta read zero in 78 of 80: both counters are bursty,
        so a rate gate clears a contaminated machine about 19 times in 20.

        An over-cautious guard costs a retried measurement. An under-cautious one
        writes a contaminated number into eval_signals.json, where it feeds the
        median, sizes the derived timeout, and hardens into config and docs --
        the mechanism that put 14.7 tok/s in MODEL_QUIRKS.md for a model that
        does 0.11. Until a signal is reliable under load, this fails toward
        refusing to record.
        """
        pages = int((MAX_CLEAN_COMPRESSOR_GB + 15) * 1024**3 / 16384)
        _fake_vm_stat(
            monkeypatch,
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
            f"Pages occupied by compressor:  {pages}.\n"
            "Compressions:  229828396.\n",
        )
        _fake_psutil(monkeypatch, available_gb=10.0, swap_gb=0.0)
        assert is_thrashing() is True

    def test_swap_in_use_is_thrashing(self, monkeypatch):
        """Swap already written to disk is unambiguous."""
        _fake_vm_stat(monkeypatch, IDLE_MAC)
        _fake_psutil(monkeypatch, available_gb=10.0, swap_gb=MAX_CLEAN_SWAP_GB + 5)
        assert is_thrashing() is True

    def test_cannot_tell_is_none_not_false(self, monkeypatch):
        """None and False must not be conflated: one is ignorance, one is a verdict."""

        def boom():
            raise RuntimeError("no psutil")

        monkeypatch.setattr("psutil.swap_memory", boom)
        assert pressure() is None
        assert is_thrashing() is None

    def test_the_page_size_comes_from_the_header(self, monkeypatch):
        """A wrong page size scales every figure in this module."""
        _fake_vm_stat(
            monkeypatch,
            "Mach Virtual Memory Statistics: (page size of 4096 bytes)\n"
            "Pages occupied by compressor:  262144.\n",
        )
        _fake_psutil(monkeypatch, available_gb=10.0, swap_gb=0.0)
        # 262144 * 4096 = 1GB, not the 4GB a hardcoded 16384 would report.
        assert pressure()[1] == pytest.approx(1.0, abs=0.01)
