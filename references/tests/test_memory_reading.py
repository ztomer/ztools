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

    def test_an_unreadable_machine_reports_zero_not_infinity(self, monkeypatch):
        def boom():
            raise RuntimeError("no psutil")

        monkeypatch.setattr("psutil.virtual_memory", boom)
        assert reclaimable_available_gb() == 0.0

    def test_a_broken_vm_stat_still_returns_the_psutil_figure(self, monkeypatch):
        _fake_psutil(monkeypatch, available_gb=45.6)
        monkeypatch.setattr(
            "subprocess.run", lambda *a, **k: (_ for _ in ()).throw(OSError("nope"))
        )
        assert reclaimable_available_gb() == pytest.approx(45.6, abs=0.1)


class TestPressure:
    def test_a_quiet_machine_is_not_thrashing(self, monkeypatch):
        _fake_vm_stat(monkeypatch, IDLE_MAC)
        _fake_psutil(monkeypatch, available_gb=45.6, swap_gb=0.0)
        assert is_thrashing() is False

    def test_swap_above_the_threshold_is_thrashing(self, monkeypatch):
        _fake_vm_stat(monkeypatch, IDLE_MAC)
        _fake_psutil(monkeypatch, available_gb=45.6, swap_gb=MAX_CLEAN_SWAP_GB + 1)
        assert is_thrashing() is True

    def test_a_full_compressor_is_thrashing(self, monkeypatch):
        """The 31GB-leak state: compressor at 29.3GB, swap at 12.88GB."""
        pages = int((MAX_CLEAN_COMPRESSOR_GB + 5) * 1024**3 / 16384)
        _fake_vm_stat(
            monkeypatch,
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
            f"Pages occupied by compressor:  {pages}.\n",
        )
        _fake_psutil(monkeypatch, available_gb=10.0, swap_gb=0.0)
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
