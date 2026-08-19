"""The two gates that stop a contended box producing a confident wrong number.

Both directions of each gate are exercised here -- the reject AND the accept --
because a gate only tested in the direction it fires is a gate nobody has proven
lets real work through. No 28.8GB model is needed for either: the memory
estimate and the lock predicate are both injectable seams.

Each test was proven able to fail by reverting its guard first (rule #2).
"""

import pytest
from eval import samples
from eval.cli_runtime import (
    OVERSIZE_MEMORY_FRACTION,
    OVERSIZE_OVERRIDE_ENV,
    oversize_refusal,
)

# Imported BY VALUE, deliberately. conftest's `deterministic_machine_contention`
# fixture is autouse and patches `eval.samples.machine_is_uncontended` to return
# True for every test, so asserting through the module attribute would assert
# against a MagicMock. Binding the name at import time -- before the fixture runs
# -- holds the real function, which is the same technique
# test_self_correcting_samples.py already uses.
from eval.samples import gpu_is_contended, machine_is_uncontended


class TestOversizeRefusal:
    def test_a_model_that_fits_is_accepted(self):
        """The accept direction: 10GB against 64GB reclaimable must just run."""
        assert oversize_refusal(10, 64, thrashing=False) == ""

    def test_a_model_at_the_limit_is_accepted(self):
        """Boundary: exactly at the fraction is still allowed."""
        assert oversize_refusal(0.8 * 50, 50, thrashing=False) == ""

    def test_the_qwen_shape_FITS_on_a_settled_64gb_box(self):
        """28.8GB against the 45.6GB actually reclaimable -- it fits.

        This test asserted the opposite until the figure behind it was checked.
        It was written against "21.9GB available", which was `Pages free` from
        vm_stat -- not available memory. macOS holds reclaimable memory in
        `inactive` and `speculative`, and the real figure on that same idle box
        was 45.6GB. The model was never too big; the reading was wrong.
        """
        assert oversize_refusal(28.8, 45.6, thrashing=False) == ""

    def test_a_model_that_genuinely_does_not_fit_is_refused(self):
        refusal = oversize_refusal(60, 45.6, thrashing=False)
        assert refusal != ""
        assert "60GB" in refusal
        assert "46GB reclaimable" in refusal

    def test_a_thrashing_machine_is_refused_however_much_it_claims_to_have(self):
        """Pressure is asked FIRST and is disqualifying on its own.

        This is the branch the first version of the gate did not have. Swap and
        compressor describe a machine already paying for memory it does not
        have; headroom does not, because clean file-backed pages holding the
        last model's weights read as unavailable while costing nothing to evict.
        """
        refusal = oversize_refusal(1, 999, thrashing=True)
        assert refusal != ""
        assert "already paging" in refusal

    def test_cannot_tell_does_not_refuse_on_its_own(self):
        """None is "cannot tell", which is not evidence of thrashing."""
        assert oversize_refusal(10, 64, thrashing=None) == ""

    def test_the_refusal_names_its_escape_hatch(self):
        """A gate that cannot be got past deliberately gets worked around."""
        assert OVERSIZE_OVERRIDE_ENV in oversize_refusal(70, 32, thrashing=False)
        assert OVERSIZE_OVERRIDE_ENV in oversize_refusal(1, 999, thrashing=True)

    def test_the_explicit_override_lets_it_through(self):
        assert oversize_refusal(70, 32, allow=True, thrashing=False) == ""

    def test_the_override_also_beats_the_pressure_gate(self):
        """Measuring a model on a loaded box is a legitimate experiment."""
        assert oversize_refusal(70, 32, allow=True, thrashing=True) == ""

    def test_the_env_override_lets_it_through(self, monkeypatch):
        monkeypatch.setenv(OVERSIZE_OVERRIDE_ENV, "1")
        assert oversize_refusal(70, 32, thrashing=False) == ""

    def test_an_unset_env_var_does_not_override(self, monkeypatch):
        monkeypatch.delenv(OVERSIZE_OVERRIDE_ENV, raising=False)
        assert oversize_refusal(70, 32, thrashing=False) != ""

    def test_the_threshold_is_the_documented_one(self):
        """Carried forward unchanged from the warning this replaced: the gate
        changes the consequence, not silently also the threshold."""
        assert OVERSIZE_MEMORY_FRACTION == 0.8


class TestGpuContentionGate:
    def test_a_peer_holding_the_lock_makes_the_sample_unclean(self, monkeypatch):
        """The recorded failure: a peer's eval was tagged CLEAN because swap and
        compressor cannot see the GPU."""
        monkeypatch.setattr(
            "lib.gpu_lock.foreign_holder", lambda: "session-2 (pid 999) eval qwen"
        )
        assert gpu_is_contended() is True
        assert machine_is_uncontended() is False

    def test_a_free_lock_does_not_block_a_clean_sample(self, monkeypatch):
        """The accept direction -- otherwise nothing could ever be recorded."""
        monkeypatch.setattr("lib.gpu_lock.foreign_holder", lambda: None)
        assert gpu_is_contended() is False

    def test_our_own_hold_is_not_contention(self, monkeypatch):
        """foreign_holder() returns None when the lock is ours; an eval holds it
        for its whole run and must still record samples."""
        monkeypatch.setattr("lib.gpu_lock.foreign_holder", lambda: None)
        monkeypatch.setattr(samples, "MAX_CLEAN_SWAP_GB", 10_000)
        monkeypatch.setattr(samples, "MAX_CLEAN_COMPRESSOR_GB", 10_000)
        assert machine_is_uncontended() is True

    def test_a_broken_lock_module_claims_nothing(self, monkeypatch):
        """Cannot tell is not the same as contended: tagging every sample on a
        machine without the lock module would stop all recording."""

        def boom():
            raise RuntimeError("no lock module here")

        monkeypatch.setattr("lib.gpu_lock.foreign_holder", boom)
        assert gpu_is_contended() is False

    @pytest.mark.parametrize(
        "swap,compressor,expected",
        [(0.0, 0.0, True), (20.0, 0.0, False), (0.0, 20.0, False)],
    )
    def test_the_pressure_gate_still_applies_when_the_gpu_is_free(
        self, monkeypatch, swap, compressor, expected
    ):
        """Adding the GPU check must not have replaced the swap/compressor one."""
        monkeypatch.setattr("lib.gpu_lock.foreign_holder", lambda: None)

        class _Swap:
            used = swap * 1024**3

        monkeypatch.setattr("psutil.swap_memory", lambda: _Swap())
        pages = int(compressor * 1024**3 / 16384)
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **k: type(
                "R", (), {"stdout": f"Pages occupied by compressor:      {pages}."}
            )(),
        )
        assert machine_is_uncontended() is expected
