"""Structural gates pinning outcomes that would otherwise depend on the
developer's own machine: memory pressure and real wall-clock waits.

Split out of conftest.py for the 500-line cap (no test exemption; see
CLAUDE.md). Imported by name into conftest.py so pytest's fixture discovery
finds them there.
"""

import pytest


@pytest.fixture(autouse=True)
def deterministic_machine_contention():
    """Structural gate: no test outcome may depend on the developer's memory pressure.

    `add_sample` tags every sample by calling `machine_is_uncontended()`, which
    reads live swap and compressor figures. That was harmless while the flag only
    steered a median, and stopped being harmless the moment `_derived_timeout`
    began consulting it: the timeout tests in test_prefill_measurement.py record
    rates and assert on the derived timeout, so they PASSED on a quiet machine and
    FAILED on a busy one -- three of them went red purely because the compressor
    was at 18GB while the suite ran.

    Pinned clean here. Tests that exercise contention patch `psutil`/`vm_stat`
    themselves (test_self_correcting_samples.py) or pass `clean=` to `add_sample`
    explicitly, both of which still work: those import the function by value or
    bypass it entirely.
    """
    from unittest.mock import patch

    # The memory readings are pinned for the same reason and by the same rule.
    # `oversize_refusal` consults `is_thrashing()` and `reclaimable_available_gb()`
    # when its callers do not inject them, so without this a test asserting that
    # a 70GB model is refused would pass on a 64GB laptop and fail on a 128GB
    # one -- a test outcome decided by the developer's hardware.
    #
    # 64GB is this project's reference machine. Tests that exercise the pressure
    # or headroom branches inject `thrashing=`/`available_gb=` explicitly, which
    # is why those parameters exist.
    with (
        patch("eval.samples.machine_is_uncontended", return_value=True),
        patch("eval.memory.is_thrashing", return_value=False),
        patch("eval.memory.reclaimable_available_gb", return_value=64.0),
    ):
        yield


@pytest.fixture(autouse=True)
def bounded_restart_ready_budget(request):
    """Structural gate: no test may sit through the real 180s readiness wait.

    `wait_until_model_serves` polls until the model answers or
    RESTART_READY_BUDGET (180s) runs out, sleeping RESTART_READY_GAP between
    attempts. Tests patch `time.sleep` so the poll does not actually wait -- but
    the DEADLINE is real wall-clock time from `time.monotonic()`, which nothing
    patches. Removing the sleep does not shorten the loop; it turns a 180-second
    wait into a 180-second BUSY-SPIN at ~95% CPU.

    Measured 2026-08-23: test_it_restarts_normally_when_the_gpu_is_free took
    189s of user CPU by itself. With several such tests the suite looked hung
    rather than slow, and the pre-push gate could never finish -- which is why
    ztools could not be pushed at all.

    The budget is deliberately resolved at CALL time (see the comment in
    wait_until_model_serves), precisely so it can be patched here. A test that
    genuinely needs the real budget can ask for it with
    @pytest.mark.real_restart_budget.
    """
    import eval.cli_runtime as rt

    if request.node.get_closest_marker("real_restart_budget"):
        yield
        return

    original = rt.RESTART_READY_BUDGET
    rt.RESTART_READY_BUDGET = 0  # the poll loop body never runs
    try:
        yield
    finally:
        rt.RESTART_READY_BUDGET = original
