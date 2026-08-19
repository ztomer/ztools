"""The GPU/osaurus lock: exclusion, reclamation, and the two quit call sites.

WHAT IS ACTUALLY BEING PROTECTED. Several agent sessions run on this Mac at once.
Any of them restarting osaurus mid-measurement evicts the running model, and the
peer sees HTTP 499 request_cancelled and a rate an order of magnitude low --
indistinguishable, from the client, from a genuinely slow model. And the machine's
own guard cannot catch it: `eval/samples.py` estimates from the median of the last
5 CLEAN samples, but `machine_is_uncontended()` gates on swap and compressor only
and is blind to the GPU, so a peer's interference is tagged CLEAN and enters the
median as though the box were quiet. For a new or thinly-sampled model, that one
sample IS the estimate.

Every test here drives the lock at `lib.gpu_lock.DIR_ENV`, which conftest's
`_gpu_lock_never_touches_the_real_one` points at a per-test tmp dir. Running these
against /tmp/mac-osaurus-gpu.lock would block whatever real eval is holding it.

The interesting cases are the RECLAIM paths, because each is a way the lock can
deadlock every eval on the machine, and a lock that never reclaims looks
identical to a lock that is working right up until the day it wedges.
"""

import os
import subprocess
import time
from unittest.mock import patch

import pytest
from lib import gpu_lock


def _own(tmp, pid, start=None, label="a peer run"):
    """Write an owner file by hand, as a dead or foreign holder would leave one."""
    d = gpu_lock.lock_dir()
    os.makedirs(d, exist_ok=True)
    start = gpu_lock._start_time(pid) if start is None else start
    with open(os.path.join(d, "owner"), "w") as fh:
        fh.write(f"{pid}\n{start}\n{label}\n")


def _dead_pid():
    """A PID that is certainly not running: spawn one and reap it."""
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


class TestThePathIsTheContract:
    """A lock only excludes the peers that agree on its name."""

    def test_the_default_path_is_machine_wide_and_not_the_desktop_lock(self):
        """/tmp, not $TMPDIR: TMPDIR is per-user on macOS and per-SESSION under
        some agent harnesses, which hands each caller a private lock and no
        exclusion at all. And a DIFFERENT path from the desktop lock, because the
        desktop and the GPU are different resources -- sharing a name would make
        every screenshot run a false 'GPU busy'."""
        assert gpu_lock.DEFAULT_LOCK_DIR == "/tmp/mac-osaurus-gpu.lock"
        assert gpu_lock.DEFAULT_LOCK_DIR != "/tmp/mac-desktop-ui.lock"

    def test_the_shell_half_agrees_on_the_path(self):
        """The two halves are ONE lock. `osaurus_one.sh` restarts the server from
        bash while `eval.cli` measures from Python; if they disagreed about the
        path they would both run while both believing they held it."""
        out = subprocess.run(
            ["bash", "-c",
             'source tools/gpu_lock.sh >/dev/null 2>&1; printf "%s" "$GPU_LOCK_DIR"'],
            capture_output=True, text=True,
            env={**os.environ, "ZTOOLS_GPU_LOCK_DIR": ""},
        ).stdout
        assert out == gpu_lock.DEFAULT_LOCK_DIR

    def test_the_override_is_in_force_for_this_suite(self):
        """Proves the conftest redirect actually took. Without this, every test
        below could be silently exercising the real machine-wide lock and
        passing for the wrong reason."""
        assert gpu_lock.lock_dir() != gpu_lock.DEFAULT_LOCK_DIR
        assert gpu_lock.DIR_ENV in os.environ


class TestExclusion:
    def test_a_free_gpu_is_acquired(self):
        assert gpu_lock.acquire("mine", log=lambda _: None) is True
        assert gpu_lock.holder().startswith("mine")

    def test_a_live_peer_is_not_displaced(self):
        """os.getppid() is a real, live process that is not us -- the shape of a
        peer agent session mid-eval."""
        _own(None, os.getppid(), label="peer eval")
        with pytest.raises(gpu_lock.GpuBusy) as excinfo:
            gpu_lock.acquire("mine", timeout=0, log=lambda _: None)
        assert "peer eval" in str(excinfo.value)

    def test_the_refusal_names_the_holder(self):
        """A bare 'GPU busy' cannot be acted on. Naming the holder is what turns
        it into 'that session is measuring; wait or ask it'."""
        _own(None, os.getppid(), label="eval qwen3.8-27b (pid 123)")
        with pytest.raises(gpu_lock.GpuBusy) as excinfo:
            gpu_lock.acquire("mine", timeout=0, log=lambda _: None)
        assert "qwen3.8-27b" in str(excinfo.value)

    def test_release_frees_it_for_the_next_run(self):
        gpu_lock.acquire("first", log=lambda _: None)
        gpu_lock.release()
        assert gpu_lock.holder() is None
        assert gpu_lock.acquire("second", log=lambda _: None) is True

    def test_the_context_manager_releases_through_an_exception(self):
        """Release #1. An eval that crashes must not wedge every later eval."""
        with pytest.raises(ValueError):
            with gpu_lock.gpu_lock("boom", log=lambda _: None):
                raise ValueError("task blew up")
        assert gpu_lock.holder() is None


class TestReclaimingADeadOwner:
    """Release #2: SIGKILL and crashes run no cleanup at all."""

    def test_a_dead_owners_lock_is_reclaimed(self):
        _own(None, _dead_pid(), start="Mon Jan 1 00:00:00 2020", label="killed run")
        assert gpu_lock.acquire("mine", timeout=0, log=lambda _: None) is True

    def test_a_recycled_pid_cannot_impersonate_the_owner(self):
        """PID ALONE IS NOT ENOUGH. PIDs are recycled, and a recycled PID reads as
        'alive' forever -- a permanent deadlock wearing the costume of a busy
        peer. The recorded START TIME is what tells a live impostor from the
        owner: this owner file names a live PID with the wrong start time."""
        _own(None, os.getpid(), start="Mon Jan 1 00:00:00 2020", label="ghost")
        assert gpu_lock._owner_alive() is False
        assert gpu_lock.acquire("mine", timeout=0, log=lambda _: None) is True

    def test_a_lock_with_no_owner_file_is_stale(self):
        """A run that died between the mkdir and the write leaves a lock nobody
        alive can ever release."""
        os.makedirs(gpu_lock.lock_dir(), exist_ok=True)
        assert gpu_lock.acquire("mine", timeout=0, log=lambda _: None) is True

    def test_a_truncated_owner_file_is_stale(self):
        os.makedirs(gpu_lock.lock_dir(), exist_ok=True)
        with open(os.path.join(gpu_lock.lock_dir(), "owner"), "w") as fh:
            fh.write("12345\n")
        assert gpu_lock._owner() is None
        assert gpu_lock.acquire("mine", timeout=0, log=lambda _: None) is True

    def test_a_non_numeric_pid_is_stale(self):
        _own(None, "not-a-pid", start="x", label="corrupt")
        assert gpu_lock._owner_alive() is False

    def test_start_time_of_a_dead_process_is_empty(self):
        assert gpu_lock._start_time(_dead_pid()) == ""

    def test_start_time_degrades_rather_than_raising_when_ps_is_unavailable(self):
        """`ps` failing must not take the caller down: an unreadable start time
        means 'treat as dead', which reclaims, and reclaiming is recoverable."""
        with patch.object(gpu_lock.subprocess, "run", side_effect=OSError("no ps")):
            assert gpu_lock._start_time(os.getpid()) == ""


class TestTheIdleCeiling:
    """Release #3: an owner that is alive but has stopped making progress.

    The ceiling runs from the last HEARTBEAT, not from acquisition, because an
    honest eval holds the GPU for hours -- 4h per model in the sweep, 10h in
    rerun_truncated. A wall-clock ceiling tight enough to catch a hang would
    reclaim the lock from a healthy run and hand it to a peer that then restarts
    the server, causing the very corruption the lock prevents.
    """

    def test_a_live_but_silent_owner_is_reclaimed(self):
        _own(None, os.getppid(), label="wedged peer")
        old = time.time() - 100
        os.utime(gpu_lock.lock_dir(), (old, old))
        assert gpu_lock.acquire("mine", max_idle=10, timeout=0, log=lambda _: None) is True

    def test_a_heartbeat_keeps_a_healthy_long_run_alive(self):
        """The distinction that matters: same elapsed time, still holding, because
        it is still finishing tasks."""
        gpu_lock.acquire("long eval", log=lambda _: None)
        old = time.time() - 100
        os.utime(gpu_lock.lock_dir(), (old, old))
        assert gpu_lock._expired(10) is True
        gpu_lock.heartbeat()
        assert gpu_lock._expired(10) is False

    def test_a_heartbeat_from_a_non_holder_cannot_prop_up_a_peers_lock(self):
        """Otherwise any passing process could keep a wedged owner's lock alive
        forever, which disables release #3 entirely."""
        _own(None, os.getppid(), label="wedged peer")
        old = time.time() - 100
        os.utime(gpu_lock.lock_dir(), (old, old))
        gpu_lock.heartbeat()
        assert gpu_lock._expired(10) is True

    def test_expiry_of_a_missing_lock_is_not_an_error(self):
        assert gpu_lock._expired(10) is False


class TestReleaseOnlyFreesWhatWeOwn:
    def test_a_process_that_never_acquired_releases_nothing(self):
        """Without the guard, a run that gave up waiting would delete the lock of
        the peer it waited for -- turning a polite failure into a break-in."""
        _own(None, os.getppid(), label="peer eval")
        gpu_lock.release()
        assert gpu_lock.holder() == "peer eval"

    def test_release_is_idempotent(self):
        """Three release paths overlap by design (cleanup callback, atexit, the
        context manager), so a second release must be free."""
        gpu_lock.acquire("mine", log=lambda _: None)
        gpu_lock.release()
        gpu_lock.release()
        assert gpu_lock.holder() is None

    def test_a_stolen_lock_is_not_deleted_by_the_previous_owner(self):
        """We held it, the ceiling handed it to a peer, and now our EXIT trap
        fires. Deleting on the way out would free the new owner's lock."""
        gpu_lock.acquire("mine", log=lambda _: None)
        _own(None, os.getppid(), label="new owner")
        gpu_lock.release()
        assert gpu_lock.holder() == "new owner"


class TestInheritance:
    """Release #4: a wrapper that holds the lock must be able to run a child."""

    def test_a_child_of_the_holder_adopts_rather_than_blocking(self):
        gpu_lock.acquire("parent run", log=lambda _: None)
        gpu_lock._held = False  # a child process starts with no hold of its own
        assert gpu_lock.acquire("child run", timeout=0, log=lambda _: None) is False

    def test_an_adopting_child_does_not_free_its_parents_lock(self):
        gpu_lock.acquire("parent run", log=lambda _: None)
        gpu_lock._held = False
        gpu_lock.acquire("child run", timeout=0, log=lambda _: None)
        gpu_lock.release()
        assert gpu_lock.holder().startswith("parent run")

    def test_a_stale_owner_env_var_does_not_grant_a_strangers_lock(self):
        """The env var is a hint, not proof. It only counts when it matches the
        CURRENT owner file -- otherwise a variable left over from a released lock
        would let any process walk past a live peer."""
        os.environ[gpu_lock.OWNER_ENV] = "999999"
        _own(None, os.getppid(), label="peer eval")
        with pytest.raises(gpu_lock.GpuBusy):
            gpu_lock.acquire("mine", timeout=0, log=lambda _: None)


class TestForeignHolder:
    """The predicate the osascript quit call sites consult."""

    def test_a_free_gpu_has_no_foreign_holder(self):
        assert gpu_lock.foreign_holder() is None

    def test_our_own_hold_is_not_foreign(self):
        """An eval that holds the GPU is entitled to restart the server it owns;
        if its own hold read as foreign it could never flush between models."""
        gpu_lock.acquire("my eval", log=lambda _: None)
        assert gpu_lock.foreign_holder() is None

    def test_a_live_peer_is_foreign(self):
        _own(None, os.getppid(), label="peer eval")
        assert gpu_lock.foreign_holder() == "peer eval"

    def test_a_dead_peer_is_not_foreign(self):
        """A crashed session must not block the machine's other tools forever."""
        _own(None, _dead_pid(), start="Mon Jan 1 00:00:00 2020", label="dead peer")
        assert gpu_lock.foreign_holder() is None

    def test_an_inherited_hold_is_not_foreign(self):
        _own(None, os.getppid(), label="parent run")
        os.environ[gpu_lock.OWNER_ENV] = str(os.getppid())
        assert gpu_lock.foreign_holder() is None

    def test_an_unlabelled_owner_still_reports_as_a_holder(self):
        """Degrade with a stated reason rather than reading as 'nobody'."""
        d = gpu_lock.lock_dir()
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "owner"), "w") as fh:
            fh.write(f"{os.getppid()}\n{gpu_lock._start_time(os.getppid())}\n\n")
        assert gpu_lock.holder() == "an unknown run"


class TestTheWaitLoop:
    def test_it_waits_and_then_takes_a_freed_lock(self):
        """The announce-and-wait branch, and the acquisition that follows it."""
        _own(None, os.getppid(), label="peer eval")
        messages = []
        real_sleep = gpu_lock.time.sleep

        def freeing_sleep(_seconds):
            gpu_lock._force_remove()
            real_sleep(0)

        with patch.object(gpu_lock.time, "sleep", freeing_sleep):
            assert gpu_lock.acquire(
                "mine", timeout=5, log=messages.append) is True
        assert any("held by peer eval" in m for m in messages)
        assert any("acquired after" in m for m in messages)


class TestCrossLanguageStartTimeParity:
    """The normalisation is a cross-language contract, and it has been broken
    before -- in the desktop lock, whose first Python half used
    `" ".join(out.split(" "))`, a no-op that looks like a whitespace squeeze,
    while the bash half really squeezed. `ps` pads single-digit days ("Aug  1"
    vs "Aug 18"), so the two disagreed on two days in three, and each read the
    other's records as impostors -- silently granting a lock a peer was holding.
    """

    def test_both_halves_produce_the_same_string_for_the_same_process(self):
        pid = os.getpid()
        shell = subprocess.run(
            ["bash", "-c",
             f'source tools/gpu_lock.sh >/dev/null 2>&1; _gpu_lock_start_time {pid}'],
            capture_output=True, text=True,
        ).stdout
        assert shell == gpu_lock._start_time(pid)
        assert shell != ""

    def test_both_halves_squeeze_padded_day_fields_identically(self):
        """The exact input that broke the desktop lock. Asserted against the
        collapsing behaviour rather than against `ps`, so it fails on any day of
        the month rather than only on single-digit ones."""
        padded = "Sat Aug  1 09:04:11 2026"
        shell = subprocess.run(
            ["bash", "-c",
             'source tools/gpu_lock.sh >/dev/null 2>&1; '
             f'raw="{padded}"; set -- $raw; printf "%s" "$*"'],
            capture_output=True, text=True,
        ).stdout
        assert shell == " ".join(padded.split())
        assert "  " not in shell
