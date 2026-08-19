"""The shell half of the GPU lock, and its wiring into tools/osaurus_one.sh.

WHY A SECOND TEST FILE. tools/gpu_lock.sh is a SEPARATE IMPLEMENTATION of the same
lock as lib/gpu_lock.py, not a wrapper around it -- osaurus_one.sh is bash and
cannot import Python, and shelling out to Python from a lock would add a failure
mode to the thing whose whole job is to fail safely. Two implementations of one
protocol means the Python tests prove nothing about the bash half, and a repo that
grows a second language inherits none of the first one's gates. So every defence
is re-proved here against the shell.

The stakes are the same ones the Python tests state: a peer session restarting
osaurus mid-measurement produces a rate an order of magnitude low, and the sample
guard files it as CLEAN because it reads swap and compressor and not the GPU.

Every test drives an isolated lock path under tmp_path. Running these against the
real /tmp/mac-osaurus-gpu.lock would block whatever eval is holding it.
"""

import os
import stat
import subprocess
import time
from pathlib import Path

import pytest

#: Drives the real tools/osaurus_one.sh against a stubbed PATH and a tmp lock.
#: Opts out of conftest's no_real_server_restart gate, which matches on the script
#: name and cannot tell a sandboxed invocation from a live one.
pytestmark = pytest.mark.sandboxed_server_script

REPO = Path(__file__).resolve().parent.parent.parent


def sh(body, lock_dir, extra_env=None, cwd=None):
    """Run a bash snippet with tui/lib.sh and tools/gpu_lock.sh sourced."""
    env = {**os.environ, "ZTOOLS_GPU_LOCK_DIR": str(lock_dir), "NO_COLOR": "1"}
    env.pop("ZTOOLS_GPU_LOCK_OWNER", None)
    env.update(extra_env or {})
    script = (
        f'source "{REPO}/tui/lib.sh"\n'
        f'source "{REPO}/tools/gpu_lock.sh"\n'
        f"{body}\n"
    )
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True,
        env=env, cwd=str(cwd or REPO),
    )


def write_owner(lock_dir, pid, start, label):
    """Plant an owner file, as a dead or foreign holder would leave one."""
    Path(lock_dir).mkdir(parents=True, exist_ok=True)
    Path(lock_dir, "owner").write_text(f"{pid}\n{start}\n{label}\n")


def live_start_time(pid):
    out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)],
                         capture_output=True, text=True).stdout
    return " ".join(out.split())


def dead_pid():
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


class TestTheShellHalfExcludes:
    def test_it_acquires_a_free_lock(self, tmp_path):
        lock = tmp_path / "gpu.lock"
        r = sh(f'gpu_lock_acquire "mine"; [ -f "{lock}/owner" ] && echo HELD', lock)
        assert "HELD" in r.stdout

    def test_a_live_peer_is_not_displaced(self, tmp_path):
        """os.getppid() is a real live process that is not the bash child -- the
        shape of a peer agent session mid-eval."""
        lock = tmp_path / "gpu.lock"
        peer = os.getppid()
        write_owner(lock, peer, live_start_time(peer), "peer eval")
        r = sh('gpu_lock_acquire "mine"; echo REACHED', lock,
               {"GPU_LOCK_TIMEOUT": "0"})
        assert r.returncode != 0
        assert "REACHED" not in r.stdout

    def test_the_refusal_names_the_holder(self, tmp_path):
        lock = tmp_path / "gpu.lock"
        peer = os.getppid()
        write_owner(lock, peer, live_start_time(peer), "eval qwen3.8-27b (pid 9)")
        r = sh('gpu_lock_acquire "mine"', lock, {"GPU_LOCK_TIMEOUT": "0"})
        assert "qwen3.8-27b" in r.stdout + r.stderr

    def test_release_frees_it(self, tmp_path):
        lock = tmp_path / "gpu.lock"
        sh('gpu_lock_acquire "mine"; gpu_lock_release', lock)
        assert not lock.exists()

    def test_the_trap_releases_on_a_signal(self, tmp_path):
        """Release #1. A killed run must not wedge every eval on the machine."""
        lock = tmp_path / "gpu.lock"
        sh('trap "gpu_lock_release" EXIT INT TERM\n'
           'gpu_lock_acquire "mine"\n'
           'kill -TERM $$', lock)
        assert not lock.exists()


class TestTheShellHalfReclaims:
    def test_a_dead_owner_is_reclaimed(self, tmp_path):
        """Release #2: SIGKILL and crashes run no trap at all."""
        lock = tmp_path / "gpu.lock"
        write_owner(lock, dead_pid(), "Mon Jan 1 00:00:00 2020", "killed run")
        r = sh('gpu_lock_acquire "mine" && echo ACQUIRED', lock,
               {"GPU_LOCK_TIMEOUT": "0"})
        assert "ACQUIRED" in r.stdout

    def test_a_recycled_pid_cannot_impersonate_the_owner(self, tmp_path):
        """PID ALONE IS NOT ENOUGH. A recycled PID reads as alive forever, which
        is a permanent deadlock wearing the costume of a busy peer. The recorded
        start time is what separates the two: this file names a LIVE pid with the
        wrong start time."""
        lock = tmp_path / "gpu.lock"
        write_owner(lock, os.getppid(), "Mon Jan 1 00:00:00 2020", "ghost")
        r = sh('gpu_lock_acquire "mine" && echo ACQUIRED', lock,
               {"GPU_LOCK_TIMEOUT": "0"})
        assert "ACQUIRED" in r.stdout

    def test_a_lock_with_no_owner_file_is_stale(self, tmp_path):
        """A run that died between the mkdir and the write leaves a lock nobody
        alive can ever release."""
        lock = tmp_path / "gpu.lock"
        lock.mkdir()
        r = sh('gpu_lock_acquire "mine" && echo ACQUIRED', lock,
               {"GPU_LOCK_TIMEOUT": "0"})
        assert "ACQUIRED" in r.stdout

    def test_a_live_but_silent_owner_is_reclaimed(self, tmp_path):
        """Release #3, measured from the last heartbeat rather than acquisition."""
        lock = tmp_path / "gpu.lock"
        peer = os.getppid()
        write_owner(lock, peer, live_start_time(peer), "wedged peer")
        old = time.time() - 100
        os.utime(lock, (old, old))
        r = sh('gpu_lock_acquire "mine" && echo ACQUIRED', lock,
               {"GPU_LOCK_TIMEOUT": "0", "GPU_LOCK_MAX_IDLE": "10"})
        assert "ACQUIRED" in r.stdout

    def test_a_heartbeat_keeps_a_healthy_long_run_alive(self, tmp_path):
        """The distinction the ceiling exists to make: same elapsed wall clock,
        but still holding, because it is still finishing tasks. Without this an
        honest 6-hour eval loses its lock to a peer that then restarts osaurus."""
        lock = tmp_path / "gpu.lock"
        r = sh('gpu_lock_acquire "mine"\n'
               f'touch -t 202001010000 "{lock}"\n'
               '_gpu_lock_expired && echo EXPIRED_BEFORE\n'
               'gpu_lock_heartbeat\n'
               '_gpu_lock_expired || echo ALIVE_AFTER', lock,
               {"GPU_LOCK_MAX_IDLE": "10"})
        assert "EXPIRED_BEFORE" in r.stdout
        assert "ALIVE_AFTER" in r.stdout

    def test_release_does_not_free_a_peers_lock(self, tmp_path):
        """Without the guard, a run that gave up waiting would delete the lock of
        the peer it waited for -- a polite failure turned into a break-in."""
        lock = tmp_path / "gpu.lock"
        peer = os.getppid()
        write_owner(lock, peer, live_start_time(peer), "peer eval")
        sh("gpu_lock_release", lock)
        assert (lock / "owner").exists()


class TestTheShellHalfReportsForeignHolders:
    def test_a_free_gpu_reports_nobody(self, tmp_path):
        r = sh('printf "[%s]" "$(gpu_lock_foreign_holder)"', tmp_path / "gpu.lock")
        assert r.stdout.strip() == "[]"

    def test_a_live_peer_is_foreign(self, tmp_path):
        lock = tmp_path / "gpu.lock"
        peer = os.getppid()
        write_owner(lock, peer, live_start_time(peer), "peer eval")
        r = sh("gpu_lock_foreign_holder", lock)
        assert "peer eval" in r.stdout

    def test_our_own_hold_is_not_foreign(self, tmp_path):
        lock = tmp_path / "gpu.lock"
        r = sh('gpu_lock_acquire "mine" >/dev/null\n'
               'printf "[%s]" "$(gpu_lock_foreign_holder)"', lock)
        assert r.stdout.strip().endswith("[]")

    def test_an_inherited_hold_is_not_foreign(self, tmp_path):
        """Release #4: a wrapper holding the lock must be able to run a child."""
        lock = tmp_path / "gpu.lock"
        peer = os.getppid()
        write_owner(lock, peer, live_start_time(peer), "parent run")
        r = sh('printf "[%s]" "$(gpu_lock_foreign_holder)"', lock,
               {"ZTOOLS_GPU_LOCK_OWNER": str(peer)})
        assert r.stdout.strip() == "[]"

    def test_an_unlabelled_owner_still_reports_as_a_holder(self, tmp_path):
        """Reporting "" would read downstream as "nobody holds it" -- which is
        exactly how the quit call sites ask -- and grant permission to quit a
        live peer's server."""
        lock = tmp_path / "gpu.lock"
        peer = os.getppid()
        write_owner(lock, peer, live_start_time(peer), "")
        r = sh("gpu_lock_holder", lock)
        assert "an unknown run" in r.stdout


class TestCrossLanguageParity:
    """One lock, two implementations. If they disagree about any of this, each
    reads the other's records as an impostor and silently grants a lock the peer
    is holding -- while both print reassuring "acquired" messages."""

    def test_both_halves_default_to_the_same_path(self, tmp_path):
        from lib import gpu_lock

        r = sh('printf "%s" "$GPU_LOCK_DIR"', "", {"ZTOOLS_GPU_LOCK_DIR": ""})
        assert r.stdout == gpu_lock.DEFAULT_LOCK_DIR

    def test_the_shell_half_can_read_a_python_owner_file(self, tmp_path):
        """The direction that matters most in practice: an eval (Python) holds
        the GPU and osaurus_one.sh (bash) must see it."""
        from lib import gpu_lock

        lock = tmp_path / "gpu.lock"
        os.environ[gpu_lock.DIR_ENV] = str(lock)
        gpu_lock.acquire("eval from python", log=lambda _: None)
        try:
            r = sh("gpu_lock_holder", lock)
        finally:
            gpu_lock.release()
        assert "eval from python" in r.stdout

    def test_python_can_read_a_shell_owner_file(self, tmp_path):
        from lib import gpu_lock

        lock = tmp_path / "gpu.lock"
        # A live holder written by bash: keep the bash process alive while Python
        # reads it, or the liveness check would correctly call it dead.
        proc = subprocess.Popen(
            ["bash", "-c",
             f'source "{REPO}/tui/lib.sh"; source "{REPO}/tools/gpu_lock.sh"; '
             'gpu_lock_acquire "osaurus_one.sh --restart" >/dev/null; '
             "read -r _"],
            stdin=subprocess.PIPE, text=True,
            env={**os.environ, "ZTOOLS_GPU_LOCK_DIR": str(lock)},
        )
        try:
            deadline = time.time() + 10
            while not (lock / "owner").exists() and time.time() < deadline:
                time.sleep(0.05)
            os.environ[gpu_lock.DIR_ENV] = str(lock)
            assert gpu_lock.holder() == "osaurus_one.sh --restart (pid %d)" % proc.pid
            assert gpu_lock.foreign_holder() == gpu_lock.holder()
        finally:
            proc.communicate("\n", timeout=10)


@pytest.fixture
def stubbed_tools(tmp_path):
    """A PATH where osaurus/lsof/pgrep/curl are scripted, not real.

    osaurus_one.sh's whole job is to start and stop a 4-35GB server. It cannot be
    exercised for real in a test suite, and stubbing the four commands it shells
    out to is what makes the LOCK wiring testable in both directions -- refuses
    under a peer, and proceeds when free -- on one machine.
    """
    binroot = tmp_path / "bin"
    binroot.mkdir()

    def stub(name, body):
        path = binroot / name
        path.write_text(f"#!/usr/bin/env bash\n{body}\n")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)

    # A REAL, LIVE, DISPOSABLE process stands in for the server, because
    # osaurus_one.sh does not merely read this PID -- `stop_all` SIGTERMs and then
    # SIGKILLs it. An earlier version of this fixture reported os.getpid(), and the
    # first mutation run that removed the lock from the script reached `stop_all`
    # and killed the pytest process itself (exit 143). A test fixture must not hand
    # the code under test a weapon aimed at the test runner.
    victim = subprocess.Popen(["sleep", "600"])
    stub("pgrep", f'echo {victim.pid}')
    stub("lsof", f'echo {victim.pid}')
    stub("curl", "exit 0")
    stub("osaurus", "exit 0")
    try:
        yield binroot
    finally:
        victim.kill()
        victim.wait()


def run_osaurus_one(args, lock_dir, stubs, extra_env=None):
    env = {
        **os.environ,
        "PATH": f"{stubs}:{os.environ['PATH']}",
        "ZTOOLS_GPU_LOCK_DIR": str(lock_dir),
        "NO_COLOR": "1",
    }
    env.pop("ZTOOLS_GPU_LOCK_OWNER", None)
    env.update(extra_env or {})
    return subprocess.run(
        ["bash", str(REPO / "tools" / "osaurus_one.sh"), *args],
        capture_output=True, text=True, env=env, cwd=str(REPO),
    )


class TestOsaurusOneCheckReportsOwnership:
    def test_a_single_free_server_passes(self, tmp_path, stubbed_tools):
        r = run_osaurus_one(["--check"], tmp_path / "gpu.lock", stubbed_tools)
        assert r.returncode == 0, r.stdout + r.stderr

    def test_a_single_server_held_by_a_peer_fails(self, tmp_path, stubbed_tools):
        """--check answers 'is it safe to measure now'. A server that is single,
        healthy and BUSY produces numbers as contaminated as a doubled one, so
        reporting it green would be answering a different question."""
        lock = tmp_path / "gpu.lock"
        peer = os.getppid()
        write_owner(lock, peer, live_start_time(peer), "peer eval")
        r = run_osaurus_one(["--check"], lock, stubbed_tools)
        assert r.returncode == 1
        assert "peer eval" in r.stdout + r.stderr

    def test_check_does_not_take_the_lock(self, tmp_path, stubbed_tools):
        """A read-only diagnostic that acquired would block on, or steal from,
        the very run it is reporting on."""
        lock = tmp_path / "gpu.lock"
        run_osaurus_one(["--check"], lock, stubbed_tools)
        assert not lock.exists()


class TestOsaurusOneMutationTakesTheLock:
    def test_it_releases_the_lock_on_the_way_out(self, tmp_path, stubbed_tools):
        lock = tmp_path / "gpu.lock"
        r = run_osaurus_one([], lock, stubbed_tools)
        assert r.returncode == 0, r.stdout + r.stderr
        assert not lock.exists(), "the lock outlived the run that took it"

    def test_it_refuses_to_touch_a_server_a_peer_is_measuring_on(
            self, tmp_path, stubbed_tools):
        """The headline case. Restarting the single healthy server another
        session is mid-measurement against corrupts that run's numbers just as
        thoroughly as starting a second server would."""
        lock = tmp_path / "gpu.lock"
        peer = os.getppid()
        write_owner(lock, peer, live_start_time(peer), "peer eval")
        r = run_osaurus_one(["--restart"], lock, stubbed_tools,
                            {"GPU_LOCK_TIMEOUT": "0"})
        assert r.returncode != 0
        assert "peer eval" in r.stdout + r.stderr
        assert "restart requested" not in r.stdout
        assert (lock / "owner").read_text().endswith("peer eval\n")

    def test_a_dead_peers_lock_does_not_block_a_restart(self, tmp_path, stubbed_tools):
        """Reclamation has to work end to end, not just in the library: a crashed
        session must not wedge every later run on the machine."""
        lock = tmp_path / "gpu.lock"
        write_owner(lock, dead_pid(), "Mon Jan 1 00:00:00 2020", "killed run")
        r = run_osaurus_one([], lock, stubbed_tools, {"GPU_LOCK_TIMEOUT": "0"})
        assert r.returncode == 0, r.stdout + r.stderr
        assert "reclaiming" in r.stdout + r.stderr
