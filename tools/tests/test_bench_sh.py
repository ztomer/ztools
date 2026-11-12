"""tools/bench.sh reaps its `osaurus bench` child BEFORE it releases the GPU lock.

ZT.TOOLFIX (2026-11-12). The scratch version of this script held the lock with an EXIT trap
and ran the bench in the foreground. A SIGTERM released the lock while the child kept
running outside it: a gemma-4-12b bench overlapped an embedding run, and both measured the
contention instead of the model. The lock exists so that no model run happens outside it, so
a child that outlives the release is the exact state the lock is there to forbid.

The observation is made from OUTSIDE, in both directions: the instant the lock directory
disappears the child (and its own child) must already be dead, and the same must hold once
bench.sh has exited. A fake `osaurus` stands in for the real one; nothing here loads a model.
"""

import os
import signal
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
BENCH = REPO / "tools" / "bench.sh"
WAIT = 20


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie reads as alive to kill(0); ps knows better.
    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        check=False,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    ).stdout.strip()
    return bool(state) and not state.startswith("Z")


def _wait_for(path, what):
    end = time.time() + WAIT
    while time.time() < end:
        if path.exists() and path.read_text().strip():
            return
        time.sleep(0.02)
    raise AssertionError(f"{what} never appeared")


def _fake_osaurus(bindir, pidfile, grandfile, mode):
    """`osaurus bench` that records its pid, spawns a grandchild, and sleeps.

    mode "ignore" also ignores SIGTERM, to prove the escalation to SIGKILL.
    """
    bindir.mkdir(exist_ok=True)
    trap = "trap '' TERM\n" if mode == "ignore" else ""
    script = bindir / "osaurus"
    script.write_text(
        "#!/bin/bash\n"
        f"{trap}"
        f'echo $$ > "{pidfile}"\n'
        f"( {'trap "" TERM; ' if mode == 'ignore' else ''}exec sleep 300 ) &\n"
        f'echo $! > "{grandfile}"\n'
        "wait\n"
    )
    script.chmod(0o755)


def _run_and_term(tmp_path, mode):
    lock = tmp_path / "gpu.lock"
    pidfile, grandfile = tmp_path / "child.pid", tmp_path / "grand.pid"
    _fake_osaurus(tmp_path / "bin", pidfile, grandfile, mode)
    env = {
        **os.environ,
        "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
        "ZTOOLS_GPU_LOCK_DIR": str(lock),
        "BENCH_KILL_GRACE": "1",
        "NO_COLOR": "1",
    }
    proc = subprocess.Popen(
        ["bash", str(BENCH), "--model", "fake", "--out", str(tmp_path / "out")],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for(pidfile, "the bench child")
        _wait_for(grandfile, "the bench grandchild")
        assert lock.exists(), "bench.sh runs its child without holding the GPU lock"
        child, grand = int(pidfile.read_text()), int(grandfile.read_text())
        proc.send_signal(signal.SIGTERM)
        # Watch the lock from outside: the moment it is gone, nothing may still be running.
        seen_alive_after_release = None
        end = time.time() + WAIT
        while time.time() < end:
            if not lock.exists():
                seen_alive_after_release = [p for p in (child, grand) if _alive(p)]
                break
            time.sleep(0.002)
        proc.wait(timeout=WAIT)
        return lock, child, grand, seen_alive_after_release
    finally:
        if proc.poll() is None:
            proc.kill()
        for name in (pidfile, grandfile):
            if name.exists() and name.read_text().strip():
                try:
                    os.kill(int(name.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass


def test_sigterm_reaps_the_child_before_the_lock_is_released(tmp_path):
    lock, child, grand, alive_at_release = _run_and_term(tmp_path, "plain")
    assert alive_at_release == [], f"the lock was released while {alive_at_release} still ran"
    assert not lock.exists()
    assert not _alive(child) and not _alive(grand)


def test_a_child_that_ignores_sigterm_is_killed_before_release(tmp_path):
    lock, child, grand, alive_at_release = _run_and_term(tmp_path, "ignore")
    assert alive_at_release == [], f"the lock was released while {alive_at_release} still ran"
    assert not _alive(child) and not _alive(grand)


def test_a_clean_run_records_each_model_and_releases_the_lock(tmp_path):
    lock = tmp_path / "gpu.lock"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "osaurus").write_text(
        '#!/bin/bash\nwhile [ $# -gt 0 ]; do [ "$1" = --json ] && echo "{}" > "$2"; shift; done\n'
        'echo "args ok"\n'
    )
    (bindir / "osaurus").chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "ZTOOLS_GPU_LOCK_DIR": str(lock),
        "NO_COLOR": "1",
    }
    r = subprocess.run(
        ["bash", str(BENCH), "--model", "a", "--model", "b", "--out", str(tmp_path / "out")],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "out" / "bench.a.json").exists() and (
        tmp_path / "out" / "bench.b.json"
    ).exists()
    assert not lock.exists()


def test_a_failing_bench_is_a_failing_run(tmp_path):
    lock = tmp_path / "gpu.lock"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "osaurus").write_text("#!/bin/bash\nexit 7\n")
    (bindir / "osaurus").chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "ZTOOLS_GPU_LOCK_DIR": str(lock),
        "NO_COLOR": "1",
    }
    r = subprocess.run(
        ["bash", str(BENCH), "--model", "a", "--out", str(tmp_path / "out")],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
    )
    assert r.returncode != 0
    assert not lock.exists()
