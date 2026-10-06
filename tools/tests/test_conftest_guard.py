"""The guard that guards the suite is itself tested.

`tools/tests/conftest.py` exists because a pytest marker claimed a guard that did
not exist. A guard nobody has watched fail is the same defect one level up, so
each piece `no_real_server_restart` is assembled from is proved here, in BOTH
directions -- and the one piece that turns out to be NECESSARY is proved
necessary, so a later edit cannot replace it with something weaker believing the
two equivalent.

Both directions matters on its own: a tripwire that refused every invocation, or
an interposer that redirected every `mkdir`, would pass a test that only checks
the harmful path.
"""

import re
import shutil
import subprocess
from pathlib import Path

from conftest import (
    DEFAULT_REAL_GPU_LOCK,
    MACHINE_REACHING_COMMANDS,
    REFUSAL_EXIT,
    _fingerprint,
    _write_mkdir_interposer,
    _write_refuser,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Stands in for the machine-wide lock in the interposer tests. NOT the real
#: /tmp/mac-osaurus-gpu.lock: those tests assert that a path did not come into
#: existence, and a test cannot own an assertion about the machine -- the first
#: version of this file asserted the real path, so it failed for a reason (a stray
#: lock left by an earlier run) that had nothing to do with the code under test.


def test_the_guarded_path_is_the_lock_gpu_lock_shell_falls_back_to(tmp_path):
    """The guard only protects the right path if it names the same one the lock
    does. Read from the source rather than restated here, so changing
    `GPU_LOCK_DIR`'s default turns this red instead of quietly leaving the guard
    watching a path nothing uses."""
    source = (Path(REPO_ROOT) / "tools" / "gpu_lock.sh").read_text()
    match = re.search(r'GPU_LOCK_DIR="\$\{ZTOOLS_GPU_LOCK_DIR:-([^}]+)\}"', source)
    assert match, "tools/gpu_lock.sh no longer sets GPU_LOCK_DIR the way it did"
    assert DEFAULT_REAL_GPU_LOCK == Path(match.group(1))


def _run(argv, cwd):
    return subprocess.run(argv, capture_output=True, text=True, cwd=cwd, check=False)


class TestTheTripwires:
    def test_each_machine_command_refuses_and_records(self, tmp_path):
        """The refusal is loud AND leaves evidence: one that only exited non-zero
        would be indistinguishable from a normal result."""
        for command in MACHINE_REACHING_COMMANDS:
            log = tmp_path / f"{command}.log"
            _write_refuser(tmp_path, command, log)
            r = _run([str(tmp_path / command), "--and-an-argument"], tmp_path)
            assert r.returncode == REFUSAL_EXIT, f"{command} did not refuse"
            assert "REFUSING" in r.stderr
            assert f"{command} --and-an-argument" in log.read_text()

    def test_it_records_nothing_until_it_runs(self, tmp_path):
        """A tripwire that recorded an invocation nobody made would fail every
        marked test for a reason with no cause."""
        log = tmp_path / "idle.log"
        _write_refuser(tmp_path, "pgrep", log)
        assert not log.exists()


class TestTheInterposer:
    def _install(self, tmp_path):
        """A stand-in 'machine' lock path under tmp_path, and the interposer."""
        shadow, log = tmp_path / "bin", tmp_path / "lock.log"
        shadow.mkdir()
        machine_lock = tmp_path / "stand-in-for-the-machine-gpu.lock"
        redirect = tmp_path / "redirected.lock"
        _write_mkdir_interposer(shadow, log, machine_lock, redirect)
        return shadow / "mkdir", log, machine_lock, redirect

    def test_it_redirects_the_machine_lock_and_leaves_it_uncreated(self, tmp_path):
        mkdir, log, machine_lock, redirect = self._install(tmp_path)

        r = _run([str(mkdir), str(machine_lock)], tmp_path)

        assert r.returncode == 0, r.stderr
        assert redirect.is_dir(), "the redirected lock was not created"
        assert not machine_lock.exists(), "the machine lock path came into existence"
        assert str(machine_lock) in log.read_text(), "the substitution was not recorded"

    def test_it_forwards_every_other_mkdir_untouched(self, tmp_path):
        """The half that proves the interposer is not merely an obstacle course: a
        test creating its own sandbox directories must be unaffected."""
        mkdir, log, _, _ = self._install(tmp_path)

        r = _run([str(mkdir), str(tmp_path / "gpu.lock")], tmp_path)

        assert r.returncode == 0, r.stderr
        assert (tmp_path / "gpu.lock").is_dir()
        assert not log.exists(), "a harmless mkdir was recorded as a machine reach"

    def test_it_matches_the_target_after_the_flags(self, tmp_path):
        """mkdir's target is its LAST argument, however the flags are spelled. An
        interposer matching argv[1] would pass the two tests above and forward
        `mkdir -p <machine lock>`."""
        mkdir, log, machine_lock, redirect = self._install(tmp_path)

        r = _run([str(mkdir), "-p", str(machine_lock)], tmp_path)

        assert r.returncode == 0, r.stderr
        assert redirect.is_dir()
        assert not machine_lock.exists()
        assert str(machine_lock) in log.read_text()


def test_a_fingerprint_alone_is_blind_to_take_and_release(tmp_path):
    """WHY THE INTERPOSER IS LOAD-BEARING, kept as an assertion so it cannot be
    deleted as belt-and-braces.

    `gpu_lock.sh` acquires the lock on entry and releases it on its EXIT trap, so a
    test that took the machine-wide lock and let the script run to completion
    leaves it byte-identical -- and the before/after comparison in the fixture
    reports "untouched". That is not a theory: an unsandboxed calibration test
    really did take the real lock, and the comparison said nothing had happened.
    """
    lock = tmp_path / "blind.lock"
    before = _fingerprint(lock)
    assert before == (False, None, None)

    lock.mkdir()
    (lock / "owner").write_text("123\nMon Jan  1 00:00:00 2020\ncalibration\n")
    held = _fingerprint(lock)
    assert held != before, "the fingerprint cannot see the lock at all"

    shutil.rmtree(lock)
    assert _fingerprint(lock) == before, (
        "take-and-release leaves the fingerprint unchanged, so the mkdir "
        "interposer is the only thing closing that hole"
    )
