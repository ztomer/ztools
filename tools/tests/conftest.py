"""Guards for the pytest suite over the shell tooling.

WHY THIS FILE EXISTS. `tools/osaurus_one.sh` exists to stop, start and SIGKILL a
4-35GB osaurus server: `stop_all` reads PIDs out of `pgrep` and `lsof` and then
kills them (osaurus_one.sh:145). A test that drives that script without stubbing
those commands does not merely measure the wrong thing -- it kills the server
another agent session is mid-measurement against, and that run's numbers get
filed as CLEAN, because the sample guard reads swap and compressor and never the
GPU. Several sessions run on this machine concurrently; that is the entire reason
the GPU lock exists.

`tools/tests/test_gpu_lock_shell.py` declares
`pytestmark = pytest.mark.sandboxed_server_script`, and until this file existed
that marker opted out of NOTHING: there was no conftest.py in the repo, so every
run printed `PytestUnknownMarkWarning` for it and the comment claiming a guard was
enforcing something was fiction. The tests WERE sandboxed -- by PATH-stubbing --
but nothing checked that they were.

So the guard is real now, and the marker is what makes it checkable instead of
assumed. A test carrying `sandboxed_server_script` DECLARES "this drives a script
that mutates the real server", and in exchange this fixture proves it.

WHAT IS PROVEN, and by what mechanism:

  * The machine is unreachable. `osaurus`, `pgrep`, `lsof` (the three
    `osaurus_one.sh:96` requires) and `curl` (its liveness probe) are shadowed
    ahead of the real ones on PATH by a tripwire that records the invocation and
    exits 97. A test that forgot its own stub therefore cannot reach the real
    server, AND the log is asserted EMPTY at teardown, so forgetting is a failure
    rather than a quietly machine-dependent result.

  * The machine-wide lock cannot come into existence. `gpu_lock.sh:260` takes the
    lock with `mkdir "$GPU_LOCK_DIR"` -- an ordinary PATH-resolved command, not a
    shell builtin -- so a `mkdir` interposer redirects that one path into this
    test's tmp_path and forwards every other mkdir. This is a seam, not a
    sampler: the wrong state is unrepresentable while a marked test runs, with no
    race and nothing to tune. (`ZTOOLS_GPU_LOCK_DIR` is also pointed at tmp_path
    by default, so a test that forgets to redirect it lands in tmp anyway. Only
    the mkdir moves; the owner write, the heartbeat and the release still name
    the real path, so such a test fails on its next write rather than quietly
    working.)

  * The lock is not otherwise disturbed. Its existence, the owner file's bytes and
    the directory's mtime are compared across the test. That catches the changes
    the interposer does not cover (a direct owner write, an `rm -rf`), and it is
    the one check that can report a FALSE positive: a peer eval that starts or
    finishes while a marked test runs moves the same bytes. The message says so.

RESIDUAL, stated rather than papered over. A read of the real lock changes none of
the three fields above, and a read is not the damage this guard prevents -- but it
does mean `--check` run against the real path is invisible here. `ZTOOLS_GPU_LOCK_DIR`
is the seam that prevents that, and the marked tests use it.

CALIBRATION, both directions. A deliberately unsandboxed test goes RED on this
fixture's teardown, and the sandboxed suite goes green. The real lock path is
injected as a fixture (`real_gpu_lock`) precisely so calibration can be proven
against a stand-in rather than against the path a live eval may be holding -- an
unchecked guard is a guard nobody has watched fail.

SCOPE, and why it is the marker rather than every test. `sweep_models.sh` and
`rerun_truncated.sh` also shell out to real `pgrep`/`kill`, and their tests
legitimately exercise those scripts; forcing the sandbox on every test in the
package would change the behaviour under test for scripts that are not the danger
this guard is about. A test that drives a server-mutating script therefore has to
say so with the marker -- which is the seam the original comment claimed and did
not have.
"""

import os
from pathlib import Path

import pytest

#: The machine-wide GPU lock `tools/gpu_lock.sh:114` falls back to when
#: `ZTOOLS_GPU_LOCK_DIR` is unset. A fixture, not a bare constant, so a test that
#: needs to watch the guard fire can substitute a stand-in path.
DEFAULT_REAL_GPU_LOCK = Path("/tmp/mac-osaurus-gpu.lock")

#: The commands through which the real machine is reachable from a test that
#: drives `osaurus_one.sh`, and which the test must therefore stub ITSELF:
#: `osaurus` starts a server, `pgrep` and `lsof` are what hand `stop_all` real
#: PIDs to SIGKILL, and `curl` is the liveness probe that decides whether the
#: script believes a server is up. Reached unstubbed, they measure -- or kill --
#: whatever the machine happens to be doing at that moment.
MACHINE_REACHING_COMMANDS = ("osaurus", "pgrep", "lsof", "curl")

#: Exit status of every refusal here. Distinct from any status the scripts under
#: test produce, so a swallowed non-zero cannot read as a normal result.
REFUSAL_EXIT = 97


@pytest.fixture
def real_gpu_lock():
    """The path whose being untouched this guard proves. Overridable for calibration."""
    return DEFAULT_REAL_GPU_LOCK


def _refusal_message(text):
    return f"REFUSING: a sandboxed_server_script test reached {text}. "


def _write_refuser(directory, name, log):
    """Shadow one real command with a stub that records the call and refuses it."""
    path = directory / name
    path.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "{name} $*" >> "{log}"\n'
        f'printf "%s\\n" "{_refusal_message(f"the REAL {name}")}" >&2\n'
        f"exit {REFUSAL_EXIT}\n"
    )
    path.chmod(0o755)
    return path


def _write_mkdir_interposer(directory, log, real_lock, redirect_to):
    """Point `mkdir` of the machine-wide lock somewhere harmless, forward the rest.

    This is the load-bearing piece. `gpu_lock.sh` takes the lock with a plain
    `mkdir` (line 260), which PATH resolves -- so intercepting the one path makes
    the real lock un-creatable while a marked test runs, deterministically. A
    comparison of the path before and after the test cannot do that: the script
    acquires on entry and releases on its EXIT trap, so a test that took the real
    lock and let the script finish left it byte-identical and the comparison
    reported "untouched". That blind spot was found by running the calibration,
    not by reasoning about it.

    REDIRECTED rather than refused, and the distinction is load-bearing too.
    `gpu_lock_acquire` loops `while ! mkdir ...` and reaches its timeout only if
    the holder looks ALIVE; a permanently-refused mkdir with no owner file reads
    as a stale lock forever, so a refusal would spin in that `while` loop until
    someone killed the run -- a hang instead of a failure.

    Only the mkdir is redirected, and that is enough. Everything downstream --
    `> "$GPU_LOCK_DIR/owner"` (gpu_lock.sh:281), the heartbeat `touch`, the
    release `rm -rf` -- still names the real path, so the script fails on its very
    next write with `No such file or directory` and stops. A test that aimed at
    the real lock therefore gets a loud failure and an untouched machine, which
    is the outcome; it does NOT get a working redirected lock, and nothing here
    pretends otherwise.
    """
    path = directory / "mkdir"
    path.write_text(
        "#!/usr/bin/env bash\n"
        "# The target of a mkdir is its LAST argument, whatever the flags are.\n"
        'for _ztools_target in "$@"; do :; done\n'
        f'if [ "$_ztools_target" = "{real_lock}" ]; then\n'
        # The doubled braces below are Python escaping; the script gets ${...}.
        f'  printf "%s\\n" "{real_lock}" >> "{log}"\n'
        f'  printf "%s\\n" "{_refusal_message(f"the machine-wide GPU lock at {real_lock}")}'
        f' -> {redirect_to}" >&2\n'
        f'  set -- "${{@:1:$#-1}}" "{redirect_to}"\n'
        "fi\n"
        'exec /bin/mkdir "$@"\n'
    )
    path.chmod(0o755)
    return path


def _fingerprint(path):
    """Every field of the real lock that touching it changes.

    Existence, the owner file's bytes, and the directory's mtime (which moves when
    a lock is taken, reclaimed or released).
    """
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return (False, None, None)
    try:
        owner = (path / "owner").read_bytes()
    except OSError:
        owner = None
    return (True, owner, mtime)


@pytest.fixture(autouse=True)
def no_real_server_restart(request, tmp_path, monkeypatch, real_gpu_lock):
    """Prove a `sandboxed_server_script` test really is sandboxed, or fail it.

    Autouse, because the failure it prevents is not a wrong number in one test --
    it is a killed server on a machine several sessions share. Scoped to the
    marker for the reason in the module docstring.
    """
    if "sandboxed_server_script" not in request.keywords:
        yield
        return

    shadow = tmp_path / "the-machine-is-not-reachable-from-here"
    shadow.mkdir()
    commands_log = tmp_path / "reached-the-real-machine.log"
    lock_log = tmp_path / "redirected-the-machine-gpu-lock.log"
    for command in MACHINE_REACHING_COMMANDS:
        _write_refuser(shadow, command, commands_log)
    _write_mkdir_interposer(shadow, lock_log, real_gpu_lock, tmp_path / "redirected-gpu.lock")

    before = _fingerprint(real_gpu_lock)

    # The seam, made the default: a test that forgets to redirect the lock lands
    # in tmp_path rather than on the machine-wide lock. gpu_lock.sh reads this
    # variable once, at source time (line 114).
    monkeypatch.setenv("PATH", os.pathsep.join([str(shadow), os.environ["PATH"]]))
    monkeypatch.setenv("ZTOOLS_GPU_LOCK_DIR", str(tmp_path / "gpu.lock"))
    # A stray `nohup osaurus serve > $LOG` must not land on the shared TMPDIR.
    monkeypatch.setenv("OSAURUS_LOG", str(tmp_path / "osaurus.log"))

    yield

    failures = []
    if commands_log.exists():
        failures.append(
            f"{request.node.name} reached the real machine:\n"
            f"{commands_log.read_text().strip()}\n"
            f"Stub every one of {', '.join(MACHINE_REACHING_COMMANDS)} on PATH in "
            "the test itself -- see the stubbed_tools fixture."
        )
    if lock_log.exists():
        failures.append(
            f"{request.node.name} asked for the machine-wide GPU lock at "
            f"{real_gpu_lock} (redirected, so nothing was taken):\n"
            f"{lock_log.read_text().strip()}\n"
            "Point ZTOOLS_GPU_LOCK_DIR at tmp_path in the test itself."
        )
    after = _fingerprint(real_gpu_lock)
    if after != before:
        failures.append(
            f"{request.node.name} changed the machine-wide GPU lock at "
            f"{real_gpu_lock}: {before} -> {after}. Either the test did it (point "
            "ZTOOLS_GPU_LOCK_DIR at tmp_path), or a peer eval started or ended "
            "while this one ran -- re-run the suite when no eval holds the GPU."
        )
    assert not failures, "\n\n".join(failures)
