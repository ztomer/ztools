"""Machine-wide mutual exclusion for the osaurus server and the GPU — Python half.

THE SAME LOCK AS tools/gpu_lock.sh, not a parallel one: same path, same owner-file
format, same staleness rules, same start-time normalisation. That is the whole
point. `tools/osaurus_one.sh` restarts the server from bash and `eval.cli` measures
against it from Python, so if the two halves disagreed about the path or about what
"stale" means they would run at the same time while both believing they held the
lock. Change one half, change the other.

DELIBERATELY NOT THE DESKTOP LOCK. `~/projects/scripts/lib/desktop_lock.sh` guards
the one physical desktop at /tmp/mac-desktop-ui.lock. The GPU is a different
resource: a screenshot run and an eval have no reason to exclude each other, and
sharing a name would make every capture a false "GPU busy". Same design, own path.

WHAT COUNTS AS TAKING THE GPU -- take this lock if you do any of:
  - start, stop, restart or quit the osaurus app or `osaurus serve`
  - measure a model (prefill rate, decode rate, task latency, memory)
  - load model weights, by any route, on a machine sized for one at a time

WHY IT IS A HARD LOCK. Several agent sessions run concurrently on this Mac. Every
model here is 4-35GB resident, and a second server does not queue behind the
first -- it loads its OWN copy, which means eviction, swapping, and requests the
server cancels itself with HTTP 499 request_cancelled. From the client that is
indistinguishable from a slow model. A stray second server once recorded
qwen3.8-27b at 0.1 tok/s decode with a 423s cold start.

AND THE MACHINE'S OWN CONTENTION GUARD CANNOT SEE THIS. `eval/samples.py` keeps a
list of samples and estimates from the median of the last `SAMPLE_WINDOW` = 5 CLEAN
ones, which does outvote a bad reading -- for the pressure it can measure. But
`machine_is_uncontended()` gates on SWAP and COMPRESSOR only, read from psutil and
vm_stat. A peer session saturating the GPU moves neither, so its interference is
tagged CLEAN and enters the median as though the box were quiet. The median cannot
outvote a reading it believes is good.

And the median only protects a model that HAS history: a new or thinly-sampled
model's estimate IS its one sample, so first measurements are the exposed case.
This lock is precisely what the swap/compressor guard is blind to.

WHY os.mkdir AND NOT fcntl.flock. An flock dies with the file descriptor, which
makes "is the holder still alive" invisible to a peer that wants to reclaim rather
than block forever. os.mkdir is atomic on every POSIX filesystem and leaves an
inspectable owner record behind, which is what makes the defences below possible.
It also matches the bash half exactly.

DEADLOCK DEFENCES, because an agent session dies in more ways than it exits.
Four independent releases:
  1. THE CONTEXT MANAGER / finally, on any clean exit or exception.
  2. OWNER LIVENESS, for SIGKILL and crashes, which run no cleanup at all.
     PID ALONE IS NOT ENOUGH -- PIDs get recycled, and a recycled PID reads as
     "alive" forever, a permanent deadlock wearing the costume of a busy peer.
     The owner's START TIME is recorded next to the PID and must match too.
  3. AN IDLE CEILING, for an owner that is alive but WEDGED.
  4. INHERITANCE, so a wrapper holding the lock can run a child that also asks
     for it without the child blocking on its own parent forever.

THE CEILING MEASURES PROGRESS, NOT DURATION. An honest desktop run is under two
minutes, so the desktop lock can treat 900s of wall clock as proof of a wedge. An
honest eval is hours -- the sweep's per-model ceiling is 4h and rerun_truncated
raises it to 10h. A wall-clock ceiling short enough to catch a wedge would reclaim
the lock out from under a healthy 6-hour measurement and hand it to a peer that
then restarts the server, causing exactly the corruption this module prevents. So
the holder calls `heartbeat()` after each unit of work and the ceiling runs from
the last beat: a run still making progress never expires, a run that has stopped
expires whether or not its process is alive.

The bias is therefore the opposite of the desktop lock's. There, reclaiming
wrongly costs one confused screenshot. Here it corrupts a tracked capability
record nobody will notice is wrong, so a waiter that cannot get the lock raises
and names the holder rather than waiting long enough to be tempted into stealing.

Usage:
    from lib.gpu_lock import gpu_lock
    with gpu_lock("eval qwen3.8-27b"):
        ...
"""

import contextlib
import os
import subprocess
import time

__all__ = [
    "DEFAULT_LOCK_DIR",
    "GpuBusy",
    "acquire",
    "foreign_holder",
    "gpu_lock",
    "heartbeat",
    "holder",
    "lock_dir",
    "release",
]

# The machine-wide default, identical to GPU_LOCK_DIR in tools/gpu_lock.sh. /tmp
# and not tempfile.gettempdir(): the contended resource is the one GPU in this
# Mac, and TMPDIR is per-user on macOS and per-SESSION under some agent
# harnesses -- which would hand each caller a private lock and no exclusion at all.
DEFAULT_LOCK_DIR = "/tmp/mac-osaurus-gpu.lock"

# Short on purpose. A peer holding this lock is usually mid-eval and will hold it
# for hours, so waiting is theatre: the useful answer is "another session owns the
# GPU, here is which one". Callers that genuinely want to queue raise it.
DEFAULT_TIMEOUT = 60

# Time since the last heartbeat, not since acquisition -- see the module header.
# The longest legitimate gap between beats is one task, and the per-task timeout
# derived from measured rates is ~7055s, so 4h leaves roughly 2x headroom.
DEFAULT_MAX_IDLE = 14400

# Names the PID that acquired the lock, so a child process can tell "my own
# session holds this" from "a stranger holds this" without inspecting ancestry.
OWNER_ENV = "ZTOOLS_GPU_LOCK_OWNER"

# Test seam. Production never sets it; a test asserts DEFAULT_LOCK_DIR is the
# machine-wide path, so the seam cannot quietly become the norm. It exists because
# a lock is only testable if a test can drive BOTH directions -- exclusion and
# reclamation -- and doing that against the real path would fight whatever
# concurrent session is holding it for real.
DIR_ENV = "ZTOOLS_GPU_LOCK_DIR"

_held = False


class GpuBusy(RuntimeError):
    """The GPU stayed held by a live, un-wedged peer past the timeout."""


def lock_dir() -> str:
    """The lock path, resolved per call so a test override actually takes."""
    return os.environ.get(DIR_ENV) or DEFAULT_LOCK_DIR


def _start_time(pid) -> str:
    """A process's start time, used to tell the real owner from a recycled PID.

    Empty when the process does not exist, which callers treat as dead.

    THE NORMALISATION IS PART OF THE CROSS-LANGUAGE CONTRACT and must match
    tools/gpu_lock.sh byte for byte, or each half reads the other's records as
    impostors and silently grants a lock the peer is holding. The desktop lock's
    untested first version did exactly that: it used `" ".join(out.split(" "))`,
    which looks like a whitespace squeeze and is a no-op, while the bash half
    really was squeezing -- and `ps` pads single-digit days ("Aug  1" vs
    "Aug 18"), so the two disagreed on two days in three. split() with no
    argument collapses every whitespace run and trims the ends; the bash half
    gets there via word splitting and "$*". A test compares both on a live PID.
    """
    try:
        out = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    return " ".join(out.split())


def _owner():
    """(pid, start_time, label) from the owner file, or None if unreadable.

    Unreadable counts as stale: it means a run died between the mkdir and the
    write, leaving a lock that nobody alive can ever release.
    """
    try:
        with open(os.path.join(lock_dir(), "owner")) as handle:
            lines = handle.read().split("\n")
    except OSError:
        return None
    if len(lines) < 3 or not lines[0].strip():
        return None
    return lines[0].strip(), lines[1], lines[2]


def _owner_alive() -> bool:
    owner = _owner()
    if owner is None:
        return False
    pid, recorded, _ = owner
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    # PID is live -- but is it the SAME process, or a recycled number?
    current = _start_time(pid)
    if recorded and current and recorded.strip() != current.strip():
        return False
    return True


def _expired(max_idle) -> bool:
    """Has the holder stopped making progress since its last heartbeat?"""
    try:
        beat = os.stat(lock_dir()).st_mtime
    except OSError:
        return False
    return (time.time() - beat) >= max_idle


def _label(owner) -> str:
    """Never the empty string. A holder that reports as "" reads downstream as
    "nobody holds it" -- `if foreign_holder():` is exactly how the quit call
    sites ask -- so an owner file whose label line is blank or missing would
    grant every caller permission to quit a live peer's server. The bash half
    degrades the same way, and must: the two read each other's files."""
    return owner[2].strip() if owner and owner[2].strip() else "an unknown run"


def holder():
    """Label of whoever holds the GPU, ours included; None when it is free."""
    return _label(_owner()) if _owner_alive() else None


def foreign_holder():
    """Label of a LIVE holder that is neither us nor an ancestor holding for us.

    None when the GPU is ours or free. This is the predicate the osascript quit
    call sites consult. They do not take the lock -- they sit deep inside
    unrelated tools (a twitter summariser, a weekend planner) whose job is not to
    queue behind an eval -- they simply must not quit a server another session is
    measuring against.
    """
    if not _owner_alive():
        return None
    owner = _owner()
    pid = owner[0]
    if pid == str(os.getpid()) or pid == os.environ.get(OWNER_ENV):
        return None
    return _label(owner)


def acquire(label="gpu run", timeout=DEFAULT_TIMEOUT,
            max_idle=DEFAULT_MAX_IDLE, log=print) -> bool:
    """Take the lock, reclaiming a dead or wedged owner's.

    Returns True if this process now owns it, False if an ancestor already held
    it on our behalf (adopted, so `release()` here must not free the parent's).
    Raises GpuBusy rather than proceeding unlocked -- proceeding is the corrupting
    case this module exists to stop.
    """
    global _held
    inherited = os.environ.get(OWNER_ENV)
    if inherited and _owner_alive() and _owner()[0] == inherited:
        log(f"→ gpu already held by {_label(_owner())} (inherited)")
        return False
    waited = 0
    announced = False
    while True:
        try:
            os.mkdir(lock_dir())
            break
        except FileExistsError:
            if not _owner_alive():
                log(f"→ stale gpu lock from {_label(_owner())} — reclaiming")
                _force_remove()
                continue
            if _expired(max_idle):
                log(f"→ gpu held with no progress for {max_idle}s by "
                    f"{_label(_owner())} — wedged, reclaiming")
                _force_remove()
                continue
            if not announced:
                log(f"→ the gpu is held by {_label(_owner())}; waiting up to {timeout}s")
                announced = True
            if waited >= timeout:
                raise GpuBusy(
                    f"gpu still held by {_label(_owner())} after {timeout}s — that "
                    f"session is measuring; do not restart osaurus under it")
            time.sleep(1)
            waited += 1
    pid = os.getpid()
    with open(os.path.join(lock_dir(), "owner"), "w") as handle:
        handle.write(f"{pid}\n{_start_time(pid)}\n{label} (pid {pid})\n")
    _held = True
    os.environ[OWNER_ENV] = str(pid)
    if announced:
        log(f"→ gpu acquired after {waited}s")
    return True


def heartbeat() -> None:
    """Say "still working", so the idle ceiling does not reclaim a healthy run.

    A no-op unless we actually hold the lock: an unheld caller must not be able
    to keep a peer's expired lock alive, which would defeat defence #3 entirely.
    """
    if not _held:
        return
    with contextlib.suppress(OSError):
        os.utime(lock_dir(), None)


def _force_remove() -> None:
    with contextlib.suppress(OSError):
        os.remove(os.path.join(lock_dir(), "owner"))
    with contextlib.suppress(OSError):
        os.rmdir(lock_dir())


def release() -> None:
    """Release a lock THIS process holds.

    Checking ownership matters twice over: without it a run that gave up waiting
    would delete the lock of the peer it waited for, and a child that inherited
    the lock would free its parent's.
    """
    global _held
    if not _held:
        return
    _held = False
    os.environ.pop(OWNER_ENV, None)
    owner = _owner()
    if owner is not None and owner[0] != str(os.getpid()):
        return
    _force_remove()


@contextlib.contextmanager
def gpu_lock(label="gpu run", timeout=DEFAULT_TIMEOUT,
             max_idle=DEFAULT_MAX_IDLE, log=print):
    acquire(label, timeout=timeout, max_idle=max_idle, log=log)
    try:
        yield
    finally:
        release()
