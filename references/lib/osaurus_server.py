#!/usr/bin/env python3
import os
import signal
import subprocess
import time
from pathlib import Path

from . import gpu_lock
from .logging_config import osaurus_logger as logger
from .osaurus_models import DEFAULT_HOST, DEFAULT_PORT, get_models, is_server_running
from .tui import die, err

_pid_dir = Path(os.environ.get("XDG_RUNTIME_DIR", str(Path.home())))
PID_FILE = _pid_dir / ".osaurus.pid"

# Configuration constants to prevent magic strings and numbers (Mitchell Hashimoto design)
RESTART_SLEEP = int(os.environ.get("OSAURUS_RESTART_SLEEP", "1"))
SERVER_WAIT = int(os.environ.get("OSAURUS_SERVER_WAIT", "20"))
ENSURE_MAX_RETRIES = int(os.environ.get("OSAURUS_MAX_RETRIES", "3"))
TEST_TIMEOUT = int(os.environ.get("OSAURUS_TEST_TIMEOUT", "10"))
OSASCRIPT_QUIT_TIMEOUT = int(os.environ.get("OSAURUS_QUIT_TIMEOUT", "5"))
OSASCRIPT_QUIT_CMD = 'quit app "osaurus"'
DEFAULT_APP_PATH = os.environ.get("OSAURUS_APP", "/Applications/osaurus.app")
GUI_LAUNCHER_CMD = "open"
CLI_LAUNCHER_CMD = os.environ.get("OSAURUS_CLI_CMD", "osaurus serve --yes").split()
POLL_SLEEP_SEC = 1
TEST_PROMPT_CONTENT = "Hi"
DUMP_DIR_NAME = "llm_dumps"
PREVIEW_LIMIT = 100
STATUS_ERROR = "error"
STATUS_OK = "ok"

# Subprocess & command constants
PS_CMD = "ps"
PS_PID_FLAG = "-p"
PS_ARGS_FLAG = "-o"
PS_ARGS_FORMAT = "args="
PROCESS_CHECK_TIMEOUT = 2
OSASCRIPT_QUIT_CMD_LOWER = "osaurus"
OSASCRIPT_CMD = "osascript"
OSASCRIPT_FLAG = "-e"
FAST_POLL_INTERVAL = 0.1
# Long enough for a cold MLX model load (~13s measured on this
# machine), short enough that a wedged server fails fast instead of
# hanging a scheduled run. See can_serve().
SERVE_PROBE_TIMEOUT = int(os.environ.get("OSAURUS_SERVE_PROBE_TIMEOUT", "45"))
# How long to wait for a quit to actually free the port before relaunching.
SHUTDOWN_WAIT = int(os.environ.get("OSAURUS_SHUTDOWN_WAIT", "15"))
# Grace period before a failed restart is allowed to quit osaurus again.
RETRY_GRACE = int(os.environ.get("OSAURUS_RETRY_GRACE", "20"))


def _is_osaurus_process(pid: int) -> bool:
    """Verify that a given PID belongs to an active process named or running osaurus."""
    try:
        res = subprocess.run(
            [PS_CMD, PS_PID_FLAG, str(pid), PS_ARGS_FLAG, PS_ARGS_FORMAT],
            capture_output=True,
            text=True,
            timeout=PROCESS_CHECK_TIMEOUT,
        )
        if res.returncode == 0 and res.stdout:
            return OSASCRIPT_QUIT_CMD_LOWER in res.stdout.lower()
    except Exception:
        pass
    return False


def _kill_osaurus() -> bool:
    """Quit osaurus. Returns False, having done nothing, if it is not ours to quit.

    THE SERVER IS A MACHINE-WIDE SINGLETON AND THIS FUNCTION IS NOT THE ONLY
    CALLER. It runs from the twitter summariser, the weekend planner and
    `check_server_or_die` -- ordinary tools whose reaction to a wedged-looking
    server is to quit it and start a fresh one. That is correct on an idle
    machine and destructive on this one: several agent sessions run here at once,
    and quitting the server a peer is measuring against evicts its model
    mid-measurement. The peer sees HTTP 499 request_cancelled and a decode rate
    an order of magnitude low, which is indistinguishable from a genuinely slow
    model -- and `eval/samples.py` records it as a CLEAN sample, because
    `machine_is_uncontended()` gates on swap and compressor and cannot see the
    GPU at all. It then enters that model's median as though the box were quiet.

    So the quit REFUSES rather than queues. Queueing would be wrong here: these
    callers are trying to recover a server for their own request, not to reserve
    the GPU, and blocking a tweet summary for the hours an eval runs helps
    nobody. Refusing degrades with a stated reason and leaves the peer's
    measurement intact -- the honest failure. Our own lock does not count; an
    eval that holds the GPU is entitled to restart the server it owns.
    """
    blocked_by = gpu_lock.foreign_holder()
    if blocked_by:
        logger.warning(
            "refusing to quit osaurus: %s holds the GPU lock and is measuring "
            "against this server. Quitting it would evict that run's model "
            "mid-measurement, and the ruined timing would be filed as a CLEAN "
            "sample -- the contention guard reads swap and compressor, not the GPU.",
            blocked_by,
        )
        return False

    try:
        subprocess.run(
            [OSASCRIPT_CMD, OSASCRIPT_FLAG, OSASCRIPT_QUIT_CMD],
            capture_output=True,
            timeout=OSASCRIPT_QUIT_TIMEOUT,
        )
    except Exception as e:
        logger.debug(f"Failed to quit osaurus app via osascript: {e}")

    if PID_FILE.exists():
        pid = None
        try:
            pid = int(PID_FILE.read_text().strip())
            if _is_osaurus_process(pid):
                logger.info(f"Terminating osaurus process with PID {pid} via SIGTERM")
                os.kill(pid, signal.SIGTERM)
                time.sleep(RESTART_SLEEP)
            else:
                logger.warning(
                    f"PID file contains process ID {pid} which is not osaurus. "
                    f"Skipping termination."
                )
        except (ProcessLookupError, ValueError, OSError) as e:
            logger.debug(f"Failed to kill process {pid}: {e}")
        PID_FILE.unlink(missing_ok=True)
    return True


def _osaurus_process_exists() -> bool:
    """Is any osaurus process still alive?

    The HTTP port frees BEFORE the process finishes exiting, so waiting only on
    the port is not enough: macOS LaunchServices still considers the app running
    and swallows the relaunch, re-activating the terminating instance instead of
    starting a new one. That is why the relaunch appeared to do nothing.
    """
    try:
        res = subprocess.run(
            ["pgrep", "-x", "osaurus"],
            capture_output=True,
            text=True,
            timeout=PROCESS_CHECK_TIMEOUT,
        )
        return res.returncode == 0 and bool(res.stdout.strip())
    except Exception:
        return False


def _wait_until_down(timeout: int = SHUTDOWN_WAIT) -> bool:
    """Block until the server stops answering, or `timeout` elapses.

    The osascript quit is ASYNCHRONOUS. The predecessor slept a flat 1s and then
    relaunched, so `open` could attach to the still-quitting instance, which then
    exited -- leaving NOTHING listening on 1337 while `restart_server` reported
    failure. Observed for real: an `ensure_server()` call turned a recoverable
    wedge into a hard outage. Waiting for the port to actually free removes the
    race instead of widening the sleep and hoping.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not is_server_running() and not _osaurus_process_exists():
            return True
        time.sleep(FAST_POLL_INTERVAL)
    return not is_server_running() and not _osaurus_process_exists()


def restart_server(app_path: str = DEFAULT_APP_PATH, wait: int = SERVER_WAIT) -> bool:
    # A refused quit must not fall through to the relaunch. Launching while a peer
    # still holds a live server is how a SECOND osaurus appears, and two of them on
    # a machine sized for one is the original contamination: each loads its own
    # copy of the weights, both swap, and both sets of numbers are ruined.
    if not _kill_osaurus():
        return False
    if not _wait_until_down():
        logger.warning(
            "osaurus still answering after quit; relaunching anyway may race with shutdown"
        )
    time.sleep(RESTART_SLEEP)

    is_gui = Path(app_path).exists()
    if is_gui:
        launcher = [GUI_LAUNCHER_CMD, app_path]
    else:
        launcher = CLI_LAUNCHER_CMD

    def _launch() -> bool:
        try:
            proc = subprocess.Popen(
                launcher, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            if not is_gui:
                PID_FILE.write_text(str(proc.pid))
            return True
        except Exception as exc:
            logger.error(f"Failed to restart osaurus server: {exc}")
            return False

    if not _launch():
        return False

    # Poll up to wait seconds (Carmack/Hashimoto fast-poll latency optimization)
    max_polls = max(1, int(wait / FAST_POLL_INTERVAL))
    for _ in range(max_polls):
        time.sleep(FAST_POLL_INTERVAL)
        if is_server_running():
            return True

    # One more LAUNCH -- never another quit. A relaunch issued while the old
    # instance was still terminating gets swallowed by LaunchServices, and the
    # cure for that is to ask again, not to kill something that is starting.
    logger.warning("osaurus did not come up; issuing one more launch (no quit)")
    if _launch():
        for _ in range(max_polls):
            time.sleep(FAST_POLL_INTERVAL)
            if is_server_running():
                return True
    logger.error(
        "osaurus did not answer within %ss of relaunch; nothing may be listening on "
        "the port. Do NOT retry blindly -- a second quit would kill an instance that "
        "is still starting.", wait
    )
    return False


def can_serve(model: str, timeout: int = SERVE_PROBE_TIMEOUT) -> tuple[bool, str]:
    """Can `model` actually produce a token? Returns (ok, reason).

    `is_server_running()` only asks `/v1/models`, which a wedged Osaurus keeps
    answering instantly while every completion hangs forever. Observed in the wild:
    a day-old process reported OK with no model resident and 0% CPU, and `ev`
    burned 600s per task discovering that a "healthy" server could not serve.

    A readiness check must exercise the same path as the work. This one asks for a
    single token, so a cold model load (~13s here) passes and a wedge fails fast
    with a stated reason instead of an unbounded hang.
    """
    import requests

    try:
        resp = requests.post(
            f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": TEST_PROMPT_CONTENT}],
                "max_tokens": 1,
            },
            timeout=timeout,
        )
    except requests.exceptions.Timeout:
        return False, f"{model}: no token within {timeout}s (server is up but not serving)"
    except requests.exceptions.ConnectionError as exc:
        return False, f"{model}: cannot connect ({exc.__class__.__name__})"
    if resp.status_code == 404:
        return False, f"{model}: not installed or registered with any provider"
    if resp.status_code != 200:
        return False, f"{model}: HTTP {resp.status_code}"
    return True, f"{model}: serving"


def ensure_server(max_retries: int = ENSURE_MAX_RETRIES, wait: int = SERVER_WAIT) -> bool:
    """Make sure osaurus is up, restarting it at most `max_retries` times.

    Between attempts the server is re-checked and given a grace period. The
    predecessor looped straight back into `restart_server`, whose first act is to
    QUIT osaurus -- so a slow-starting instance was killed by the next attempt,
    and three retries reliably ended with nothing listening. For an unattended
    routine that converts a recoverable wedge into a hard outage.
    """
    for attempt in range(1, max_retries + 1):
        if is_server_running():
            return True
        logger.warning(f"Server not responding (attempt {attempt}/{max_retries})")
        if restart_server(wait=wait):
            return True
        # Never quit again without first giving a slow start time to finish.
        if _wait_for_up(RETRY_GRACE):
            logger.info("osaurus came up during the grace period; not restarting again")
            return True
        if attempt == max_retries:
            logger.error("Server failed to restart after %s attempts", max_retries)
            return False
    return is_server_running()


def _wait_for_up(timeout: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_server_running():
            return True
        time.sleep(FAST_POLL_INTERVAL)
    return is_server_running()


def test_connection(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, model: str = None) -> dict:
    from .osaurus_lib import call

    try:
        if not is_server_running(host, port):
            return {"status": STATUS_ERROR, "message": "Server not running"}
        models = get_models(host, port)
        if not models:
            return {"status": STATUS_ERROR, "message": "No models available"}
        if model is None:
            model = models[0]
        result = call(
            model,
            [{"role": "user", "content": TEST_PROMPT_CONTENT}],
            host,
            port,
            timeout=TEST_TIMEOUT,
        )
        if result.get("error"):
            return {"status": STATUS_ERROR, "message": result["error"]}
        return {
            "status": STATUS_OK,
            "models": models,
            "test_model": model,
            "response_preview": result.get("content", "")[:PREVIEW_LIMIT],
        }
    except Exception as e:
        return {"status": STATUS_ERROR, "message": str(e)}


def panic_dump(content: str) -> None:
    _dump_base = Path(os.environ.get("OSAURUS_DUMP_DIR", str(Path.home())))
    dump_dir = _dump_base / DUMP_DIR_NAME
    dump_dir.mkdir(exist_ok=True)
    dump_file = dump_dir / f"panic_{int(time.time())}.txt"
    dump_file.write_text(content or "(empty)")
    logger.warning(f"Dumped problematic output to {dump_file}")


def check_server_or_die(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, model: str | None = None
) -> None:
    """Fail fast with a friendly message if the Osaurus/Ollama server is down or wedged.

    ZTools is a thin client and cannot do anything useful without a running
    server, so we surface the fix (install/start Osaurus) instead of letting the
    tool proceed into a confusing downstream crash.

    If `model` is given, also runs `can_serve(model)` to catch wedged instances
    (answering /v1/models instantly while completions hang forever).
    """
    if not is_server_running(host, port):
        host_str = host if host.startswith("http") else f"http://{host}:{port}"
        err(f"Osaurus server not reachable at {host_str}")
        print("  Install: brew install --cask osaurus")
        # Deliberately NOT `osaurus serve &`. A hand-started server checks for no
        # existing one and takes no GPU lock, so this advice was a recipe for two
        # servers on a machine sized for one -- and that contention is invisible to
        # the sample guard, which reads swap and compressor and not the GPU.
        # osaurus_one.sh is idempotent and refuses under a peer's run.
        print("  Start:   ./tools/osaurus_one.sh   (never start one by hand)")
        die("Start the server and retry.", code=1)

    if model:
        ok, reason = can_serve(model)
        if not ok:
            logger.warning(f"Server wedged probe failed: {reason}. Attempting restart...")
            if not restart_server():
                err(f"Osaurus server is wedged and restart failed: {reason}")
                die("Restart osaurus manually and retry.", code=1)
