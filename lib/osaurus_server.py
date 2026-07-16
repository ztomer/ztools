#!/usr/bin/env python3
import os
import signal
import subprocess
import time
from pathlib import Path

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


def _kill_osaurus():
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


def restart_server(app_path: str = DEFAULT_APP_PATH, wait: int = SERVER_WAIT) -> bool:
    _kill_osaurus()
    time.sleep(RESTART_SLEEP)

    is_gui = Path(app_path).exists()
    if is_gui:
        launcher = [GUI_LAUNCHER_CMD, app_path]
    else:
        launcher = CLI_LAUNCHER_CMD
    try:
        proc = subprocess.Popen(
            launcher,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not is_gui:
            PID_FILE.write_text(str(proc.pid))
    except Exception as e:
        logger.error(f"Failed to restart osaurus server: {e}")
        return False

    # Poll up to wait seconds (Carmack/Hashimoto fast-poll latency optimization)
    max_polls = int(wait / FAST_POLL_INTERVAL)
    for _ in range(max_polls):
        time.sleep(FAST_POLL_INTERVAL)
        if is_server_running():
            return True
    return False


def ensure_server(max_retries: int = ENSURE_MAX_RETRIES, wait: int = SERVER_WAIT) -> bool:
    for attempt in range(1, max_retries + 1):
        if is_server_running():
            return True
        logger.warning(f"Server not responding (attempt {attempt}/{max_retries})")
        if not restart_server(wait=wait):
            if attempt == max_retries:
                logger.error("Server failed to restart")
                return False
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


def check_server_or_die(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Fail fast with a friendly message if the Osaurus/Ollama server is down.

    ZTools is a thin client and cannot do anything useful without a running
    server, so we surface the fix (install/start Osaurus) instead of letting the
    tool proceed into a confusing downstream crash.
    """
    if is_server_running(host, port):
        return
    host_str = host if host.startswith("http") else f"http://{host}:{port}"
    err(f"Osaurus server not reachable at {host_str}")
    print("  Install: brew install --cask osaurus")
    print("  Start:   osaurus serve &>/dev/null &   (or launch the Osaurus.app GUI)")
    die("Start the server and retry.", code=1)
