#!/usr/bin/env python3
import os
import signal
import time
import subprocess
from pathlib import Path


from .logging_config import osaurus_logger as logger
from .osaurus_models import is_server_running, get_models, DEFAULT_HOST, DEFAULT_PORT

PID_FILE = Path.home() / ".osaurus.pid"

# Configuration constants to prevent magic strings and numbers (Mitchell Hashimoto design)
RESTART_SLEEP = 1
SERVER_WAIT = 20
ENSURE_MAX_RETRIES = 3
TEST_TIMEOUT = 10
OSASCRIPT_QUIT_TIMEOUT = 5
OSASCRIPT_QUIT_CMD = 'quit app "osaurus"'
DEFAULT_APP_PATH = "/Applications/osaurus.app"
GUI_LAUNCHER_CMD = "open"
CLI_LAUNCHER_CMD = ["osaurus", "serve", "--yes"]
POLL_SLEEP_SEC = 1
TEST_PROMPT_CONTENT = "Hi"
DUMP_DIR_NAME = "llm_dumps"
PREVIEW_LIMIT = 100
STATUS_ERROR = "error"
STATUS_OK = "ok"


def _is_osaurus_process(pid: int) -> bool:
    """Verify that a given PID belongs to an active process named or running osaurus."""
    try:
        res = subprocess.run(
            ["ps", "-p", str(pid), "-o", "args="],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if res.returncode == 0 and res.stdout:
            return "osaurus" in res.stdout.lower()
    except Exception:
        pass
    return False


def _kill_osaurus():
    try:
        subprocess.run(["osascript", "-e", OSASCRIPT_QUIT_CMD], capture_output=True, timeout=OSASCRIPT_QUIT_TIMEOUT)
    except Exception as e:
        logger.debug(f"Failed to quit osaurus app via osascript: {e}")

    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            if _is_osaurus_process(pid):
                logger.info(f"Terminating osaurus process with PID {pid} via SIGTERM")
                os.kill(pid, signal.SIGTERM)
                time.sleep(RESTART_SLEEP)
            else:
                logger.warning(f"PID file contains process ID {pid} which is not osaurus. Skipping termination.")
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
    
    # Poll every 0.1s up to wait seconds (Carmack/Hashimoto fast-poll latency optimization)
    poll_interval = 0.1
    max_polls = int(wait / poll_interval)
    for _ in range(max_polls):
        time.sleep(poll_interval)
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
    dump_dir = Path.home() / DUMP_DIR_NAME
    dump_dir.mkdir(exist_ok=True)
    dump_file = dump_dir / f"panic_{int(time.time())}.txt"
    dump_file.write_text(content or "(empty)")
    logger.warning(f"Dumped problematic output to {dump_file}")
