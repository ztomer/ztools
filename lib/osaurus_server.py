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


def _kill_osaurus():
    try:
        subprocess.run(["osascript", "-e", OSASCRIPT_QUIT_CMD], capture_output=True, timeout=OSASCRIPT_QUIT_TIMEOUT)
    except Exception:
        pass

    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            time.sleep(RESTART_SLEEP)
        except (ProcessLookupError, ValueError, OSError):
            pass
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
        print(f"Failed to restart: {e}")
        return False
    for i in range(wait):
        time.sleep(POLL_SLEEP_SEC)
        if is_server_running():
            return True
    return False


def ensure_server(max_retries: int = ENSURE_MAX_RETRIES, wait: int = SERVER_WAIT) -> bool:
    for attempt in range(1, max_retries + 1):
        if is_server_running():
            return True
        print(f"Server not responding (attempt {attempt}/{max_retries})")
        if not restart_server(wait=wait):
            if attempt == max_retries:
                print("Server failed to restart")
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
