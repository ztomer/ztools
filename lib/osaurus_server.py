#!/usr/bin/env python3
import os
import signal
import time
import subprocess
from pathlib import Path


from .logging_config import osaurus_logger as logger
from .osaurus_models import is_server_running, get_models, DEFAULT_HOST, DEFAULT_PORT

PID_FILE = Path.home() / ".osaurus.pid"

# Sleep durations (seconds)
RESTART_SLEEP = 1
SERVER_WAIT = 20

# Retries
ENSURE_MAX_RETRIES = 3

# API call timeout (seconds)
TEST_TIMEOUT = 10


def _kill_osaurus():
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            time.sleep(RESTART_SLEEP)
        except (ProcessLookupError, ValueError, OSError):
            pass
        PID_FILE.unlink(missing_ok=True)


def restart_server(app_path: str = "/Applications/osaurus.app", wait: int = SERVER_WAIT) -> bool:
    _kill_osaurus()
    time.sleep(RESTART_SLEEP)

    if Path(app_path).exists():
        launcher = ["open", app_path]
    else:
        launcher = ["osaurus", "serve", "--yes"]
    try:
        proc = subprocess.Popen(
            launcher,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        PID_FILE.write_text(str(proc.pid))
    except Exception as e:
        print(f"Failed to restart: {e}")
        return False
    for i in range(wait):
        time.sleep(1)
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
            return {"status": "error", "message": "Server not running"}
        models = get_models(host, port)
        if not models:
            return {"status": "error", "message": "No models available"}
        if model is None:
            model = models[0]
        result = call(
            model,
            [{"role": "user", "content": "Hi"}],
            host,
            port,
            timeout=TEST_TIMEOUT,
        )
        if result.get("error"):
            return {"status": "error", "message": result["error"]}
        return {
            "status": "ok",
            "models": models,
            "test_model": model,
            "response_preview": result.get("content", "")[:100],
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def panic_dump(content: str) -> None:
    dump_dir = Path.home() / "llm_dumps"
    dump_dir.mkdir(exist_ok=True)
    dump_file = dump_dir / f"panic_{int(time.time())}.txt"
    dump_file.write_text(content or "(empty)")
    logger.warning(f"Dumped problematic output to {dump_file}")
