#!/usr/bin/env python3
import time
import subprocess
from pathlib import Path


from .logging_config import osaurus_logger as logger
from .osaurus_models import is_server_running, get_models, DEFAULT_HOST, DEFAULT_PORT


def restart_server(app_path: str = "/Applications/osaurus.app", wait: int = 20) -> bool:
    try:
        subprocess.run(["pkill", "-f", "osaurus"], stderr=subprocess.DEVNULL)
    except Exception:
        pass
    time.sleep(2)
    try:
        subprocess.Popen(
            ["open", app_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"Failed to restart: {e}")
        return False
    for i in range(wait):
        time.sleep(1)
        if is_server_running():
            return True
    return False


def ensure_server(max_retries: int = 3, wait: int = 20) -> bool:
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
            timeout=10,
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
