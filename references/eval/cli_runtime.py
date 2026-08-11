#!/usr/bin/env python3
"""Runtime helpers for the evaluator CLI: memory guards, server probes, config.

Split out of cli.py for the repo's 500-line limit. cli.py re-exports these, so
`from eval.cli import check_memory_safe` and friends keep working.
"""

import os
import re
import time

import requests
import tomli_w
from lib.config import get_model_prompts_all
from lib.llm.constants import API_TAGS
from lib.osaurus_lib import call
from lib.paths import conf_path

# cli_runtime is the lowest layer of the CLI split (cli.py imports it), so it
# owns the console seam: cli.py and cli_results.py resolve it through this
# module, keeping ONE patch point (eval.cli_runtime.console) for all three.
from lib.tui import FAIL, STEP, WARN, console

from eval.run import MEMORY_WARNING_THRESHOLD

# Default server port
DEFAULT_SERVER_PORT = 1337

# Timeouts for server checks (seconds)
SERVER_RESPONSIVE_TIMEOUT = 5
RESTART_CHECK_TIMEOUT = int(os.environ.get("EVAL_RESTART_TIMEOUT", "2"))
FLUSH_CALL_TIMEOUT = 30

# Sleep durations during model flush (seconds)
FLUSH_QUIT_WAIT = 3
FLUSH_RESTART_WAIT = 8
FLUSH_SETTLE_WAIT = 2

# Retries for server restart check
RESTART_CHECK_RETRIES = 5



# ==========================================================
# Memory monitoring
# ==========================================================


def get_memory_percent() -> float:
    """Get current memory usage percent."""
    try:
        import psutil

        return psutil.virtual_memory().percent
    except ImportError:
        return 0.0


def check_memory_safe(out=None) -> bool:
    """Check if memory is safe to run eval."""
    out = out or console
    mem_pct = get_memory_percent()
    if mem_pct > MEMORY_WARNING_THRESHOLD:
        out.print(f"{WARN} Memory at {mem_pct}% - may cause OOM")
        return False
    return True


def is_server_responsive(
    host: str = "localhost",
    port: int = DEFAULT_SERVER_PORT,
    timeout: int = SERVER_RESPONSIVE_TIMEOUT,
) -> bool:
    """Check if osaurus server is responsive."""

    try:
        with requests.Session() as s:
            resp = s.get(f"http://{host}:{port}{API_TAGS}", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def print_memory_usage(out=None):
    """Print current memory usage once (no thread)."""
    out = out or console
    mem = get_memory_percent()
    if mem > MEMORY_WARNING_THRESHOLD:
        out.print(f"{WARN} Memory at {mem}%")
    return mem


def estimate_model_memory(model: str) -> int:
    """Estimate memory needed for a model (in GB). Extract size from model name."""

    match = re.search(r"(\d+)b", model.lower())
    if match:
        return int(match.group(1))
    return 4


# ==========================================================
# Helper to build tasks from model configs
# ==========================================================


def load_tasks_from_config(model: str):
    """Build task prompts from model config YAML."""
    prompts = get_model_prompts_all(model)
    if not prompts:
        return None

    built = {}

    if "weekend_fixed" in prompts:
        built["detailed_json"] = prompts["weekend_fixed"]
    if "weekend_transient" in prompts:
        built["json"] = prompts["weekend_transient"]
    if "summarize" in prompts:
        built["summarize"] = prompts["summarize"]
    if "filename" in prompts:
        built["filename"] = prompts["filename"]
    if "file_summary" in prompts:
        built["file_summary"] = prompts["file_summary"]

    return built


def update_config(best_models: dict, out=None):
    """Update config with best models per task."""
    out = out or console
    import tomllib


    config_path = conf_path("config.toml")
    toml_path = config_path
    if not toml_path.exists():
        out.print(f"{WARN} Config file not found, skipping update.")
        return

    with open(toml_path, "rb") as f:
        config = tomllib.load(f)

    if "best_models" not in config:
        config["best_models"] = {}

    for task, model in best_models.items():
        if model:
            config["best_models"][task] = model

    with open(toml_path, "wb") as f:
        tomli_w.dump(config, f)


def flush_between_models(prev_model: str, next_model: str, out=None) -> None:
    out = out or console
    import subprocess

    out.print(f"{STEP} Flushing {prev_model} -> {next_model}...")
    try:
        r = call(next_model, [{"role": "user", "content": "ok"}], timeout=FLUSH_CALL_TIMEOUT)
        if r.get("error"):
            out.print(f"{WARN} Flush failed, attempting restart...")
            try:
                subprocess.run(["osascript", "-e", 'quit app "osaurus"'], capture_output=True)
            except Exception:
                pass
            time.sleep(FLUSH_QUIT_WAIT)
            try:
                subprocess.run(["open", "-n", "-a", "osaurus"], capture_output=True)
            except Exception:
                pass
            time.sleep(FLUSH_RESTART_WAIT)
            for _ in range(RESTART_CHECK_RETRIES):
                try:
                    import requests

                    with requests.Session() as s:
                        _llm_host = os.environ.get("OLLAMA_BASE_URL", "http://localhost:1337")
                        resp = s.get(f"{_llm_host}{API_TAGS}", timeout=RESTART_CHECK_TIMEOUT)
                    if resp.status_code == 200:
                        out.print(f"{STEP} Server restarted")
                        break
                except Exception:
                    time.sleep(FLUSH_SETTLE_WAIT)
    except Exception as e:
        out.print(f"{FAIL} Flush error: {e}")
    time.sleep(FLUSH_SETTLE_WAIT)
