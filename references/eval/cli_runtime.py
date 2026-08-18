#!/usr/bin/env python3
"""Runtime helpers for the evaluator CLI: memory guards, server probes, config.

Split out of cli.py for the repo's 500-line limit. cli.py re-exports these, so
`from eval.cli import check_memory_safe` and friends keep working.
"""

import atexit
import os
import re
import sys
import time

import requests
import tomli_w
from lib import gpu_lock
from lib.config import get_model_prompts_all
from lib.llm.constants import API_TAGS
from lib.osaurus_lib import call
from lib.paths import conf_path
from lib.signal_handling import register_cleanup

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


def hold_gpu_for_eval(label: str, out=None, timeout: int = gpu_lock.DEFAULT_TIMEOUT) -> None:
    """Take the GPU lock for the whole run, and arrange for it to come back.

    An eval is exactly the run this repo's worst silent failure happens to.
    Several agent sessions run on this Mac at once, and any of them restarting
    osaurus mid-measurement -- via tools/osaurus_one.sh, or via any tool whose
    reaction to a slow server is to quit it -- evicts this run's model. The
    symptom is HTTP 499 request_cancelled and a rate an order of magnitude low,
    which looks exactly like a slow model -- and `eval/samples.py` files it as a
    CLEAN sample, because `machine_is_uncontended()` reads swap and compressor and
    cannot see the GPU. The median outvotes noise it can detect; this it cannot.

    Held for the whole process rather than per model: the invariant is not
    established by having been true at startup, which is the same lesson the
    per-task contention warning already encodes.

    THREE releases, because a run dies in more ways than it exits.
    `register_cleanup` covers SIGINT/SIGTERM, `atexit` covers a normal return and
    an unhandled exception, and the lock's own liveness check covers SIGKILL --
    which runs neither. Release is idempotent, so overlapping paths are free.
    """
    out = out or console
    gpu_lock.acquire(label, timeout=timeout, log=lambda msg: out.print(msg))
    register_cleanup(gpu_lock.release)
    atexit.register(gpu_lock.release)


# ==========================================================
# Memory monitoring
# ==========================================================


_warned_no_psutil = False


def get_memory_percent() -> float:
    """Current memory usage percent, or 0.0 with a stated reason.

    psutil is a hard dependency now. Returning a bare 0.0 on ImportError made
    the memory guard read "0% used" and never fire, so the degrade says so
    rather than looking like a healthy machine.
    """
    global _warned_no_psutil
    try:
        import psutil

        return psutil.virtual_memory().percent
    except ImportError:
        if not _warned_no_psutil:
            _warned_no_psutil = True
            print(
                f"{WARN} psutil is not installed — the memory guard is disabled",
                file=sys.stderr,
            )
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
    """Memory a model needs, in GB, from its weight files where they can be found.

    The parameter count in the NAME is not the answer: qwen3.8-27b-4bit and
    qwen3.8-27b-mxfp8 are both "27b" and occupy 15GB and 27GB respectively. Reading
    "27" out of the name warned "Model needs ~27GB, low memory" about the 15GB build
    while saying nothing useful about the 27GB one, which genuinely did not fit and
    ran at 0.08 tok/s because of it.

    On-disk weight bytes is the quantity that predicts whether a model fits. Falls
    back to the name only for models with nothing on disk to measure.
    """
    from lib.model_caps import model_disk_bytes

    disk = model_disk_bytes(model)
    if disk:
        # Round up: a model needs at least its weights, plus room for activations and
        # a KV cache, so the honest direction for a memory WARNING is generous.
        return max(1, -(-disk // 1_073_741_824))

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
            # Not ours to restart. This runs BETWEEN models in a sweep, so an eval
            # that reached here normally holds the GPU lock itself and sails
            # through -- foreign_holder() ignores our own hold. A non-empty answer
            # means a DIFFERENT session is measuring against this server, and
            # quitting it would evict that run's model mid-measurement. The peer
            # would see HTTP 499 request_cancelled and record a rate an order of
            # magnitude low -- as a CLEAN sample, since the sample guard reads
            # swap and compressor and is blind to the GPU.
            #
            # `open -n` is the sharper edge of the two. It launches a SECOND
            # osaurus unconditionally, so if the quit above were merely skipped
            # and this were not, the recovery path would itself create the
            # two-server contention it is trying to recover from.
            blocked_by = gpu_lock.foreign_holder()
            if blocked_by:
                out.print(
                    f"{WARN} Flush failed, but {blocked_by} holds the GPU — not "
                    f"restarting osaurus under another session's measurement"
                )
                time.sleep(FLUSH_SETTLE_WAIT)
                return
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
