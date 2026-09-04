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

from eval import memory
from eval.run import MEMORY_WARNING_THRESHOLD

# Default server port
DEFAULT_SERVER_PORT = 1337

# Timeouts for server checks (seconds)
SERVER_RESPONSIVE_TIMEOUT = 5
FLUSH_CALL_TIMEOUT = 30

# Sleep durations during model flush (seconds)
FLUSH_SETTLE_WAIT = 2

#: How long to keep trying a real completion after a restart, and the gap between
#: tries. A 13-35GB model does not become servable in the ~21s the previous budget
#: allowed, and the sweep then recorded INFRA zeros for a server that was merely
#: still loading.
RESTART_READY_BUDGET = int(os.environ.get("EVAL_RESTART_BUDGET", "180"))
RESTART_READY_GAP = 5
#: `osaurus_one.sh --restart` has to kill, wait out a quit, relaunch and poll.
OSAURUS_ONE_TIMEOUT = 240


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


#: Fraction of RECLAIMABLE memory a model's weights may occupy before the eval
#: refuses to run it. Weights are only part of the footprint -- activations and
#: the KV cache come on top -- so the headroom is not slack, it is the rest of
#: the model.
#:
#: PROVISIONAL. The number that would settle it is a measurement of
#: qwen3.8-27b-mxfp8 (28.8GB on disk) on a quiet 64GB box, which has not been
#: taken: its only samples were recorded while the compressor held 18.07GB and
#: are tagged unclean. Until then 0.8 is carried forward unchanged from the
#: warning this refusal replaces, so the gate changes the CONSEQUENCE without
#: silently also changing the threshold.
OVERSIZE_MEMORY_FRACTION = 0.8

#: Escape hatch for the deliberate case: measuring whether an oversize model can
#: run here AT ALL is a legitimate experiment, and the refusal must not make it
#: impossible -- only conscious.
OVERSIZE_OVERRIDE_ENV = "EVAL_ALLOW_OVERSIZE"


def oversize_refusal(
    model_gb: float,
    available_gb: float = None,
    allow: bool = False,
    thrashing: bool = None,
) -> str:
    """Why this model must not be measured here, or "" to proceed.

    WHY A REFUSAL AND NOT A WARNING. The line this replaces printed "low memory
    - will be slower" and ran the model anyway. Warn-and-continue is the shape
    this repo keeps paying for: it produced a 0.1158 tok/s decode reading for
    qwen3.8-27b-mxfp8, which `max_tokens / decode` turned into a ~138,000s
    derived timeout, which then permitted a wedged server to idle 83 minutes. A
    number measured under thrashing is not a slow measurement, it is a
    measurement of the thrashing -- and it hardens into config and docs exactly
    like a real one.

    WHY PRESSURE IS ASKED FIRST. The first version of this gate asked only about
    headroom, and headroom is the quantity `eval/memory.py` documents as
    misleading here: after a sweep the page cache holds the previous model's
    weights as `active` file-backed pages, which psutil does not report as
    available even though the kernel evicts them for free. That gate would
    refuse a model that runs fine, on exactly the machine you most want readings
    from. Swap and compressor are unambiguous by comparison -- they describe a
    machine that is ALREADY paying for memory it does not have.

    So: thrashing is disqualifying on its own, and headroom is measured against
    what is RECLAIMABLE rather than what is merely free.

    Both `available_gb` and `thrashing` are injectable so every branch is
    testable without a 28.8GB model or a deliberately wrecked machine.
    """
    if allow or os.environ.get(OVERSIZE_OVERRIDE_ENV):
        return ""

    if thrashing is None:
        thrashing = memory.is_thrashing()
    if thrashing:
        reading = memory.pressure()
        detail = ""
        if reading:
            detail = f" (swap {reading[0]:.1f}GB, compressor {reading[1]:.1f}GB)"
        return (
            f"the machine is already paging{detail}. A timing taken here would "
            f"describe the paging, not the model. Wait for it to settle, or set "
            f"{OVERSIZE_OVERRIDE_ENV}=1 to measure it deliberately."
        )
    # None means "cannot tell", which is not "fine" -- but it is also not
    # evidence of thrashing, so it does not refuse on its own. The headroom
    # check below still applies.

    if available_gb is None:
        available_gb = memory.reclaimable_available_gb()
    if model_gb <= available_gb * OVERSIZE_MEMORY_FRACTION:
        return ""
    return (
        f"needs ~{model_gb:.0f}GB against {available_gb:.0f}GB reclaimable "
        f"(limit {OVERSIZE_MEMORY_FRACTION:.0%}). A timing taken here would "
        f"describe the swapping, not the model. Re-run on a quieter machine, or "
        f"set {OVERSIZE_OVERRIDE_ENV}=1 to measure it deliberately."
    )


def broken_model_refusal(model: str, allow: bool = False) -> str:
    """Why this model is broken/unviable on disk, or empty string to proceed.

    Identifies packaging anomalies (e.g. unaccelerated MTP speculative shards,
    missing weight shards, incomplete downloads) before wasting time on an unviable run.
    """
    if allow or os.environ.get(OVERSIZE_OVERRIDE_ENV) or os.environ.get("EVAL_ALLOW_BROKEN"):
        return ""
    from lib.model_caps import probe_model_defects

    defects = probe_model_defects(model)
    if defects:
        return "; ".join(defects)
    return ""



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


def osaurus_one_script():
    """Path to tools/osaurus_one.sh, or None when running from an install.

    The wheel does not ship `tools/`, so this legitimately returns None and the
    caller must degrade with a stated reason rather than inventing a path.
    """
    from lib.paths import repo_root

    root = repo_root()
    if root is None:
        return None
    script = root / "tools" / "osaurus_one.sh"
    return script if script.is_file() else None


def restart_server(out=None) -> bool:
    """Restart osaurus through the ONE sanctioned path, or say why we could not.

    This used to be hand-rolled here: `osascript` quit, `sleep 3`, then
    `open -n -a osaurus`. Three things were wrong with it and all three were live.

    `open -n` forces a NEW INSTANCE. That is exactly the second server
    `tools/osaurus_one.sh` exists to prevent -- two servers do not queue, they each
    load their own copy of the model, which is what produced the 0.1 tok/s decode and
    the 423s cold start already recorded in CLAUDE.md. The quit was never verified
    either, so the new instance could race a still-dying one for port 1337.

    The script already does this correctly: it enumerates every osaurus pid, escalates
    to SIGKILL, and polls until the port actually answers. Calling it here removes a
    parallel reimplementation of an invariant the repo had already centralised.
    """
    out = out or console
    import subprocess

    # Not ours to restart. A foreign holder means a DIFFERENT session is measuring
    # against this server, and restarting it would evict that run's model
    # mid-measurement: the peer sees HTTP 499 request_cancelled and records a rate
    # an order of magnitude low -- as a CLEAN sample, because the sample guard
    # reads swap and compressor and is blind to the GPU.
    #
    # The check lives HERE rather than at the call site. flush_between_models had
    # it, and that left two ways past: a transport EXCEPTION reached the restart
    # without passing the guard, and eval/watchdog.py calls this function directly.
    # A guard every caller has to remember is a guard that one of them will not.
    blocked_by = gpu_lock.foreign_holder()
    if blocked_by:
        out.print(
            f"{WARN} {blocked_by} holds the GPU — not restarting osaurus under "
            "another session's measurement"
        )
        return False

    script = osaurus_one_script()
    if script is None:
        out.print(
            f"{WARN} tools/osaurus_one.sh not found (running from an install?) — "
            "cannot enforce the single-server invariant; skipping restart"
        )
        return False
    try:
        proc = subprocess.run(
            [str(script), "--restart"],
            capture_output=True,
            text=True,
            timeout=OSAURUS_ONE_TIMEOUT,
        )
    except Exception as e:
        # Deliberately broad. This function's contract is "return a bool, never
        # raise": a failed restart must degrade one model, not abort a sweep that
        # has already spent hours. A narrower catch let a subprocess error escape
        # flush_between_models and take main() down with it.
        out.print(f"{FAIL} osaurus_one.sh --restart could not run: {e}")
        return False
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        out.print(
            f"{FAIL} osaurus_one.sh --restart failed (exit {proc.returncode})"
            + (f": {detail[-1]}" if detail else "")
        )
        return False
    out.print(f"{STEP} Server restarted via osaurus_one.sh")
    return True


def wait_until_model_serves(model: str, out=None, budget: int | None = None) -> bool:
    """Poll a real completion until `model` answers, or the budget runs out.

    Deliberately NOT a GET on the model-list endpoint. That is what this code used
    to check, and it returns 200 as soon as the HTTP layer is up -- which is why the
    sweep printed "Server restarted" and "Server: OK" and then failed every
    completion. Listing models proves the server is listening; it proves nothing
    about whether a 13-35GB model can be loaded and served.
    """
    out = out or console
    # Resolved at CALL time. As a default argument this bound at import time, so
    # patching RESTART_READY_BUDGET could not shorten it and the suite sat through
    # the full wait -- the same import-time-alias defect fixed elsewhere in this repo.
    budget = RESTART_READY_BUDGET if budget is None else budget
    deadline = time.monotonic() + budget
    last = ""
    while time.monotonic() < deadline:
        try:
            r = call(model, [{"role": "user", "content": "ok"}], timeout=FLUSH_CALL_TIMEOUT)
            if not r.get("error"):
                return True
            last = str(r.get("error"))
        except Exception as e:  # transport down entirely
            last = str(e)
        time.sleep(RESTART_READY_GAP)
    out.print(f"{WARN} {model} still not serving after {budget}s" + (f" ({last})" if last else ""))
    return False


def flush_between_models(prev_model: str, next_model: str, out=None) -> None:
    """Warm `next_model` before it is timed, restarting the server if it will not.

    Returns without raising in every path. A server that cannot be recovered is
    announced loudly rather than left for the caller to discover as a column of
    INFRA zeros -- the previous version printed nothing at all when its restart
    checks were exhausted and ran the model anyway.
    """
    out = out or console
    out.print(f"{STEP} Flushing {prev_model} -> {next_model}...")
    try:
        r = call(next_model, [{"role": "user", "content": "ok"}], timeout=FLUSH_CALL_TIMEOUT)
        if not r.get("error"):
            time.sleep(FLUSH_SETTLE_WAIT)
            return
        why = r.get("error")
    except Exception as e:
        # Same route as an error dict. These used to diverge, and the exception
        # path skipped the GPU-lock guard that the error path honoured.
        why = e
    out.print(f"{WARN} Flush failed ({why}), attempting restart...")

    if not restart_server(out=out):
        out.print(
            f"{FAIL} Could not restart the server before {next_model}. Its scores are "
            "NOT quality results."
        )
        return
    if not wait_until_model_serves(next_model, out=out):
        out.print(
            f"{FAIL} Server is up but {next_model} does not serve. Its scores are "
            "NOT quality results."
        )
        return
    time.sleep(FLUSH_SETTLE_WAIT)
