"""Per-model, per-task observations the eval accumulates across runs.

Split out of eval/run.py: the store is read by the probe, by the eval loop and
by lib/model_caps.py, and a module that three consumers share should not live
inside one of them. EVAL_SIGNALS_PATH is read through the module at call time,
so pointing it at a tmp file in tests keeps working after this move.
"""

import json
import os
from pathlib import Path

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is a hard dep in practice
    psutil = None

from lib.config import Task, get_timeout
from lib.paths import conf_dir

DEFAULT_EVAL_TIMEOUT = int(os.environ.get("EVAL_DEFAULT_TIMEOUT", "900"))

# Learned per-model timeouts. This file is tracked, so a test run that exercises the
# eval loop would otherwise rewrite it and dirty the working tree on every `pytest`.
# tests/conftest.py::_signals_files_stay_clean patches EVAL_SIGNALS_PATH (the resolved
# path, not this dir) at a tmp file for the whole session, to keep runs side-effect free.
EVAL_SIGNALS_DIR = Path(
    os.environ.get("EVAL_SIGNALS_DIR", str(conf_dir()))
)
EVAL_SIGNALS_PATH = EVAL_SIGNALS_DIR / "eval_signals.json"


def _load_eval_signals():
    try:
        if EVAL_SIGNALS_PATH.exists():
            return json.loads(EVAL_SIGNALS_PATH.read_text())
    except Exception:
        pass
    return {}


def _save_eval_signals(signals):
    EVAL_SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_SIGNALS_PATH.write_text(json.dumps(signals, indent=2, sort_keys=True))


# A POLICY ceiling, not an estimate: the point past which a request is assumed
# wedged rather than slow. It is deliberately not derived, because its job is to
# bound the damage when the measurements are wrong. Everything else in this
# calculation is measured on the host.
MAX_EVAL_TIMEOUT = int(os.environ.get("EVAL_MAX_TIMEOUT", "7200"))


def _derived_timeout(model: str, prompt_chars: int, max_tokens: int) -> int:
    """How long this model plausibly needs: load + ingest + generate.

    A flat 900s killed qwen3.6-35b on every task. The arithmetic says why:
    ingesting the largest eval prompt costs it ~228s, but generating a
    max_tokens=16000 answer costs 500s+ on top, and the ceiling cut in before it
    finished. Each abandoned request kept its server slot, twelve of those
    exhausted the pool, and every model after it got HTTP 503 -- so one
    hand-picked constant took out the rest of the sweep.

    Both rates are measured per model by the prefill probe. When a model has
    never been measured the caller keeps the old floor, since guessing fast here
    means killing a request that was working.
    """
    caps = (_load_eval_signals().get(model) or {}).get("_capabilities") or {}
    prefill = caps.get("prefill_chars_per_sec")
    decode = caps.get("decode_tokens_per_sec")
    cold_start = caps.get("cold_start_seconds")
    # Every term measured, or no answer at all. Filling a missing one with a
    # plausible constant is how a guess ends up wearing a measurement's
    # authority -- the caller keeps its documented floor instead.
    if not prefill or not decode or not cold_start:
        return 0
    seconds = cold_start + prompt_chars / prefill + max_tokens / decode
    return int(min(seconds * TIMEOUT_SAFETY_FACTOR, MAX_EVAL_TIMEOUT))


# Headroom over the measured estimate. Chosen, not measured -- deriving it would
# need a variance estimate this has no samples for -- so it is kept as a single
# explicit multiplier rather than smuggled into the terms above. It matches the
# factor the learned-p95 path has always applied.
TIMEOUT_SAFETY_FACTOR = float(os.environ.get("EVAL_TIMEOUT_SAFETY", "1.5"))


def _effective_timeout(
    model: str, task_name: str, prompt_chars: int = 0, max_tokens: int = 0
) -> int:
    signals = _load_eval_signals()
    task_signals = signals.get(model, {}).get(task_name, {})
    learned = task_signals.get("timeout", 0)
    try:
        configured = get_timeout(Task(task_name))
    except Exception:
        configured = DEFAULT_EVAL_TIMEOUT
    derived = _derived_timeout(model, prompt_chars, max_tokens)
    return max(learned, configured, DEFAULT_EVAL_TIMEOUT, derived)


def _record_signal(
    model: str,
    task_name: str,
    time_taken: float,
    had_retries: bool,
    is_parse_failure: bool,
):
    if not time_taken and not had_retries:
        return
    signals = _load_eval_signals()
    per_task = signals.setdefault(model, {}).setdefault(task_name, {})
    samples = per_task.get("samples", 0)
    p95 = per_task.get("p95_latency", 0)

    if time_taken > 0:
        if p95:
            p95 = max(time_taken, p95 * 0.95 + time_taken * 0.05)
        else:
            p95 = time_taken
        per_task["p95_latency"] = round(p95, 1)

    per_task["samples"] = samples + 1
    per_task["total_retries"] = per_task.get("total_retries", 0) + (1 if had_retries else 0)
    per_task["parse_failures"] = per_task.get("parse_failures", 0) + (1 if is_parse_failure else 0)

    if time_taken > 0 and p95 > 0:
        new_timeout = max(DEFAULT_EVAL_TIMEOUT, int(p95 * 1.5))
        if new_timeout != per_task.get("timeout"):
            per_task["timeout"] = new_timeout

    _save_eval_signals(signals)


#: Processes named exactly this are osaurus servers. Matched on the process NAME, not
#: on a command-line substring: a `pgrep -f` for the binary path also matches any shell
#: whose command line happens to quote that path -- including the diagnostic commands
#: you run while investigating, which is how a count of "3 servers" turned out to be one
#: server and two of my own greps.
OSAURUS_PROCESS_NAME = "osaurus"


def count_osaurus_servers() -> int:
    """How many osaurus servers are running. -1 when it cannot be determined.

    `psutil` is imported at module scope (top of this file) rather than here, so that
    a test can patch it. Imported inside the function it is unreachable from the
    outside, and a test that tries to mock it measures the REAL machine instead --
    which passes or fails depending on what happens to be running, i.e. it tests
    nothing and reports green.
    """
    if psutil is None:
        return -1
    try:
        return sum(
            1 for p in psutil.process_iter(["name"])
            if (p.info.get("name") or "") == OSAURUS_PROCESS_NAME
        )
    except Exception:
        return -1


def contended_server_warning(model: str, task_name: str) -> str:
    """Non-empty when more than one osaurus is running, i.e. the measurement is dirty.

    A second server does not queue behind the first, it loads its own copy of the
    weights. Two of them on a machine sized for one produce evictions, swapping, and
    requests the server cancels itself with HTTP 499 -- which from the client is
    indistinguishable from a slow model. That is how qwen3.8-27b got recorded at
    0.1 tok/s with a 423s cold start.

    Checked PER TASK rather than once per model, because the invariant is not
    established by having been true at startup. tools/osaurus_one.sh runs once before
    a model and then hours pass; a second server appeared mid-run at least once, sat
    idle burning CPU, and was noticed only by accident. The check exists and is
    correct -- nothing was calling it.
    """
    n = count_osaurus_servers()
    if n <= 1:
        return ""
    return (
        f"{n} osaurus servers running while measuring {model}/{task_name}. They "
        f"compete for the same GPU and RAM, so this timing is not comparable with "
        f"the rest of the sweep. Fix with: ./tools/osaurus_one.sh --restart"
    )
