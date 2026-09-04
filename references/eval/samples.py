"""Performance samples that can RECOVER from a bad reading.

The recorders used to keep the extreme observation -- slowest rate, longest cold
start -- on the reasoning that a timeout must hold on a bad run rather than a lucky
one. That reasoning is sound for a timeout and fatal for a measurement, because the
extreme is exactly what a contended machine produces, and nothing can ever displace
it. A single reading taken while something else held the box became permanent.

It was not hypothetical. A leaked plugin daemon held 31GB of this machine's 64 for a
day. Everything measured under it was wrong, and because the estimator kept the worst
value, re-measuring on a healthy machine changed nothing:

    nemotron-3.5-lightning-30b   recorded 0.68 tok/s   actually ~33 tok/s
    gemma-4-e2b-it-8bit (2B)     recorded 7.04 tok/s   its 4B sibling does 26.5
    gemma-4-12b-it-mxfp8         recorded 309s cold    a 16GB model loads in ~33s

Those numbers reached `conf/config.toml`, `docs/MODEL_QUIRKS.md` and a `default_model`
choice, and the only remedy was deleting the file by hand -- a step nobody is reminded
to take.

So samples are kept as a LIST, each tagged with whether the machine was quiet when it
was taken, and the estimate is the MEDIAN OF RECENT CLEAN SAMPLES. A bad reading is
outvoted rather than enshrined. The scalar key is still written alongside, so every
existing reader keeps working unchanged.
"""

import statistics
import time
from typing import Dict, List, Optional

from eval import memory

# Re-exported: these moved to eval/memory.py so the oversize refusal and the
# sample-clean gate share ONE definition of "the machine is thrashing" rather
# than two that drift. Callers importing them from here keep working.
from eval.memory import MAX_CLEAN_COMPRESSOR_GB, MAX_CLEAN_SWAP_GB

#: How many recent samples the estimate considers. Small enough to track a real
#: change in the machine, large enough that one bad reading cannot carry the median.
SAMPLE_WINDOW = 5


def gpu_is_contended() -> bool:
    """Is another session measuring against the GPU right now?

    Swap and compressor cannot see this. A peer agent session running its own
    eval leaves both quiet while competing for the exact resource being timed,
    so its contention was recorded as a CLEAN sample -- the documented hole in
    the guard below, and the reason `restart_server` already consults this same
    predicate before evicting a peer's model.

    BE CLEAR WHAT THIS BUYS. It catches a peer holding the machine-wide GPU lock,
    which is the failure this repo actually recorded. It does NOT catch GPU work
    that never takes the lock -- Blender, a game, a video encode -- because
    nothing short of real GPU telemetry does and `powermetrics` needs sudo. A
    gate that overstates its coverage is worse than one that admits its hole.
    """
    try:
        from lib import gpu_lock

        return gpu_lock.foreign_holder() is not None
    except Exception:
        # Cannot tell, so claim nothing. Returning True here would tag every
        # sample on a machine without the lock module as contended.
        return False


def machine_is_uncontended() -> bool:
    """Is the machine quiet enough for a timing to mean anything?

    Gates on PRESSURE (swap, compressor), not on free memory. After a sweep the page
    cache legitimately holds tens of GB of model weights and "available" drops to
    ~12GB on a perfectly healthy box, so a headroom threshold refuses to record on
    exactly the machine you most want readings from.

    Plus the GPU: see `gpu_is_contended` for what that check does and does not
    cover.
    """
    if gpu_is_contended():
        return False
    reading = memory.pressure()
    if reading is None:
        # Cannot tell. Record the sample as unverified rather than dropping it or
        # calling it clean -- both would be inventing an answer.
        return False
    swap_gb, compressor_gb = reading
    return swap_gb <= MAX_CLEAN_SWAP_GB and compressor_gb <= MAX_CLEAN_COMPRESSOR_GB


def add_sample(caps: Dict, key: str, value: float, clean: Optional[bool] = None) -> float:
    """Append a sample for `key` and return the re-derived estimate.

    Mutates `caps` in place: writes `<key>_samples` (the history) and `key` itself
    (the derived scalar), so callers that read the scalar need no changes.
    """
    if clean is None:
        clean = machine_is_uncontended()
    history: List[Dict] = list(caps.get(f"{key}_samples") or [])
    history.append({"v": round(float(value), 4), "ts": round(time.time(), 1), "clean": bool(clean)})
    history = history[-(SAMPLE_WINDOW * 2):]
    caps[f"{key}_samples"] = history
    estimate = estimate_from(history)
    caps[key] = round(estimate, 2)
    return estimate


def estimate_from(history: List[Dict]) -> float:
    """Median of the most recent clean samples, falling back to all recent ones.

    Preferring clean samples is what lets a contaminated reading be outvoted. Falling
    back when there are none is what stops the estimator returning nothing on a
    machine that has only ever been measured under load -- a stale number is worth
    more than no number, provided it can still be displaced later.
    """
    if not history:
        return 0.0
    clean = [s["v"] for s in history if s.get("clean")][-SAMPLE_WINDOW:]
    if clean:
        return statistics.median(clean)
    return statistics.median([s["v"] for s in history][-SAMPLE_WINDOW:])


def migrate_scalar(caps: Dict, key: str) -> None:
    """Seed the history from a pre-existing scalar, ONCE, marked unclean.

    Unclean on purpose. The scalars on disk were recorded under the old
    extreme-keeping rule and some were taken during the leak, so they must not be
    trusted as clean baselines -- but discarding them would throw away the only
    reading some models have. Tagged this way they are used until real clean samples
    arrive, then outvoted.
    """
    if f"{key}_samples" in caps or key not in caps:
        return
    value = caps.get(key)
    if not isinstance(value, (int, float)) or value <= 0:
        return
    caps[f"{key}_samples"] = [
        {"v": round(float(value), 4), "ts": 0.0, "clean": False, "legacy": True}
    ]


def clean_estimate(caps: Dict, key: str) -> Optional[float]:
    """The estimate for `key`, but ONLY when a clean sample backs it.

    `estimate_from` deliberately falls back to unclean samples so a model
    measured only under load still reports something rather than nothing. That is
    right for display and wrong for sizing a timeout, and the difference was not
    academic: qwen3.8-27b-mxfp8 was measured on a box whose compressor held
    18.07GB, recorded decode at 0.1158 tok/s, and `max_tokens / decode` alone came
    to ~138,000s. Capped at MAX_EVAL_TIMEOUT that still bought a 2-hour per-task
    timeout, so a wedged server sat unnoticed for 83 minutes.

    A contended machine makes measurements slow, slow measurements inflate the
    timeout, and the inflated timeout permits a longer stall. The estimator
    self-corrects in the WRONG direction, so the timeout path asks for clean
    samples only and takes the documented floor when there are none.
    """
    history = caps.get(f"{key}_samples") or []
    clean = [s["v"] for s in history if s.get("clean")][-SAMPLE_WINDOW:]
    if not clean:
        return None
    return statistics.median(clean)
