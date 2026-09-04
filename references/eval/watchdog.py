"""Stall watchdog for a model's task loop.

Separate from the per-task timeout on purpose, because the two answer different
questions. The timeout asks "is this REQUEST taking too long", and it is DERIVED
from measured rates (`_derived_timeout`: cold_start + chars/prefill +
tokens/decode). The watchdog asks "is this MODEL making any progress at all", and
deliberately depends on no measurement.

That independence is the whole point. A contended machine makes measurements
slow, slow measurements inflate the derived timeout, and the inflated timeout
permits a longer stall -- the estimator self-corrects in the wrong direction
exactly when the box is least able to deliver. qwen3.8-27b-mxfp8 was measured on
a machine whose compressor held 18.07GB, recorded decode at 0.1158 tok/s, and
`max_tokens / decode` alone came to ~138,000s. Capped at MAX_EVAL_TIMEOUT that
still bought a 2-hour per-task ceiling, and the run sat wedged for 83 minutes
having completed zero tasks without tripping a single guard.
"""

from __future__ import annotations

import os
import time

from lib.tui import FAIL, WARN, console

#: Wall-clock a model may spend without COMPLETING a task before it is abandoned.
#:
#: Generous on purpose: bonsai-27b-ternary-jang legitimately spent 866s on one
#: task and completed all 30, so this must never fire on a slow-but-working model.
#: It is a backstop against no progress, not a performance budget.
MODEL_STALL_SECONDS = int(os.environ.get("EVAL_MODEL_STALL_SECONDS", "2400"))


def restart_after_stall(out=None) -> None:
    """Restart the server through the sanctioned path only.

    Imported lazily: `eval.cli_runtime` imports from `eval.run`, whose
    `eval.run_loop` imports this module, so a top-level import would be
    circular.
    """
    out = out or console
    try:
        from eval.cli_runtime import restart_server

        restart_server(out=out)
    except Exception as e:
        out.print(f"{WARN} watchdog could not restart the server: {e}")


def stalled_for(last_completion: float, now: float | None = None) -> float:
    """Seconds since the last completed task."""
    return (time.monotonic() if now is None else now) - last_completion


def check_stall(model: str, last_completion: float, out=None) -> bool:
    """True when `model` should be abandoned for lack of progress.

    Announces the abandonment and restarts the server. Says "NOT quality results"
    in the same breath, because a partial average reads exactly like a real one:
    a truncated run once reported bonsai-27b-ternary-jang at 62% over 19 tasks
    when its complete score was 79%.
    """
    out = out or console
    elapsed = stalled_for(last_completion)
    if elapsed <= MODEL_STALL_SECONDS:
        return False
    out.print(
        f"{FAIL} Abandoning {model}: no task completed in {elapsed / 60:.0f} min "
        f"(watchdog limit {MODEL_STALL_SECONDS / 60:.0f} min). The server is wedged "
        "or this model cannot serve here -- these are NOT quality results."
    )
    restart_after_stall(out=out)
    return True
