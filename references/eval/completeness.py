"""Did the run FINISH, and does its mean describe what it claims to?

A model's task loop has two abandon paths, and both of them print the right thing
and enforce nothing:

    eval/watchdog.py   "Abandoning {model}: no task completed in {n} min ...
                        these are NOT quality results"
    eval/run.py        "Abandoning {model}: {n} consecutive infrastructure
                        failures ... this is not a quality result and must not
                        be read as one"

Both then `break`, and everything downstream treats the short list of results
exactly like a complete one. `compute_score_stats` takes the mean over whatever
finished, `save_historical_results` writes those entries to disk forever, and the
next session reads the JSON rather than the console scrollback that carried the
warning.

That is not hypothetical. A truncated run reported `bonsai-27b-ternary-jang` at
62% when its complete score was 79%; losing `gemma-4-12b` to an obvious 0% was
the visible damage and bonsai's plausible-looking 62% was the dangerous one.
`ornith-1.0-9b-mxfp8` stopped at 11 of 30 with 6 timeouts, and its 55 history
entries include that run with no way to tell it apart from the rest.

THE CLASS: a warning that exists only on stdout is not a gate.

WHY THIS COMPARES SETS RATHER THAN THREADING A FLAG. The obvious fix is a
`truncated=True` set at each `break`. That is the per-knob invalidation hook this
repo already has a rule about (#12): there are two break paths today, a third is
one bug away, and the one that forgets to set the flag is the one that ships. So
completeness is DERIVED by diffing the tasks that were asked for against the
tasks that reported back. A future abandon path is covered the day it is written,
without knowing this module exists.
"""

from typing import Dict, List


def is_runnable(task_cfg: Dict) -> bool:
    """Whether `run_eval` will actually attempt this task.

    Mirrors the skip in `eval.run.run_eval`: a task with no `messages` key is
    skipped with a warning and can never appear in the results, so counting it
    as expected would report every run as truncated. Kept as a named predicate
    rather than an inline check so the two cannot drift apart silently.
    """
    return isinstance(task_cfg, dict) and "messages" in task_cfg


def expected_task_names(tasks: Dict) -> List[str]:
    """The tasks a complete run is obliged to report on."""
    return [name for name, cfg in (tasks or {}).items() if is_runnable(cfg)]


def _reason(missing: List[str], results: List[Dict]) -> str:
    """Why the run is short, in the terms the abandon paths already use.

    Derived from the results rather than passed in, for the same reason the
    verdict is: a reason threaded from the break site is a reason the next break
    site forgets. The last result's failure category is what distinguishes a
    wedged server (TIMEOUT/INFRA) from a run the user interrupted.
    """
    if not results:
        return f"no task completed; all {len(missing)} missing"
    last = results[-1] or {}
    category = last.get("failure_category") or "unknown"
    return (
        f"abandoned after {len(results)} task(s); {len(missing)} not run "
        f"({', '.join(missing[:3])}{'...' if len(missing) > 3 else ''}); "
        f"last failure category {category}"
    )


def assess(tasks: Dict, results: List[Dict]) -> Dict:
    """Compare what was asked for against what reported back.

    Returns a record that travels with the run: `complete` is the verdict every
    consumer gates on, and the counts are what makes an incomplete run legible
    instead of merely rejected.
    """
    expected = expected_task_names(tasks)
    reported = [r.get("task") for r in (results or []) if isinstance(r, dict)]
    seen = set(reported)
    missing = [name for name in expected if name not in seen]
    complete = not missing
    return {
        "expected": len(expected),
        "completed": len(seen & set(expected)),
        "missing": missing,
        "complete": complete,
        "reason": "" if complete else _reason(missing, results or []),
    }


def is_complete(record: Dict) -> bool:
    """Whether a per-model record in `all_results` came from a finished run.

    Absent metadata reads as COMPLETE on purpose. Every historical record and
    every test fixture predates this module, and defaulting them to incomplete
    would retroactively disqualify every real measurement this repo has taken.
    The cost of that default is that a caller which forgets to attach the record
    is trusted; `eval.cli` is the one such caller and a test pins it.
    """
    if not isinstance(record, dict):
        return True
    meta = record.get("completeness")
    if not isinstance(meta, dict):
        return True
    return bool(meta.get("complete", True))
