"""Which tasks can RANK models, and which only gate them.

A task that every competent model passes is still worth running -- it catches a
regression, and `image_real` catches a model that cannot see at all -- but it
cannot order the models that pass it. Averaged in with the rest it does something
worse than nothing: it pulls every model toward the same number and dilutes the
tasks that do separate them.

Measured over the 8 models with complete 30/30 runs, on the date in MEASURED:

    taxes_yoy_narrative   7 distinct values / 8   ranks
    taxes_qa              5 distinct values / 8   ranks, partly saturated
    taxes_slip_qa         2 distinct values / 8   GATE, 7 of 8 at 100
    image_real            2 distinct values / 8   GATE

Until now "gates are weighted at zero for ranking" was a convention held in
someone's head. Nothing in `eval/` mentioned weights at all, so both gates
entered every mean at full weight.

THE ANTI-ROT MECHANISM. A hand-typed list of gates is exactly the kind of thing
this repo has watched rot twice (`vlm_preferred`/`text_preferred` named models
nothing read; `conf/config.toml` named models that had been deleted from disk).
So the list below is a RECORD OF A MEASUREMENT, and `disagreements()` re-derives
the classification from any result set and reports where the record and the data
no longer agree. The record is the claim; the data is the check.
"""

from typing import Dict, Iterable, List

#: When the counts below were taken. One named constant rather than the same date
#: repeated in three docstrings, so a re-measurement updates it in one place.
MEASURED = "2026-08-19"  # check-ok: provenance in prose, not a value logic reads

#: Minimum distinct scores across models before a task is credited with ranking.
#: Two models scoring 100 and 0 is a gate that one model failed, not a ranking:
#: it sorts models into pass and fail, which is what a gate does. Three distinct
#: values is the first count that can order anything.
MIN_RANKING_VALUES = 3

#: How many models must have reported a task before its spread means anything.
#: Two models trivially produce at most two values, so any task looks like a gate.
MIN_MODELS_FOR_VERDICT = 4

#: Tasks measured as gates, with the evidence. NOT a policy list -- a record of
#: what was observed, checkable by `disagreements()` against any later run.
GATE_TASKS: Dict[str, str] = {
    "image_real": (
        f"2 distinct values over 8 complete runs ({MEASURED}). Proves a model can "
        "see and catches the transport silently dropping the image; cannot order "
        "the models that pass. Separating them needs harder images."
    ),
    "taxes_slip_qa": (
        f"2 distinct values over 8 complete runs ({MEASURED}), 7 of 8 at 100. Its "
        "empty-flags snapshot admits exactly one right answer, so every competent "
        "model reaches it. Worth keeping -- it catches a model that invents "
        "figures -- but it must never be counted on to rank."
    ),
}


def is_gate(task: str) -> bool:
    """Whether `task` is recorded as unable to rank."""
    return task in GATE_TASKS


def ranking_tasks(tasks: Iterable[str]) -> List[str]:
    """The subset of `tasks` that can order models."""
    return [t for t in tasks if not is_gate(t)]


def scores_by_task(all_results: List[Dict]) -> Dict[str, List[float]]:
    """Every model's score for each task, from a set of per-model records.

    Only COMPLETE runs contribute. A truncated run's absent tasks would otherwise
    read as a narrower spread and could reclassify a ranking task as a gate --
    the incomplete-run class again, one layer up. See eval/completeness.py.
    """
    from eval.completeness import is_complete

    by_task: Dict[str, List[float]] = {}
    for record in all_results or []:
        if not is_complete(record):
            continue
        for res in record.get("results", []) or []:
            task = res.get("task")
            if task is None:
                continue
            by_task.setdefault(task, []).append(res.get("quality_score", 0))
    return by_task


def distinct_values(all_results: List[Dict], task: str) -> int:
    """How many different scores `task` produced across models."""
    return len(set(scores_by_task(all_results).get(task, [])))


def classify(all_results: List[Dict]) -> Dict[str, str]:
    """Derive, from data alone, which tasks rank and which gate.

    Returns task -> "ranks" | "gate" | "unknown". "unknown" is not a hedge: with
    fewer than MIN_MODELS_FOR_VERDICT models reporting, a narrow spread is a
    property of the sample size and calling it a gate would be inventing a
    finding.
    """
    verdicts = {}
    for task, scores in scores_by_task(all_results).items():
        if len(scores) < MIN_MODELS_FOR_VERDICT:
            verdicts[task] = "unknown"
        elif len(set(scores)) >= MIN_RANKING_VALUES:
            verdicts[task] = "ranks"
        else:
            verdicts[task] = "gate"
    return verdicts


def disagreements(all_results: List[Dict]) -> List[str]:
    """Where the recorded classification and this run's data conflict.

    Two directions, and the second is the one that matters more:

    - a task recorded as a GATE that now ranks -- the record is stale, and a task
      that earned its place is being thrown away;
    - a task counted for RANKING that now behaves as a gate -- it is diluting
      every mean, which is the failure this module exists to stop.

    Reported rather than acted on. Reclassifying a task automatically from one
    run's data is how a single contended sweep silently rewrites what the suite
    measures.
    """
    found = []
    for task, verdict in classify(all_results).items():
        if verdict == "unknown":
            continue
        if verdict == "ranks" and is_gate(task):
            found.append(
                f"{task}: recorded as a GATE but produced "
                f"{distinct_values(all_results, task)} distinct values here -- "
                "it may have started ranking; re-check before trusting either."
            )
        elif verdict == "gate" and not is_gate(task):
            found.append(
                f"{task}: counted for RANKING but produced only "
                f"{distinct_values(all_results, task)} distinct values here -- "
                "it is diluting the mean without ordering anything."
            )
    return found


def ranking_mean(results: List[Dict]) -> float:
    """Mean over the tasks that can actually order models.

    Falls back to the full mean when a run contains ONLY gate tasks -- which is
    what `--task image_real` produces. Returning 0 there would report a model
    that scored 100 on the one task it was asked for as having failed.
    """
    scores = [
        res.get("quality_score", 0)
        for res in results or []
        if not is_gate(res.get("task"))
    ]
    if not scores:
        scores = [res.get("quality_score", 0) for res in results or []]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
