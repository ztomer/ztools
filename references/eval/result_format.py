"""Convert quality.py ScoreCards into the evaluator's result dicts.

Split out of eval/run.py for the 500-line limit. `eval.run` re-exports it, so
`from eval.run import _quality_results_to_eval_format` keeps working -- the same
shim pattern the other splits in this repo use.

A pure value transformer with no I/O, which is why it was the right thing to
lift: nothing else in run.py's model loop depends on it, and it depends on
nothing there.
"""

from __future__ import annotations

#: A composite at or above this is reported as a clean pass.
STATUS_OK_AT = 90
#: Below this the run is a failure rather than a partial result.
STATUS_PARTIAL_AT = 50


def _quality_results_to_eval_format(scorecards: list, model: str) -> list[dict]:
    """Convert quality.py ScoreCards to model_eval's result format."""
    results = []
    for sc in scorecards:
        failures = [f for d in sc.dimensions for f in d.failures]
        composite = sc.composite
        status = (
            "ok"
            if composite >= STATUS_OK_AT
            else ("partial" if composite >= STATUS_PARTIAL_AT else "fail")
        )
        results.append(
            {
                "task": sc.task,
                "case_id": sc.case_id,
                "status": status,
                "quality_score": round(composite, 1),
                "time": round(sc.elapsed, 1),
                "error": None,
                "failure_reason": "; ".join(failures) if failures else "",
                "failure_category": None,
                "failure_evidence": "",
                "result": {"model": model, "time": sc.elapsed, "content": sc.output},
            }
        )
    return results


__all__ = ["_quality_results_to_eval_format"]
