import time
from typing import List, Optional

from lib import osaurus_lib
from lib.config import Task, _safe_format_prompt, get_model_prompt
from lib.eval_data import ALL_TEST_CASES
from lib.quality_models import ScoreCard, TestCase, _str
from lib.quality_scorers import score_output
from lib.tui import FAIL, STEP, WARN

LLM_TIMEOUT = 600


def query_model(model: str, prompt: str, input_text: str, task: str) -> Optional[str]:
    try:
        filled = _safe_format_prompt(prompt, input_text)
        result = osaurus_lib.call(
            model=model,
            messages=[{"role": "user", "content": filled}],
            timeout=LLM_TIMEOUT,
            task=task,
        )
        return _str(result.get("content"))
    except Exception:
        return None


def query_model_direct(model: str, full_prompt: str) -> Optional[str]:
    try:
        result = osaurus_lib.call(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            timeout=LLM_TIMEOUT,
        )
        return _str(result.get("content"))
    except Exception:
        return None


def run_suite(
    models: List[str], cases: List[TestCase] = None, verbose: bool = True
) -> List[ScoreCard]:
    if cases is None:
        cases = ALL_TEST_CASES

    results = []

    for i, model in enumerate(models):
        for j, case in enumerate(cases):
            if verbose:
                print(
                    f"  {STEP} {model[:30]:30s} {case.task:12s} {case.description}",
                    end=" ",
                    flush=True,
                )

            t0 = time.time()

            if case.task in ("weekend_transient", "weekend_fixed"):
                output = query_model_direct(model, case.input_text)
            else:
                prompt = get_model_prompt(model, Task(case.task))
                if not prompt:
                    if verbose:
                        print(f"{STEP} skip")
                    continue
                output = query_model(model, prompt, case.input_text, case.task)

            elapsed = time.time() - t0

            if output is None:
                if verbose:
                    print(FAIL)
                results.append(
                    ScoreCard(
                        model=model,
                        task=case.task,
                        case_id=case.description,
                        dimensions=[],
                        output="",
                        elapsed=elapsed,
                    )
                )
                continue

            sc = score_output(output, case.task, case)
            sc.model = model
            sc.elapsed = elapsed
            results.append(sc)

            if verbose:
                comp = sc.composite
                if sc.dimensions:
                    worst = min(d.score for d in sc.dimensions)
                    prefix = STEP if worst >= 60 else (WARN if comp >= 40 else FAIL)
                    failures = [d.failures for d in sc.dimensions if d.failures]
                    all_fails = [f for dim_fails in failures for f in dim_fails]
                    fail_str = f" [{'; '.join(all_fails)}]" if all_fails else ""
                    print(f"{prefix}  {comp:5.1f}%  ({elapsed:.1f}s){fail_str}")
                else:
                    print(f"{FAIL}  0.0%")

    return results
