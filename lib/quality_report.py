import json
from pathlib import Path
from typing import Dict, List

from lib.quality_models import ScoreCard


BASELINE_PATH = Path(__file__).resolve().parent.parent / "docs" / "eval_baseline.json"


def generate_report(results: List[ScoreCard]) -> str:
    by_model: Dict[str, List[ScoreCard]] = {}
    for sc in results:
        by_model.setdefault(sc.model, []).append(sc)

    lines = []

    for model in sorted(by_model.keys()):
        cards = by_model[model]
        lines.append(f"\n  {model}")

        by_task: Dict[str, List[ScoreCard]] = {}
        for sc in cards:
            by_task.setdefault(sc.task, []).append(sc)

        for task in ["filename", "summarize", "file_summary"]:
            task_cards = by_task.get(task, [])
            if not task_cards:
                continue

            avg_dim = {}
            for sc in task_cards:
                for d in sc.dimensions:
                    avg_dim.setdefault(d.name, []).append(d.score)

            dim_avgs = {name: sum(scores)/len(scores)
                        for name, scores in avg_dim.items()}

            composites = [sc.composite for sc in task_cards]
            avg_comp = sum(composites) / len(composites)

            lines.append(f"\n    {task}:")
            for name, avg in sorted(dim_avgs.items()):
                lines.append(f"      {name:18s} {avg:5.1f}%")
            lines.append(f"      {'Composite':18s} {avg_comp:5.1f}%")

            times = [sc.elapsed for sc in task_cards]
            avg_time = sum(times) / len(times) if times else 0
            lines.append(f"      {'Avg time':18s} {avg_time:5.1f}s  ({len(times)} cases)")

    lines.append("")
    lines.append(f"  {'Model':35s} {'Filename':>9} {'Summarize':>11} {'FileSum':>9} {'Speed':>7} {'Fail':>5}")
    lines.append(f"  {'-'*35} {'-'*9} {'-'*11} {'-'*9} {'-'*7} {'-'*5}")

    for model in sorted(by_model.keys()):
        cards = by_model[model]
        task_avgs = {}
        task_times = {}
        failures = 0
        for sc in cards:
            task_avgs.setdefault(sc.task, []).append(sc.composite)
            task_times.setdefault(sc.task, []).append(sc.elapsed)
            for d in sc.dimensions:
                if d.failures:
                    failures += 1

        def avg(vals):
            return sum(vals) / len(vals) if vals else 0

        f_avg = avg(task_avgs.get("filename", []))
        s_avg = avg(task_avgs.get("summarize", []))
        fs_avg = avg(task_avgs.get("file_summary", []))
        all_times = [t for times in task_times.values() for t in times]
        speed = avg(all_times) if all_times else 0

        lines.append(
            f"  {model:35s} {f_avg:8.1f}% {s_avg:10.1f}% {fs_avg:8.1f}% "
            f"{speed:6.1f}s {failures:5}"
        )

    return "\n".join(lines)


def _model_task_key(model: str, task: str, case_id: str) -> str:
    return f"{model}::{task}::{case_id}"


def save_baseline(results: List[ScoreCard]):
    baseline = {}
    for sc in results:
        key = _model_task_key(sc.model, sc.task, sc.case_id)
        baseline[key] = {
            "composite": sc.composite,
            "dimensions": {d.name: d.score for d in sc.dimensions},
            "elapsed": sc.elapsed,
        }

    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=2)
    return baseline


def load_baseline() -> dict:
    if not BASELINE_PATH.exists():
        return {}
    with open(BASELINE_PATH) as f:
        return json.load(f)


def compare_to_baseline(results: List[ScoreCard]) -> List[str]:
    baseline = load_baseline()
    if not baseline:
        return ["  No baseline found. Run with --save-baseline to create one."]

    warnings = []
    for sc in results:
        key = _model_task_key(sc.model, sc.task, sc.case_id)
        prev = baseline.get(key)
        if not prev:
            continue

        curr_comp = sc.composite
        prev_comp = prev["composite"]
        delta = curr_comp - prev_comp

        if delta < -10:
            dim_deltas = []
            for d in sc.dimensions:
                prev_d = prev.get("dimensions", {}).get(d.name, 0)
                dd = d.score - prev_d
                if dd < -15:
                    dim_deltas.append(f"{d.name}: {prev_d:.0f}→{d.score:.0f} ({dd:+.0f})")
            detail = f" [{'; '.join(dim_deltas)}]" if dim_deltas else ""
            warnings.append(
                f"  ⚠ REGRESSION: {sc.model} / {sc.task} / {sc.case_id}\n"
                f"    {prev_comp:.1f}% → {curr_comp:.1f}% ({delta:+.1f}pts){detail}"
            )
        elif delta > 10:
            warnings.append(
                f"  ↑ IMPROVEMENT: {sc.model} / {sc.task} / {sc.case_id}\n"
                f"    {prev_comp:.1f}% → {curr_comp:.1f}% ({delta:+.1f}pts)"
            )

    return warnings
