# Analysis and reporting functions

from typing import List, Dict
from rich.console import Console
from pathlib import Path
import json
import time


console = Console()


# ===================== Analysis Functions =====================


def print_cross_model_comparison(all_results: list) -> None:
    """Print comparison table across all models."""
    if not all_results:
        return
    
    console.print("")
    console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print("[bold cyan]Cross-Model Comparison")
    console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    models = [r["model"] for r in all_results]
    if not models:
        return
    
    first_results = all_results[0].get("results", [])
    if not first_results:
        return
    
    tasks = [res["task"] for res in first_results]
    if not tasks:
        return
    
    header = f"{'Task':<20}"
    for m in models:
        header += f" | {m[:15]:>15}"
    console.print(header)
    console.print("-" * len(header))
    
    for task in tasks:
        row = f"{task:<20}"
        task_scores = {}
        for r in all_results:
            score = 0
            for res in r.get("results", []):
                if res.get("task") == task:
                    score = res.get("quality_score", 0)
                    break
            task_scores[r["model"]] = score
            row += f" | {score:>15}"
        
        best_model = max(task_scores, key=task_scores.get)
        best_score = task_scores[best_model]
        row += f" | {best_score:>4}*"
        console.print(row)


def compute_score_stats(all_results: list) -> dict:
    """Compute aggregate statistics for each model."""
    stats = {}
    
    for r in all_results:
        model = r["model"]
        scores = [res.get("quality_score", 0) for res in r.get("results", [])]
        
        if not scores:
            continue
        
        import statistics
        stats[model] = {
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "stdev": statistics.stdev(scores) if len(scores) > 1 else 0,
            "min": min(scores),
            "max": max(scores),
            "count": len(scores),
        }
    
    return stats


def print_score_stats(stats: dict) -> None:
    """Print score statistics table."""
    if not stats:
        return
    
    console.print("")
    console.print("[bold]Score Statistics:[/bold]")
    console.print(f"{'Model':<20} | {'Mean':>6} | {'Med':>6} | {'Stdev':>6} | {'Min':>4} | {'Max':>4}")
    console.print("-" * 65)
    
    for model, s in sorted(stats.items(), key=lambda x: x[1]["mean"], reverse=True):
        console.print(
            f"{model:<20} | {s['mean']:>6.1f} | {s['median']:>6.1f} | "
            f"{s['stdev']:>6.1f} | {s['min']:>4} | {s['max']:>4}"
        )


def categorize_failures(all_results: list) -> dict:
    """Group failures by category across all models."""
    categories = {}
    
    for r in all_results:
        for res in r.get("results", []):
            if res.get("quality_score", 0) >= 90:
                continue
            
            cat = res.get("failure_category", "UNKNOWN")
            if cat not in categories:
                categories[cat] = {"count": 0, "models": set(), "tasks": set()}
            
            categories[cat]["count"] += 1
            categories[cat]["models"].add(r["model"])
            categories[cat]["tasks"].add(res.get("task"))
    
    for cat in categories:
        categories[cat]["models"] = list(categories[cat]["models"])
        categories[cat]["tasks"] = list(categories[cat]["tasks"])
    
    return categories


def print_failure_summary(categories: dict) -> None:
    """Print failure categorization summary."""
    if not categories:
        return
    
    console.print("")
    console.print("[bold]Failure Categories:[/bold]")
    
    for cat, info in sorted(categories.items(), key=lambda x: x[1]["count"], reverse=True):
        count = info["count"]
        models = ", ".join(info["models"][:3])
        tasks = ", ".join(info["tasks"][:2])
        console.print(f"  [{count}] {cat}: {models} ({tasks})")


def compute_verbosity(all_results: list) -> dict:
    """Compute avg response length per task per model."""
    verbosity = {}
    
    for r in all_results:
        model = r["model"]
        verbosity[model] = {}
        
        for res in r.get("results", []):
            task = res.get("task")
            content = res.get("result", {}).get("content", "")
            length = len(content) if content else 0
            verbosity[model][task] = length
    
    return verbosity


def print_verbosity(verbosity: dict) -> None:
    """Print response length per task."""
    if not verbosity:
        return
    
    console.print("")
    console.print("[bold]Response Verbosity (chars):[/bold]")
    
    first_model = next(iter(verbosity.keys()))
    tasks = verbosity[first_model].keys()
    
    header = f"{'Task':<20}"
    for m in verbosity.keys():
        header += f" | {m[:12]:>12}"
    console.print(header)
    console.print("-" * len(header))
    
    for task in tasks:
        row = f"{task:<20}"
        for model, task_lengths in verbosity.items():
            length = task_lengths.get(task, 0)
            row += f" | {length:>12,}"
        console.print(row)


def compute_error_rates(all_results: list) -> dict:
    """Compute error rates: infra errors vs quality failures."""
    rates = {}
    
    for r in all_results:
        model = r["model"]
        infra = 0
        quality = 0
        success = 0
        
        for res in r.get("results", []):
            category = res.get("failure_category", "OK")
            error = res.get("error")
            
            if error or category == "INFRA":
                infra += 1
            elif res.get("quality_score", 0) < 50:
                quality += 1
            else:
                success += 1
        
        total = infra + quality + success
        rates[model] = {
            "infra": infra,
            "quality": quality,
            "success": success,
            "infra_rate": infra / total if total else 0,
            "quality_rate": quality / total if total else 0,
            "success_rate": success / total if total else 0,
        }
    
    return rates


def print_error_rates(rates: dict) -> None:
    """Print error rate breakdown."""
    if not rates:
        return
    
    console.print("")
    console.print("[bold]Error Rates:[/bold]")
    console.print(f"{'Model':<20} | {'Infra':>6} | {'Quality':>8} | {'Success':>8} | {'Rate'}")
    console.print("-" * 65)
    
    for model, r in sorted(rates.items(), key=lambda x: x[1]["success_rate"], reverse=True):
        rate = r["success_rate"] * 100
        console.print(
            f"{model:<20} | {r['infra']:>6} | {r['quality']:>8} | "
            f"{r['success']:>8} | {rate:>5.0f}%"
        )


def compute_task_winners(all_results: list) -> dict:
    """Find which model wins each task."""
    winners = {}
    
    for r in all_results:
        for res in r.get("results", []):
            task = res.get("task")
            score = res.get("quality_score", 0)
            
            if task not in winners or score > winners[task][1]:
                winners[task] = (r["model"], score)
    
    return winners


def diff_from_last_run(all_results: list) -> dict:
    """Compare current scores to last run for each model."""
    eval_dir = Path("~/.config/ztools").expanduser()
    prev_file = eval_dir / "eval_results.json"
    
    if not prev_file.exists():
        return {}
    
    try:
        with open(prev_file) as f:
            prev_data = json.load(f)
    except:
        return {}
    
    prev_results = prev_data.get("models", [])
    if not prev_results:
        return {}
    
    diffs = {}
    for r in all_results:
        model = r["model"]
        prev_model_data = next((p for p in prev_results if p.get("model") == model), None)
        
        if not prev_model_data:
            continue
        
        diffs[model] = {}
        
        for res in r.get("results", []):
            task = res.get("task")
            score = res.get("quality_score", 0)
            
            prev_score = 0
            for p in prev_model_data.get("results", []):
                if p.get("task") == task:
                    prev_score = p.get("quality_score", 0)
                    break
            
            diff = score - prev_score
            if diff != 0:
                diffs[model][task] = {"current": score, "prev": prev_score, "diff": diff}
    
    return diffs


def print_diff(diffs: dict) -> None:
    """Print score changes from last run."""
    if not diffs:
        return
    
    console.print("")
    console.print("[bold]Changes from Last Run:[/bold]")
    
    has_changes = False
    for model, changes in diffs.items():
        for task, d in changes.items():
            if d.get("diff", 0) != 0:
                has_changes = True
    
    if not has_changes:
        console.print("  (no changes)")
        return
    
    console.print(f"{'Model':<18} | {'Task':<18} | {'Prev':>4} | {'Now':>4} | {'Diff'}")
    console.print("-" * 60)
    
    for model, changes in diffs.items():
        for task, d in changes.items():
            diff = d.get("diff", 0)
            if diff != 0:
                arrow = "↑" if diff > 0 else "↓"
                console.print(
                    f"{model[:18]:<18} | {task[:18]:<18} | "
                    f"{d['prev']:>4} | {d['current']:>4} | {arrow}{abs(diff):>3}"
                )


def export_to_csv(all_results: list, output_file: str = None) -> None:
    """Export results to CSV for reporting."""
    if output_file is None:
        eval_dir = Path("~/.config/ztools").expanduser()
        output_file = eval_dir / "eval_results.csv"
    else:
        output_file = Path(output_file)
    
    import csv
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Task", "Score", "Status", "Time(s)", "Failure", "Failure_Category"])
        
        for r in all_results:
            model = r["model"]
            for res in r.get("results", []):
                score = res.get("quality_score", 0)
                status = "PASS" if score >= 90 else ("WARN" if score >= 50 else "FAIL")
                time_s = res.get("time", "")
                failure = res.get("failure_reason", "")
                category = res.get("failure_category", "")
                
                writer.writerow([model, res.get("task"), score, status, time_s, failure, category])
    
    console.print(f"[green]Exported to {output_file}[/green]")


# ===================== Historical Tracking =====================


def save_historical_results(all_results: list, stats: dict, categories: dict) -> None:
    """Save per-model scores that persist even when models change."""
    eval_dir = Path("~/.config/ztools").expanduser()
    eval_dir.mkdir(parents=True, exist_ok=True)
    history_file = eval_dir / "eval_history.json"
    
    history = {}
    if history_file.exists():
        try:
            with open(history_file) as f:
                history = json.load(f)
        except:
            pass
    
    for r in all_results:
        model = r["model"]
        if model not in history:
            history[model] = []
        
        for res in r.get("results", []):
            entry = {
                "date": time.strftime("%Y-%m-%d"),
                "timestamp": time.time(),
                "task": res.get("task"),
                "score": res.get("quality_score", 0),
                "time": res.get("time"),
            }
            history[model].append(entry)
    
    for model in history:
        history[model] = history[model][-100:]
    
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)


def load_historical_stats() -> dict:
    """Load per-model historical scores."""
    eval_dir = Path("~/.config/ztools").expanduser()
    history_file = eval_dir / "eval_history.json"
    
    if not history_file.exists():
        return {}
    
    try:
        with open(history_file) as f:
            history = json.load(f)
    except:
        return {}
    
    if not history:
        return {}
    
    stats = {}
    import statistics
    for model, entries in history.items():
        if not entries:
            continue
        
        scores = [e["score"] for e in entries if e.get("score")]
        if scores:
            stats[model] = {
                "mean": statistics.mean(scores),
                "median": statistics.median(scores),
                "stdev": statistics.stdev(scores) if len(scores) > 1 else 0,
                "min": min(scores),
                "max": max(scores),
                "runs": len(entries),
            }
    
    return stats


def check_model_history(model: str) -> dict:
    """Check if model has historical data."""
    eval_dir = Path("~/.config/ztools").expanduser()
    history_file = eval_dir / "eval_history.json"
    
    if not history_file.exists():
        return {}
    
    try:
        with open(history_file) as f:
            history = json.load(f)
    except:
        return {}
    
    return history.get(model, [])


def print_historical_trends() -> None:
    """Print historical score trends per model."""
    stats = load_historical_stats()
    if not stats:
        return
    
    console.print("")
    console.print("[bold]Historical Trends:[/bold]")
    console.print(f"{'Model':<20} | {'Runs':>5} | {'Mean':>6} | {'Stdev':>6} | {'Trend'}")
    console.print("-" * 60)
    
    for model, s in sorted(stats.items(), key=lambda x: x[1]["mean"], reverse=True):
        runs = s.get("runs", 0)
        mean = s.get("mean", 0)
        stdev = s.get("stdev", 0)
        
        if runs >= 3:
            if stdev < 5:
                trend = "stable"
            elif stdev < 15:
                trend = "variable"
            else:
                trend = "unstable"
        else:
            trend = "new"
        
        console.print(f"{model:<20} | {runs:>5} | {mean:>6.0f} | {stdev:>6.1f} | {trend}")