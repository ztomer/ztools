#!/usr/bin/env python3
"""
Model Evaluator - Test models on REAL-WORLD tasks from ZTools.
Evaluates local models against the actual prompts used in the tools.
"""

import sys
import argparse
from rich.console import Console

from lib import init_config
from lib.osaurus_lib import (
    call,
    get_models,
    is_server_running,
)

from lib.validators_lib import (
    validate_detailed_json,
    validate_summary,
    validate_filename,
    has_item_details,
)

console = Console()

# ==========================================================
# REAL-WORLD PROMPTS FROM ZTOOLS
# ==========================================================

WEEKEND_SYS_TRANSIENT = """
Extract JSON from context. 10 items with name, location, target_ages, price, duration, weather, day.

Default values if not in context:
- target_ages: "6-13 years"
- price: $20-30 or Free
- duration: "2-3 hours"
- weather: "indoor"
- day: Friday/Saturday/Sunday
"""

WEEKEND_SYS_FIXED = """
Extract JSON from context. 10 items with name, location, target_ages, price, weather.

Default values if not in context:
- target_ages: "6-13 years"
- price: $20-30 or Free
- weather: "indoor"
- location: city name
"""
WEEKEND_USR_TRANSIENT = """
Current Context for the upcoming weekend:
Dates: April 20 to April 22, 2026
Friday: 15.0°C, Clear (0mm)
Saturday: 12.0°C, Precipitation (5mm)
Sunday: 14.0°C, Clear (0mm)

High-Signal Transient Events (Filter these strictly! Ensure they match the Dates provided!):
- Spring Festival at Downsview Park: Outdoor rides and games. April 20-22. All ages.
- Indoor Coding Workshop for Kids: Learn Python. April 21. Ages 8-14.
- Outdoor Movie Night: Watch a movie under the stars. April 21. All ages.
- Farmers Market at Maple Village: Fresh produce and local crafts. April 20. All ages.
- Pottery Wheel Workshop: Create clay art. April 22. Ages 12+.
- Puppet Show at Vaughan Library: "The Magical Forest". April 20. Ages 4-10.
- Kids Yoga in the Park: Morning yoga for families. April 20. Ages 5-12.
- Magic Show at Markham Theatre: Illusionist show. April 21. All ages.
- Nature Walk at Boyd Conservation: guided family hike. April 22. All ages.
- Board Game Marathon at Community Centre: Family games. April 21. All ages.
- Pizza Making Class: Learn to make pizza. April 22. Ages 8-16.
- Easter Egg Hunt at Raccoon Creek: Egg hunt and crafts. April 20. Ages 3-10.
"""
WEEKEND_USR_FIXED = """
Current Context for the upcoming weekend:
Dates: April 20 to April 22, 2026
Friday: 15.0°C, Clear (0mm)
Saturday: 12.0°C, Precipitation (5mm)
Sunday: 14.0°C, Clear (0mm)

Potential Venues and Current Exhibits:
- Vaughan Sports Arena: Indoor trampoline and dodgeball. All ages.
- High Park: Large outdoor playground and zoo. All ages.
- Aga Khan Museum: Islamic art and culture. Indoor. All ages.
- McMichael Canadian Art Collection: Canadian art exhibits. Indoor. All ages.
- Gibson Park: Playground and splash pad. Outdoor. Ages 3-12.
- Richmond Hill Centre for the Performing Arts: Live theater. Indoor. All ages.
- Maplewood Park Conservation: Hiking trails and picnic area. Outdoor. All ages.
- Ezra Avenue Skatepark: Skateboarding and BMX. Outdoor. Ages 10+.
- Oakridge Arts Festival: Art installations and workshops. April 20-22. Indoor/outdoor. All ages.
- Toronto Fun Zone: Indoor play centre with wall climb. Indoor. Ages 4-14.
- Lake Simcoe Sugar Bush: Maple syrup tours. Outdoor. All ages.
- Markham Museum: Heritage buildings and events. Indoor/outdoor. All ages.

Execute the task based on the system instructions and the provided context to find 10 year-round fixed activities, prioritizing current exhibits or highly-rated venues from the context. Output ONLY JSON.
"""

RENAME_PROMPT = """You are a file naming assistant. Read the following text extracted from an image and suggest a short, descriptive filename (without extension). Use lowercase, underscores for spaces, and no special characters other than hyphens/underscores. Keep it under 50 characters. Do NOT include any reasoning, thinking process, or introductory text. Output ONLY the final filename string, nothing else.

TEXT:
CONFIDENTIAL - Q3 2025 Financial Results & Board Meeting Minutes"""

TWITTER_PROMPT = """You are an objective news distillation system. Your task is to extract hard facts from the provided chronological Twitter/X timeline.

<instructions>
1. First, analyze the timeline in your <think> block.
2. Identify clusters of related events and synthesize duplicates.
3. Output ONLY the final briefing after the </think> tag. No introductory text.
</instructions>

<formatting_rules>
- Use headers starting with ##
- Use bullet points for facts
- Keep it concise and factual
- 40 tweets to analyze
</formatting_rules>

<timeline>
[@TechCrunch | 08:00]: OpenAI announces GPT-5 with advanced reasoning capabilities, available next month.
[@TheVerge | 08:15]: Apple Vision Pro 2 enters mass production, expected fall release.
[@TechCrunch | 08:30]: Google unveils Gemini 2.5 Pro with 1M context window.
[@Wired | 08:45]: NVIDIA stock hits all-time high after data center revenue beats estimates.
[@Bloomberg | 09:00]: Federal Reserve signals potential rate cut in June meeting.
[@LocalNews_TOR | 09:15]: TTC subway Line 1 delays due to signal problems at Sheppard West.
[@TechCrunch | 09:30]: Microsoft acquires AI startup for $2B to boost Azure AI capabilities.
[@TheVerge | 09:45]: Samsung Galaxy S25 Ultra features new titanium frame and AI camera.
[@CNBC | 10:00]: Oil prices drop 3% on increased production concerns.
[@TorontoStar | 10:15]: Mayor announces new bike lane infrastructure for downtown Toronto.
[@TechCrunch | 10:30]: Meta announces Llama 4 open source with commercial license.
[@Wired | 10:45]: SpaceX successfully launches 60 Starlink satellites on Falcon 9.
[@LocalNews_TOR | 11:00]: Highway 401 collision causes 2-hour delays westbound near Jane Street.
[@TheVerge | 11:15]: Sony PlayStation 6 prototype leaks, features 8K gaming support.
[@TechCrunch | 11:30]: Anthropic launches Claude 4 with improved coding capabilities.
[@Bloomberg | 11:45]: Bitcoin surges past $75K on ETF approval news.
[@LocalNews_TOR | 12:00]: Toronto Maple Leafs win playoff game, celebrations in downtown core.
[@Wired | 12:15]: Amazon launches drone delivery in select Toronto neighborhoods.
[@TechCrunch | 12:30]: Adobe acquires Figma for $20B in largest tech acquisition of year.
[@TheVerge | 12:45]: Tesla Cybertruck production ramps up to 10K units per week.
[@CNBC | 13:00]: US jobs report shows 250K new jobs, beating expectations.
[@LocalNews_TOR | 13:15]: Pearson Airport reports record spring break travel volumes.
[@TechCrunch | 13:30]: IBM unveils quantum computer with 1000+ qubit capability.
[@Wired | 13:45]: Nintendo confirms new Switch model launching holiday season.
[@Bloomberg | 14:00]: Shopify reports 40% revenue growth, stock jumps 15%.
[@LocalNews_TOR | 14:15]: Ontario Place undergoing major renovation, new spa opening 2026.
[@TechCrunch | 14:30]: Salesforce announces AI-powered CRM with autonomous agents.
[@TheVerge | 14:45]: Intel Core Ultra chips debut with breakthrough efficiency.
[@CNBC | 15:00]: Housing market cools as mortgage rates stay elevated.
[@LocalNews_TOR | 15:15]: Raptors playoff game tonight at Scotiabank Arena, expect crowds.
[@TechCrunch | 15:30]: Databricks IPO values company at $60B, largest since Arm.
[@Wired | 15:45]: Apple announces carbon-neutral products by 2030.
[@Bloomberg | 16:00]: TD Bank reports strong Q2 earnings, beats analyst estimates.
[@LocalNews_TOR | 16:15]: Toronto Fire crews respond to warehouse fire in Leslieville area.
[@TechCrunch | 16:30]: Netflix launches live sports streaming with NBA games.
[@TheVerge | 16:45]: Google Pixel 9a launches at $599 with旗舰 AI features.
[@CNBC | 17:00]: Crypto regulation bill passes US Senate unanimously.
[@LocalNews_TOR | 17:15]: CN Tower reopens to visitors after maintenance.
[@TechCrunch | 17:30]: Uber launches autonomous taxi service in Phoenix.
[@Wired | 17:45]: Meta unveils holographic AR glasses prototype.
[@Bloomberg | 18:00]: Canadian GDP grows 0.5% in Q1, exceeding forecasts.
</timeline>

Provide the summary (start your response with <think>):"""

TASKS = {
    "json": {
        "messages": [
            {"role": "system", "content": WEEKEND_SYS_TRANSIENT},
            {"role": "user", "content": WEEKEND_USR_TRANSIENT},
        ],
        "validator": validate_detailed_json,
        "parse_json": True,
        "source": WEEKEND_USR_TRANSIENT,
    },
    "detailed_json": {
        "messages": [
            {"role": "system", "content": WEEKEND_SYS_FIXED},
            {"role": "user", "content": WEEKEND_USR_FIXED},
        ],
        "validator": validate_detailed_json,
        "parse_json": True,
        "source": WEEKEND_USR_FIXED,
    },
    "filename": {
        "messages": [
            {"role": "user", "content": RENAME_PROMPT},
        ],
        "validator": validate_filename,
        "parse_json": False,
    },
    "summarize": {
        "messages": [
            {"role": "user", "content": TWITTER_PROMPT},
        ],
        "validator": validate_summary,
        "parse_json": False,
    },
}

MAX_RETRIES = 1
EVAL_TIMEOUT = 600


# ==========================================================
# Helper to build tasks from model configs
# ==========================================================

def load_tasks_from_config(model: str):
    """Build task prompts from model config YAML."""
    from lib.config import get_model_prompts_all

    prompts = get_model_prompts_all(model)
    if not prompts:
        return None

    built = {}

    # Map config prompts to tasks
    if "weekend_fixed" in prompts:
        built["detailed_json"] = prompts["weekend_fixed"]
    if "weekend_transient" in prompts:
        built["json"] = prompts["weekend_transient"]
    if "summarize" in prompts:
        built["summarize"] = prompts["summarize"]
    if "filename" in prompts:
        built["filename"] = prompts["filename"]

    return built

# ==========================================================
# FAILURE DIAGNOSIS
# ==========================================================

# Failure categories — answers "WHY did the model fail?"
FAIL_INFRA   = "INFRA"    # Model not found, connection error
FAIL_TIMEOUT = "TIMEOUT"  # Model took too long to respond
FAIL_PARSE   = "PARSE"    # Model output had JSON-like content but our parser couldn't extract it
FAIL_FORMAT  = "FORMAT"   # Model ignored format instructions (e.g. output prose instead of JSON)
FAIL_CONTENT = "CONTENT"  # Model followed format but produced low-quality content
FAIL_NONE    = None       # No failure


def _classify_failure(result: dict, task_cfg: dict, score: int, failure_reason: str) -> dict:
    """Classify WHY a result failed. Returns a diagnosis dict.

    Returns:
        {
            "category": one of FAIL_* constants,
            "reason": human-readable failure reason,
            "evidence": specific evidence for the diagnosis,
        }
    """
    error = result.get("error") or ""
    content = result.get("content") or ""
    parsed = result.get("parsed")
    raw_len = len(content)

    # --- Perfect score: no diagnosis needed ---
    if score >= 90:
        return {"category": FAIL_NONE, "reason": "", "evidence": ""}

    # --- Infrastructure failures ---
    if "Model not found" in error:
        return {
            "category": FAIL_INFRA,
            "reason": error,
            "evidence": "Model not loaded or wrong identifier",
        }
    if "Connection" in error:
        return {
            "category": FAIL_INFRA,
            "reason": error,
            "evidence": "Server unreachable",
        }

    # --- Timeout ---
    if "Timeout" in error or "timed out" in error.lower():
        return {
            "category": FAIL_TIMEOUT,
            "reason": error,
            "evidence": f"Model did not respond within {EVAL_TIMEOUT}s",
        }

    # --- JSON tasks: distinguish PARSE vs FORMAT vs CONTENT ---
    if task_cfg["parse_json"]:
        has_json_chars = "{" in content or "[" in content
        has_prose_before_json = False
        if has_json_chars:
            first_bracket = min(
                (content.find(c) for c in "[{" if content.find(c) >= 0),
                default=raw_len,
            )
            has_prose_before_json = first_bracket > 200

        if not parsed and not has_json_chars:
            # Model output pure prose — didn't follow JSON format instruction
            return {
                "category": FAIL_FORMAT,
                "reason": failure_reason or "No JSON in output",
                "evidence": f"Output was {raw_len} chars of prose with no JSON brackets",
            }

        if not parsed and has_json_chars:
            # Model tried to output JSON but it didn't parse
            return {
                "category": FAIL_PARSE,
                "reason": failure_reason or "JSON extraction failed",
                "evidence": f"Output had JSON-like content at char {first_bracket} of {raw_len} but parser couldn't extract valid JSON",
            }

        if parsed and has_prose_before_json:
            # Parsed OK, but model emitted reasoning before JSON
            evidence = f"Model emitted {first_bracket} chars of reasoning before first JSON bracket (total {raw_len} chars)"
            if score < 50:
                return {
                    "category": FAIL_FORMAT,
                    "reason": failure_reason,
                    "evidence": evidence + "; reasoning may have consumed the context window",
                }
            # Score was partial (50-89) — content quality issue with format warning
            return {
                "category": FAIL_CONTENT,
                "reason": failure_reason,
                "evidence": evidence,
            }

        # JSON parsed fine — pure content quality issue
        return {
            "category": FAIL_CONTENT,
            "reason": failure_reason,
            "evidence": _describe_content_failure(parsed, failure_reason),
        }

    # --- Non-JSON tasks (filename, summarize) ---
    if not content:
        return {
            "category": FAIL_FORMAT,
            "reason": "Empty content",
            "evidence": "Model returned empty response",
        }

    # Check if model output reasoning instead of a direct answer
    reasoning_markers = ["Let me", "I'll", "Wait,", "Actually,", "Here's my", "Thinking"]
    has_reasoning = any(marker in content[:200] for marker in reasoning_markers)
    if has_reasoning and raw_len > 200:
        return {
            "category": FAIL_FORMAT,
            "reason": failure_reason,
            "evidence": f"Model output {raw_len} chars starting with reasoning instead of a direct answer",
        }

    return {
        "category": FAIL_CONTENT,
        "reason": failure_reason,
        "evidence": f"Output was {raw_len} chars but failed validation: {failure_reason}",
    }


def _describe_content_failure(parsed, failure_reason: str) -> str:
    """Generate human-readable evidence for content quality failures."""
    if isinstance(parsed, list):
        item_count = len(parsed)
        with_details = sum(
            1 for item in parsed
            if isinstance(item, dict) and has_item_details(item)
        )
        return f"Parsed {item_count} items, {with_details} with details. {failure_reason}"
    elif isinstance(parsed, dict):
        keys = list(parsed.keys())[:5]
        return f"Parsed dict with keys {keys}. {failure_reason}"
    return failure_reason


def _validate_result(result: dict, task_cfg: dict, task_name: str) -> tuple[int, str, dict]:
    """Run validation on a library result. Returns (score, failure_reason, diagnosis)."""
    validator = task_cfg["validator"]

    if result.get("error"):
        diagnosis = _classify_failure(result, task_cfg, 0, result["error"])
        return 0, result["error"], diagnosis

    if task_cfg["parse_json"]:
        parsed = result.get("parsed")
        if not parsed:
            failure = "Could not parse JSON from output"
            diagnosis = _classify_failure(result, task_cfg, 0, failure)
            return 0, failure, diagnosis

        # Run validator with source check for quality
        source = task_cfg.get("source", "")
        validated = validator(parsed, source_text=source)
    else:
        content = result.get("content") or ""
        if not content:
            failure = "Empty content"
            diagnosis = _classify_failure(result, task_cfg, 0, failure)
            return 0, failure, diagnosis
        validated = validator(content)

    if isinstance(validated, tuple):
        score, failure_reason = validated
    else:
        score, failure_reason = validated, ""

    diagnosis = _classify_failure(result, task_cfg, score, failure_reason)
    return score, failure_reason, diagnosis

def _call_model(model: str, task_cfg: dict, task_name: str, host: str, port: int, backend: str) -> dict:
    """Call model via the appropriate backend (pure transport, no validation)."""
    if backend == "mlx":
        from lib.mlx_lib import call as mlx_call
        return mlx_call(
            model=model,
            messages=task_cfg["messages"],
            task=task_name,
            parse_json=task_cfg["parse_json"],
        )
    else:
        return call(
            model=model,
            messages=task_cfg["messages"],
            host=host,
            port=port,
            task=task_name,
            parse_json=task_cfg["parse_json"],
            timeout=EVAL_TIMEOUT,
        )

def run_eval(
    model: str, tasks: dict = None, host: str = "localhost", port: int = 1337, backend: str = "osaurus"
) -> dict:
    """Run evaluation on model using real-world tasks.
    
    This function owns all validation and retry logic.
    The library call() functions are pure transport/parsing layers.
    """
    from lib.logging_config import osaurus_logger as eval_logger

    tasks = tasks or TASKS
    results = []

    console.print(f"[cyan]Testing {model} ({backend})...[/cyan]")

    for task_name, task_cfg in tasks.items():
        best_score = -1
        best_result = None
        best_failure = ""
        best_diagnosis = {"category": FAIL_NONE, "reason": "", "evidence": ""}
        
        for attempt in range(MAX_RETRIES + 1):
            if attempt > 0:
                eval_logger.warning(f"Retrying task '{task_name}' with model {model} (Attempt {attempt+1}/{MAX_RETRIES+1})...")
            
            result = _call_model(model, task_cfg, task_name, host, port, backend)
            score, failure_reason, diagnosis = _validate_result(result, task_cfg, task_name)
            
            eval_logger.info(f"Quality score: {score}/100")
            
            if score < 90:
                category = diagnosis.get("category", "")
                evidence = diagnosis.get("evidence", "")
                eval_logger.warning(
                    f"[DEBUG_OUTPUT] model={model} task={task_name} score={score} "
                    f"category={category} failure={failure_reason} "
                    f"evidence={evidence}"
                )

            if score > best_score:
                best_score = score
                best_result = result
                best_failure = failure_reason
                best_diagnosis = diagnosis
            
            if score >= 90:
                break

        status = "ok" if best_score >= 90 else ("partial" if best_score >= 50 else "fail")
        category = best_diagnosis.get("category")

        results.append(
            {
                "task": task_name,
                "status": status,
                "quality_score": best_score,
                "time": best_result.get("time") if best_result else None,
                "error": best_result.get("error") if best_result else None,
                "failure_reason": best_failure,
                "failure_category": category,
                "failure_evidence": best_diagnosis.get("evidence", ""),
            }
        )

        status_symbol = "✅" if status == "ok" else ("⚠️" if status == "partial" else "❌")
        category_tag = f" [{category}]" if category else ""
        fail_info = f" - {best_failure}" if best_failure else ""
        evidence_info = f"\n    ↳ {best_diagnosis['evidence']}" if best_diagnosis.get("evidence") else ""
        time_taken = best_result.get('time') if best_result else None
        time_taken_str = f"{time_taken}s" if time_taken is not None else "N/A"
        console.print(
            f"  {status_symbol} {task_name}: {best_score}% ({time_taken_str}){category_tag}{fail_info}{evidence_info}"
        )

    return results

def update_config(best_models: dict):
    import yaml
    from pathlib import Path
    config_path = Path("conf/config.yaml")
    if not config_path.exists():
        console.print("[yellow]Config file not found, skipping update.[/yellow]")
        return
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
        
    if "best_models" not in config:
        config["best_models"] = {}
        
    for task, model in best_models.items():
        if model:
            config["best_models"][task] = model
            if task == "detailed_json":
                pass
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
    console.print("[green]Updated conf/config.yaml with best models.[/green]")

def main():
    init_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Run evaluation for a specific model")
    parser.add_argument("--task", help="Run a specific task only (json, detailed_json, filename, summarize)")
    parser.add_argument("--quick", action="store_true", help="Quick mode: run single task with one retry (faster iteration)")
    parser.add_argument("--config-tasks", action="store_true", help="Load tasks from YAML config instead of hardcoded prompts")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging to console")
    args = parser.parse_args()

    models_to_test = []
    
    if is_server_running():
        osaurus_models = get_models()
        for m in osaurus_models:
            models_to_test.append((m, "osaurus"))
    else:
        console.print("[yellow]Warning: Osaurus server not running[/yellow]")

    # MLX backend disabled - not working reliably (2026-04)
    # Uncomment to re-enable:
    # from lib.mlx_lib import list_mlx_models, normalize_mlx_model_name
    # mlx_models = list_mlx_models()
    # osaurus_base_names = {normalize_mlx_model_name(m) for m in osaurus_models}
    # for m in mlx_models:
    #     normalized = normalize_mlx_model_name(m)
    #     if normalized in osaurus_base_names:
    #         models_to_test.append((m, "mlx"))

    if args.model:
        models_to_test = [(m, b) for m, b in models_to_test if m == args.model]

    if not models_to_test:
        console.print("[red]No models found[/red]")
        sys.exit(1)

    console.print(f"[green]Found {len(models_to_test)} models to test[/green]")

    # Filter to specific task if requested
    tasks_to_run = TASKS
    if args.task:
        if args.task not in TASKS:
            console.print(f"[red]Unknown task: {args.task}. Available: {list(TASKS.keys())}[/red]")
            sys.exit(1)
        tasks_to_run = {args.task: TASKS[args.task]}
        console.print(f"[yellow]Running only task: {args.task}[/yellow]")

    # Load tasks from YAML config
    if args.config_tasks:
        from lib.config import build_tasks_from_model

        config_model = args.model if args.model else "qwen"
        console.print(f"[yellow]Loading tasks from YAML config: {config_model}[/yellow]")
        config_tasks = build_tasks_from_model(config_model)
        if config_tasks:
            if args.task:
                if args.task in config_tasks:
                    tasks_to_run = {args.task: config_tasks[args.task]}
                else:
                    console.print(f"[red]Task '{args.task}' not in config[/red]")
            else:
                tasks_to_run = config_tasks
            console.print(f"[green]Loaded {len(tasks_to_run)} tasks from config[/green]")
        else:
            console.print("[red]No config tasks found, using default TASKS[/red]")

    # In quick mode, only run first task once (no retries)
    if args.quick:
        console.print(f"[yellow]Quick mode: single run, no retries[/yellow]")
        import model_eval as me
        original_run_eval = me.run_eval
        
        def quick_run_eval(model, backend="osaurus"):
            # Monkey-patch MAX_RETRIES temporarily
            import model_eval
            original_retries = model_eval.MAX_RETRIES
            model_eval.MAX_RETRIES = 0
            try:
                return original_run_eval(model, backend=backend)
            finally:
                model_eval.MAX_RETRIES = original_retries
        
        me.run_eval = quick_run_eval

    all_results = []
    best_scores = {task: -1 for task in tasks_to_run.keys()}
    best_models = {task: None for task in tasks_to_run.keys()}

    for model, backend in models_to_test:
        results = run_eval(model, tasks=tasks_to_run, backend=backend)
        scores = [r["quality_score"] for r in results]
        avg = sum(scores) / len(scores) if scores else 0
        console.print(f"[bold]{model} ({backend}): {avg:.0f}%[/bold]")
        
        # Print per-task scores
        for r in results:
            task = r["task"]
            score = r["quality_score"]
            status = "✅" if score >= 90 else ("⚠️" if score >= 50 else "❌")
            failure_info = f" ({r.get('failure_reason', '')})" if score < 90 else ""
            print(f"  {status} {task}: {score}{failure_info}")
        
        all_results.append({'model': model, 'backend': backend, 'results': results})
        for r in results:
            task = r["task"]
            score = r["quality_score"]
            if score > best_scores[task]:
                best_scores[task] = score
                best_models[task] = model

    console.print("[bold]Best Models per Task:[/bold]")
    for task, model in best_models.items():
        console.print(f"  {task}: {model} ({best_scores[task]}%)")


    import json
    with open("eval_results.json", "w") as f:
        json.dump({
            "models": all_results,
            "best_scores": best_scores,
            "best_models": best_models,
        }, f, indent=2)
    console.print("[green]Saved benchmark results to eval_results.json[/green]")

if __name__ == "__main__":
    main()
