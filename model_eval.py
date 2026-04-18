#!/usr/bin/env python3
"""
Model Evaluator - Test models on REAL-WORLD tasks from ZTools.
Evaluates local models against the actual prompts used in the tools.
"""

import sys
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
)

console = Console()

# ==========================================================
# REAL-WORLD PROMPTS FROM ZTOOLS
# ==========================================================

# 1. Weekend Planner: Transient Events Prompt
WEEKEND_SYS_TRANSIENT = """
Act as a data-extraction agent for family events in Vaughan/Toronto.
You must output ONLY valid JSON. Do not include markdown formatting or conversational text.

Constraints:
- Target: 13yo girl, 10yo boy, 6yo boy. Must accommodate all simultaneously or in parallel zones.
- Weather Logic: Check the daily forecast. Only recommend outdoor events on days where the weather is 'Clear'. If a day has precipitation, events for that day MUST be indoor.
- Temporal Logic: Verify the temporal logic of the events provided. Reject events tied to holidays or seasons that do not occur during the provided dates.

Expected JSON Schema:
{
    "transient_events": [
    {"name": "...", "location": "...", "target_ages": "...", "price": "...", "duration": "...", "weather": "...", "day": "..."}
    ]
}
Extract up to 10 valid events from the provided context.
"""
WEEKEND_USR_TRANSIENT = """
Current Context for the upcoming weekend:
Dates: April 20 to April 22, 2026
Friday: 15.0°C, Clear (0mm)
Saturday: 12.0°C, Precipitation (5mm)
Sunday: 14.0°C, Clear (0mm)

High-Signal Transient Events (Filter these strictly! Ensure they match the Dates provided!):
- Spring Festival at Downsview Park: Outdoor rides and games. April 20-22.
- Indoor Coding Workshop for Kids: Learn Python. April 21.
- Outdoor Movie Night: Watch a movie under the stars. April 21.

Execute the task based on the system instructions and the provided context. Output ONLY JSON.
"""

# 2. Weekend Planner: Fixed Venues Prompt
WEEKEND_SYS_FIXED = """
Act as a creative planning agent for family activities in Vaughan/Toronto.
You must output ONLY valid JSON. Do not include markdown formatting or conversational text.

Constraints:
- Target: 13yo girl, 10yo boy, 6yo boy. Must accommodate all simultaneously.
- Exclude: The Art of the Brick, Reptilia, ROM, Ripley's, Little Canada, LEGOLAND, CN Tower, Museum of Illusions, Canada's Wonderland, Ontario Science Centre, Toronto Zoo.
- Weather Logic: Check the daily forecast. Only recommend outdoor activities on days where the weather is 'Clear'. If a day has precipitation, activities for that day MUST be indoor.
- Diversity: Provide a random, diverse mix of 10 year-round places to visit to ensure randomization and discovery of new places.

Expected JSON Schema:
{
    "fixed_activities": [
    {"name": "...", "location": "...", "target_ages": "...", "price": "...", "weather": "..."}
    ]
}
Extract exactly 10 valid activities.
"""
WEEKEND_USR_FIXED = """
Current Context for the upcoming weekend:
Dates: April 20 to April 22, 2026
Friday: 15.0°C, Clear (0mm)
Saturday: 12.0°C, Precipitation (5mm)
Sunday: 14.0°C, Clear (0mm)

Potential Venues and Current Exhibits:
- Vaughan Sports Arena: Indoor trampoline and dodgeball.
- High Park: Large outdoor playground and zoo.
- Aga Khan Museum: Islamic art and culture (indoor).

Execute the task based on the system instructions and the provided context to find 10 year-round fixed activities, prioritizing current exhibits or highly-rated venues from the context. Output ONLY JSON.
"""

# 3. File Renamer Prompt
RENAME_PROMPT = """You are a file naming assistant. Read the following text extracted from an image and suggest a short, descriptive filename (without extension). Use lowercase, underscores for spaces, and no special characters other than hyphens/underscores. Keep it under 50 characters. Do NOT include any reasoning, thinking process, or introductory text. Output ONLY the final filename string, nothing else.

TEXT:
CONFIDENTIAL - Q3 2025 Financial Results & Board Meeting Minutes"""

# 4. Twitter Summarizer Prompt
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
</formatting_rules>

<timeline>
[@TechCrunch | 09:15]: OpenAI announces new GPT-5 model with reasoning capabilities.
[@TheVerge | 09:20]: Microsoft integrates GPT-5 into Copilot across Office 365.
[@LocalNews | 10:00]: Massive traffic jam on Highway 401 due to construction.
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
    },
    "detailed_json": {
        "messages": [
            {"role": "system", "content": WEEKEND_SYS_FIXED},
            {"role": "user", "content": WEEKEND_USR_FIXED},
        ],
        "validator": validate_detailed_json,
        "parse_json": True,
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

def run_eval(
    model: str, tasks: dict = None, host: str = "localhost", port: int = 1337
) -> dict:
    """Run evaluation on model using real-world tasks."""
    tasks = tasks or TASKS
    results = []

    console.print(f"[cyan]Testing {model}...[/cyan]")

    for task_name, task in tasks.items():
        result = call(
            model=model,
            messages=task["messages"],
            host=host,
            port=port,
            task=task_name,
            parse_json=task["parse_json"],
            validator=task["validator"],
        )

        score = result.get("quality_score", 0)
        status = "ok" if score >= 90 else ("partial" if score >= 50 else "fail")

        results.append(
            {
                "task": task_name,
                "status": status,
                "quality_score": score,
                "time": result.get("time"),
                "error": result.get("error"),
            }
        )

        status_symbol = "✅" if status == "ok" else ("⚠️" if status == "partial" else "❌")
        failure = result.get("failure_reason")
        fail_info = f" - {failure}" if failure else ""
        console.print(
            f"  {status_symbol} {task_name}: {score}% ({result.get('time', 0)}s){fail_info}"
        )

    return results

def main():
    init_config()
    if not is_server_running():
        console.print("[red]Error: Osaurus server not running[/red]")
        sys.exit(1)

    models = get_models()
    if not models:
        console.print("[red]No models found[/red]")
        sys.exit(1)

    console.print(f"[green]Found {len(models)} models[/green]")

    for model in models:
        results = run_eval(model)
        scores = [r["quality_score"] for r in results]
        avg = sum(scores) / len(scores) if scores else 0
        console.print(f"[bold]{model}: {avg:.0f}%[/bold]\n")

if __name__ == "__main__":
    main()
