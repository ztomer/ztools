#!/usr/bin/env python3
import os
import sys
import datetime
import time
import json
import yaml
import re
import concurrent.futures
import requests
from ddgs import DDGS
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from lib import init_config
from lib.osaurus_lib import (
    restart_server,
    get_best_model,
    is_server_running,
    call_llm_api,
    strip_thinking,
    panic_dump,
    get_available_models,
    select_best_model,
)

# MLX support
from lib.mlx_lib import (
    find_text_mlx_model,
    call_mlx_text,
    process_mlx_content,
)

# ==========================================
# CONFIGURATION
# ==========================================

console = Console(force_terminal=True, force_interactive=True)


def load_weekend_config():
    from pathlib import Path
    config_path = Path(__file__).parent / "conf" / "weekend.yaml"
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load weekend.yaml: {e}")
        return {}

WEEKEND_CONFIG = load_weekend_config()
VISITED_PLACES = WEEKEND_CONFIG.get("visited_places", [])
EXCLUDED_PLACES = WEEKEND_CONFIG.get("excluded_places", [])
CHILDREN = WEEKEND_CONFIG.get("children", [])
CHILDREN_STR = ", ".join([f"{c['age']}yo {c['gender']}" for c in CHILDREN]) if CHILDREN else "{CHILDREN_STR}"
CITY = WEEKEND_CONFIG.get("location", {}).get("city", "Vaughan")
REGION = WEEKEND_CONFIG.get("location", {}).get("region", "Toronto")


MODEL_CONFIG = os.path.expanduser("~/.config/model_eval.json")


# Use consolidated functions from osaurus_lib
MODEL_NAME = os.environ.get(
    "OLLAMA_MODEL", get_best_model() or "gemma-4-26b-a4b-it-4bit"
)
OSAURUS_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:1337")

OSAURUS_APP = "/Applications/osaurus.app"


# Use consolidated server management functions from osaurus_lib
# is_server_running_ours, restart_osaurus, ensure_server - all in osaurus_lib.py


def is_server_running_ours():
    """Check if osaurus server is running (wrapper for compatibility)."""
    return is_server_running()


def restart_osaurus(wait=20):
    """Restart Osaurus app (wrapper for compatibility)."""
    return restart_server(app_path=OSAURUS_APP, wait=wait)


def ensure_server(max_retries=3, wait=20):
    """Ensure server is running (wrapper for compatibility)."""
    # Import here to avoid circular dependency
    from lib.osaurus_lib import ensure_server as osaurus_ensure_server
    return osaurus_ensure_server(max_retries=max_retries, wait=wait)


# ==========================================
# DATA RETRIEVAL (Deterministic & Bounded)
# ==========================================


def get_weekend_date_objects():
    today = datetime.date.today()
    friday = today + datetime.timedelta((4 - today.weekday()) % 7)
    sunday = friday + datetime.timedelta(days=2)
    return friday, sunday


def get_weekend_dates_string(friday, sunday):
    return f"{friday.strftime('%B %d')} to {sunday.strftime('%B %d, %Y')}"


def fetch_weather(friday, sunday):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": 43.8361,
            "longitude": -79.5083,
            "daily": "temperature_2m_max,precipitation_sum",
            "timezone": "America/Toronto",
            "start_date": friday.strftime("%Y-%m-%d"),
            "end_date": sunday.strftime("%Y-%m-%d"),
        }
        resp = requests.get(url, params=params, timeout=10).json()
        daily = resp.get("daily", {})

        dates = daily.get("time", [])
        precip_array = daily.get("precipitation_sum", [])
        temp_array = daily.get("temperature_2m_max", [])

        forecasts = []
        for i in range(len(dates)):
            date_str = dates[i]
            precip = precip_array[i] if i < len(precip_array) else 0
            temp = temp_array[i] if i < len(temp_array) else 0
            condition = "Precipitation" if precip > 0.5 else "Clear"
            day_name = datetime.datetime.strptime(
                date_str, "%Y-%m-%d").strftime("%A")
            forecasts.append(
                f"{day_name}: {temp:.1f}°C, {condition} ({precip}mm)")

        return "Daily Forecast:\n" + "\n".join(forecasts)
    except Exception as e:
        print(f"[ERROR] Weather fetch failed: {e}", file=sys.stderr)
        return "Forecast: Precipitation expected (fallback due to error)."


def fetch_transient_events(dates_str, year, month_name):
    try:
        print("[DEBUG] Running query 1...", flush=True)
        queries = [
            f'(site:todocanada.ca OR site:blogto.com OR site:toronto.com) "family" "events" "{month_name} {year}"',
            f'"Vaughan" "kids events" "{month_name} {year}"',
            f'"Toronto" "family weekend festivals" "{month_name} {year}"',
        ]

        all_results = []
        for i, q in enumerate(queries):
            print(f"[DEBUG] Query {i + 1}...", flush=True)
            results = list(DDGS().text(q, max_results=10))
            print(f"[DEBUG] Query {i + 1} done: {len(results)}", flush=True)
            all_results.extend(results)

        seen = set()
        unique_results = []
        for r in all_results:
            title = r.get("title", "")
            if title and title not in seen:
                seen.add(title)
                unique_results.append(r)

        text_output = "\n".join(
            [
                f"- {r.get('title', 'Event')}: {r.get('body', '')}"
                for r in unique_results
            ]
        )
        return text_output
    except Exception as e:
        print(f"[ERROR] Transient event fetch failed: {e}", file=sys.stderr)
        return "Error fetching transient events."


def fetch_fixed_venues(year, month_name):
    try:
        queries = [
            f'"Toronto" "best indoor play places kids" "{year}"',
            f'"Vaughan" "family activities" "museums OR arcades" "{year}"',
            f'"Toronto" "current exhibits" "family" "{month_name} {year}"',
        ]

        all_results = []
        for q in queries:
            results = list(DDGS().text(q, max_results=10))
            all_results.extend(results)

        seen = set()
        unique_results = []
        for r in all_results:
            title = r.get("title", "")
            if title and title not in seen:
                seen.add(title)
                unique_results.append(r)

        text_output = "\n".join(
            [
                f"- {r.get('title', 'Venue/Exhibit')}: {r.get('body', '')}"
                for r in unique_results
            ]
        )
        return text_output
    except Exception as e:
        print(f"[ERROR] Fixed venue fetch failed: {e}", file=sys.stderr)
        return "Error fetching fixed venues."


def scrape_review_score(place_name):
    try:
        query = f'"{place_name}" Vaughan Toronto Google reviews rating'
        results = DDGS().text(query, max_results=3)
        combined_text = " ".join([r.get("body", "") for r in results])

        match = re.search(
            r"([1-4]\.\d|5\.0)(?:\/5|\s*stars?|\s*out of 5)",
            combined_text,
            re.IGNORECASE,
        )
        if match:
            return float(match.group(1))

        match_fallback = re.search(
            r"rating[^\d]*([1-4]\.\d|5\.0)", combined_text, re.IGNORECASE
        )
        if match_fallback:
            return float(match_fallback.group(1))

    except Exception:
        pass
    return 0.0


# ==========================================
# PROMPTS & INSTRUCTIONS
# ==========================================


def build_fixed_system_prompt():
    exclusion_string = ", ".join(
        EXCLUDED_PLACES + VISITED_PLACES
    )

    return f"""
    Act as a creative planning agent for family activities in {CITY}/{REGION}.
    You must output ONLY valid JSON. Do not include markdown formatting or conversational text.

    Constraints:
    - Target: {CHILDREN_STR}. Must accommodate all simultaneously.
    - Exclude: {exclusion_string}.
    - Weather Logic: Check the daily forecast. Only recommend outdoor activities on days where the weather is 'Clear'. If a day has precipitation, activities for that day MUST be indoor.
    - Diversity: Provide a random, diverse mix of 10 year-round places to visit to ensure randomization and discovery of new places.

    Expected JSON Schema:
    {{
      "fixed_activities": [
        {{"name": "...", "location": "...", "target_ages": "...", "price": "...", "weather": "..."}}
      ]
    }}
    Extract exactly 10 valid activities.
    """


def build_fixed_user_prompt(dates_str, weather_str, venues_str):
    return f"""
    Current Context for the upcoming weekend:
    Dates: {dates_str}
    {weather_str}

    Potential Venues and Current Exhibits:
    {venues_str}

    Execute the task based on the system instructions and the provided context to find 10 year-round fixed activities, prioritizing current exhibits or highly-rated venues from the context. Output ONLY JSON.
    """


def build_transient_system_prompt():
    return f"""
    Act as a data-extraction agent for family events in {CITY}/{REGION}.
    You must output ONLY valid JSON. Do not include markdown formatting or conversational text.

    Constraints:
    - Target: {CHILDREN_STR}. Must accommodate all simultaneously or in parallel zones.
    - Weather Logic: Check the daily forecast. Only recommend outdoor events on days where the weather is 'Clear'. If a day has precipitation, events for that day MUST be indoor.
    - Temporal Logic: Verify the temporal logic of the events provided. Reject events tied to holidays or seasons that do not occur during the provided dates.

    Expected JSON Schema:
    {{
      "transient_events": [
        {{"name": "...", "location": "...", "target_ages": "...", "price": "...", "duration": "...", "weather": "...", "day": "..."}}
      ]
    }}
    Extract up to 10 valid events from the provided context.
    """


def build_transient_user_prompt(dates_str, weather_str, events_str):
    return f"""
    Current Context for the upcoming weekend:
    Dates: {dates_str}
    {weather_str}

    High-Signal Transient Events (Filter these strictly! Ensure they match the Dates provided!):
    {events_str}

    Execute the task based on the system instructions and the provided context. Output ONLY JSON.
    """


# ==========================================
# INFERENCE & PROCESSING
# ==========================================


def get_llm_json(system_prompt, user_prompt, max_retries=3):
    """
    Get JSON from LLM with robust parsing.
    Tries MLX first, then falls back to server.
    Uses consolidated functions from osaurus_lib and mlx_lib.
    """

    # Try MLX first
    mlx_model = find_text_mlx_model(["qwen", "llama", "phi"])
    if mlx_model:
        print(f"[llm] Trying MLX: {mlx_model.name}")
        try:
            raw = call_mlx_text(
                mlx_model, f"System: {system_prompt}\n\nUser: {user_prompt}"
            )
            if raw:
                return json.loads(process_mlx_content(raw))
        except Exception:
            pass

    # Fall back to server
    for attempt in range(1, max_retries + 1):
        result = call_llm_api(
            OSAURUS_BASE_URL.rstrip("/"),
            select_best_model(get_available_models()) or MODEL_NAME,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            timeout=600,
        )

        if result and "content" in result:
            try:
                import json as json_module

                cleaned = strip_thinking(result["content"])
                json_start = cleaned.find("{")
                json_end = cleaned.rfind("}")
                if json_start >= 0:
                    return json_module.loads(cleaned[json_start: json_end + 1])
            except Exception:
                if attempt == max_retries:
                    panic_dump(result["content"])

        ensure_server()

    sys.exit(1)


def fetch_scores_for_items(items):
    def fetch_score(item):
        item["score"] = scrape_review_score(
            f"{item['name']} {item['location']}")
        return item

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(fetch_score, items))


def build_markdown_tables(dates_str, weather_str, structured_data, fixed_activities):
    md = f"# Weekend Plan: {dates_str}\n\n{weather_str}\n\n"

    fixed = fixed_activities
    fetch_scores_for_items(fixed)

    fixed.sort(key=lambda x: x["score"], reverse=True)

    md += "### Table 1: Fixed / Year-Round Activities (Ranked by Review Score)\n"
    md += "| Score | Activity & Location | Target Age(s) | Estimated Price (CAD) | Weather Appropriateness |\n"
    md += "| :--- | :--- | :--- | :--- | :--- |\n"
    for item in fixed:
        score_str = f"⭐ {item['score']}/5" if item["score"] > 0 else "N/A"
        md += f"| {score_str} | **{item.get('name')}** ({item.get('location')}) | {item.get('target_ages')} | {item.get('price')} | {item.get('weather')} |\n"

    transient = structured_data.get("transient_events", [])

    grouped_transient = {}
    for item in transient:
        name = item.get("name", "Unknown")
        if name in grouped_transient:
            existing_day = grouped_transient[name].get("day", "")
            new_day = item.get("day", "")
            if new_day and new_day not in existing_day:
                grouped_transient[name]["day"] = f"{existing_day}, {new_day}"
        else:
            grouped_transient[name] = item

    grouped_transient_list = list(grouped_transient.values())
    fetch_scores_for_items(grouped_transient_list)

    grouped_transient_list.sort(key=lambda x: x["score"], reverse=True)

    md += "\n### Table 2: Transient / Limited-Time Events (Ranked by Review Score)\n"
    md += "| Score | Event & Location | Target Age(s) | Est. Price | Duration / End Date | Day | Weather Appr. |\n"
    md += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    for item in grouped_transient_list:
        score_str = f"⭐ {item['score']}/5" if item["score"] > 0 else "N/A"
        md += f"| {score_str} | **{item.get('name')}** ({item.get('location')}) | {item.get('target_ages')} | {item.get('price')} | {item.get('duration')} | {item.get('day', 'N/A')} | {item.get('weather')} |\n"

    return md


def print_to_cli(markdown_content):
    console.print("\n")
    console.print(Markdown(markdown_content))
    console.print("\n")


# ==========================================
# ORCHESTRATOR
# ==========================================


def main():
    init_config()
    start_time = time.time()
    console.print("[bold green]=== Weekend Generator Started ===[/bold green]")
    ensure_server()
    fri, sun = get_weekend_date_objects()
    dates_str = get_weekend_dates_string(fri, sun)
    console.print(f"  [cyan]Bounding Dates:[/cyan] {dates_str}")

    weather_str = fetch_weather(fri, sun)
    console.print(
        f"  [cyan]Weather Forecast:[/cyan]\n  {weather_str.replace('Daily Forecast:', '').strip().replace(chr(10), chr(10) + '  ')}\n"
    )

    with Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_events = progress.add_task(
            "[bold blue]Fetching high-signal events...[/bold blue]", total=None
        )
        task_venues = progress.add_task(
            "[bold blue]Fetching fixed venues/exhibits...[/bold blue]", total=None
        )
        task_fixed = progress.add_task(
            "[bold blue]Generating Fixed Activities (LLM Pass 2)...[/bold blue]",
            start=False,
            total=None,
        )
        task_transient = progress.add_task(
            "[bold blue]Generating Transient Events (LLM Pass 1)...[/bold blue]",
            start=False,
            total=None,
        )
        task_format = progress.add_task(
            "[bold blue]Scraping review scores and formatting...[/bold blue]",
            start=False,
            total=None,
        )

        # Run sequentially in main thread to avoid ProcessPoolExecutor hangs
        year = fri.strftime("%Y")
        month_name = fri.strftime("%B")

        print("[INFO] Starting event fetch...", flush=True)

        # 1. Fetch data sequentially
        progress.update(
            task_events, description="[dim]Fetching events...[/dim]")
        events_str = fetch_transient_events(dates_str, year, month_name)
        print(f"[INFO] Events fetched: {len(events_str)} chars")
        progress.update(
            task_events, description="[dim]✓ Fetched events[/dim]", completed=100
        )

        print("[INFO] Starting venue fetch...")
        progress.update(
            task_venues, description="[dim]Fetching venues...[/dim]")
        venues_str = fetch_fixed_venues(year, month_name)
        print(f"[INFO] Venues fetched: {len(venues_str)} chars")
        progress.update(
            task_venues, description="[dim]✓ Fetched venues[/dim]", completed=100
        )

        print("[INFO] Starting LLM calls...")

        # 2. Generate activities via LLM (in main thread to avoid hangs)
        progress.start_task(task_transient)
        sys_transient = build_transient_system_prompt()
        usr_transient = build_transient_user_prompt(
            dates_str, weather_str, events_str)
        print(
            f"[DEBUG] user_prompt length: {len(usr_transient)}, venues preview: {venues_str[:100]}..."
        )
        print(f"[DEBUG] Using model: {MODEL_NAME}")
        json_transient = get_llm_json(sys_transient, usr_transient)
        progress.update(
            task_transient,
            description="[dim]✓ Generated Transient Events[/dim]",
            completed=100,
        )

        progress.start_task(task_fixed)
        sys_fixed = build_fixed_system_prompt()
        usr_fixed = build_fixed_user_prompt(dates_str, weather_str, venues_str)
        json_fixed = get_llm_json(sys_fixed, usr_fixed)
        progress.update(
            task_fixed,
            description="[dim]✓ Generated Fixed Activities[/dim]",
            completed=100,
        )

        # 3. Format and scrape reviews
        progress.start_task(task_format)
        fixed_acts = json_fixed.get("fixed_activities", [])
        final_markdown = build_markdown_tables(
            dates_str, weather_str, json_transient, fixed_acts
        )
        progress.update(
            task_format, description="[dim]✓ Formatted output[/dim]", completed=100
        )

    print_to_cli(final_markdown)

    output_dir = os.path.expanduser("~/Documents/")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(
        output_dir, f"weekend_plan_{dates_str.replace(' ', '_').replace(',', '')}.md"
    )

    with open(filepath, "w") as f:
        f.write(final_markdown)

    elapsed_time = time.time() - start_time
    console.print(
        f"\n[bold green]Success! Output saved to:[/bold green] {filepath}")
    console.print(
        f"[bold dim]Total Execution Time: {elapsed_time / 60:.2f} minutes[/bold dim]"
    )


if __name__ == "__main__":
    main()
