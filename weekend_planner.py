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
from pathlib import Path
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
    call_mlx,
    process_mlx_content,
)

# ==========================================
# CONFIGURATION
# ==========================================

console = Console(force_terminal=True, force_interactive=True)

DEBUG_EVENTS_FILE = Path.home() / ".weekend_events_debug_cache.json"
DEBUG_VENUES_FILE = Path.home() / ".weekend_venues_debug_cache.json"


def load_events_cache():
    if DEBUG_EVENTS_FILE.exists():
        return DEBUG_EVENTS_FILE.read_text()
    return None


def save_events_cache(events_str):
    DEBUG_EVENTS_FILE.write_text(events_str)


def load_venues_cache():
    if DEBUG_VENUES_FILE.exists():
        return DEBUG_VENUES_FILE.read_text()
    return None


def save_venues_cache(venues_str):
    DEBUG_VENUES_FILE.write_text(venues_str)


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
        # Focused searches: specific venues + event listing pages
        # Each result should have dates/prices embedded
        queries = [
            "Ontario Science Centre family workshops April 26 2026",
            "Toronto Zoo special events April 2026",
            "LEGOLAND Discovery Centre Toronto April May 2026 events",
            "Royal Ontario Museum ROM family programs April 2026",
            "Vaughan community centres kids April 2026",
        ]

        all_results = []
        for q in queries:
            try:
                results = list(DDGS().text(q, max_results=8))
                all_results.extend(results)
            except Exception as e:
                print(f"[WARN] Query failed: {q[:30]}... - {e}")

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
        # Focused venue queries - each targets a specific venue type
        queries = [
            "indoor play centre Toronto Vaughan 2026 prices",
            "trampoline park Toronto kids 2026",
            "children museum Toronto 2026",
            "family arcade Vaughan 2026",
            "playplace Vaughan indoor kids 2026 prices",
        ]

        all_results = []
        for q in queries:
            try:
                results = list(DDGS().text(q, max_results=8))
                all_results.extend(results)
            except Exception as e:
                print(f"[WARN] Query failed: {q[:30]}... - {e}")

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
    Output JSON now.

    Act as a creative planning agent for family activities in {CITY}/{REGION}.
    Output ONLY valid JSON. No markdown, no conversational text.

    Schema - MANDATORY (every activity must have ALL 5 fields):
    {{
      "fixed_activities": [
        {{"name": "...", "location": "...", "target_ages": "...", "price": "...", "weather": "..."}}
      ]
    }}

    CRITICAL: Use EXACT key "fixed_activities" - NO other keys. Each object MUST have: name, location, target_ages, price, weather.

    EXTRACTION RULE:
    - "at X" → location = X
    - Prices like "$25" → use as price
    - If no price → estimate based on typical costs
    - If no target_ages → use "6-13" as default for kids activities

    Constraints:
    - Target: {CHILDREN_STR}. Must accommodate all simultaneously.
    - Exclude: {exclusion_string}.
    - Weather: use "outdoor" or "indoor"
    - Output ONLY the JSON - no explanation
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
    Output JSON now.

    Act as a data-extraction agent for family events in {CITY}/{REGION}.
    Output ONLY valid JSON. No markdown, no conversational text.

    Schema - MANDATORY (every event must have ALL 7 fields):
    {{
      "transient_events": [
        {{"name": "...", "location": "...", "target_ages": "...", "price": "...", "duration": "...", "weather": "...", "day": "..."}}
      ]
    }}

    CRITICAL: Use EXACT key "transient_events" - NO other keys. Output that exact schema.

    EXTRACTION RULE: Extract ANY event that mentions dates in April 2026:
    - Event name → use "name"
    - Location info → use "location"
    - Date info → use "day" (Friday/Saturday/Sunday only for April 24-26)
    - Duration/Price → fabricate reasonable estimates if not in data

    Constraints:
    - Target: {CHILDREN_STR}. Must accommodate all simultaneously.
    - Weather: use "outdoor" or "indoor"
    - Output ONLY the JSON - no explanation
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
    Tries Osaurus server first, then falls back to MLX.
    """
    from lib.osaurus_lib import extract_json

    # Try Osaurus server first
    for attempt in range(1, max_retries + 1):
        target_model = get_best_model("json")
        print(f"[llm] Trying Osaurus model: {target_model}")
        result = call_llm_api(
            OSAURUS_BASE_URL.rstrip("/"),
            target_model,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            timeout=600,
            parse_json=True,
        )

        if result and "content" in result:
            try:
                import json as json_module

                cleaned = strip_thinking(result["content"])
                json_start = cleaned.find("{")
                json_bracket = cleaned.find("[")
                start_idx = min(x for x in [json_start, json_bracket] if x >= 0) if (json_start >= 0 or json_bracket >= 0) else -1
                json_end = cleaned.rfind("}")
                json_end_bracket = cleaned.rfind("]")
                end_idx = max(json_end, json_end_bracket)

                if start_idx >= 0 and end_idx >= 0:
                    return json_module.loads(cleaned[start_idx: end_idx + 1])
            except Exception:
                if attempt == max_retries:
                    panic_dump(result["content"])

        ensure_server()

    # Fall back to MLX
    mlx_model = find_text_mlx_model(["qwen", "llama", "phi"])
    if mlx_model:
        print(f"[llm] Falling back to MLX: {mlx_model.name}")
        try:
            raw = call_mlx(
                mlx_model, f"System: {system_prompt}\n\nUser: {user_prompt}"
            )
            if raw:
                import json as json_module
                cleaned = process_mlx_content(raw)
                json_start = cleaned.find("{")
                json_bracket = cleaned.find("[")
                start_idx = min(x for x in [json_start, json_bracket] if x >= 0) if (json_start >= 0 or json_bracket >= 0) else -1
                json_end = cleaned.rfind("}")
                json_end_bracket = cleaned.rfind("]")
                end_idx = max(json_end, json_end_bracket)
                
                if start_idx >= 0 and end_idx >= 0:
                    return json_module.loads(cleaned[start_idx: end_idx + 1])
        except Exception as e:
            print(f"[llm] MLX failed: {e}")

    sys.exit(1)


def normalize_llm_items(items):
    """Normalize LLM output for different model formats."""
    if not items:
        return items

    normalized = []
    for item in items:
        if isinstance(item, str):
            normalized.append({"name": item})
        elif isinstance(item, dict):
            # Normalize Gemma field names
            if "age_group" in item and "target_ages" not in item:
                item["target_ages"] = item["age_group"]
            if "setting" in item and "weather" not in item:
                item["weather"] = item["setting"]
            normalized.append(item)
    return normalized


def fetch_scores_for_items(items):
    items = normalize_llm_items(items)

    def fetch_score(item):
        name = item.get("name") or item.get("activity") or item.get("title", "")
        loc = item.get("location") or item.get("address") or item.get("venue", "")
        item["score"] = scrape_review_score(f"{name} {loc}")
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
        score_str = f"⭐ {item['score']}/5" if item.get("score", 0) > 0 else "N/A"
        name = item.get("name") or item.get("activity") or item.get("title", "Unknown")
        loc = item.get("location") or item.get("address") or ""
        age = item.get("target_ages") or item.get("age_group") or ""
        price = item.get("price") or item.get("cost") or ""
        weather = item.get("weather") or item.get("weather_appropriateness") or ""
        md += f"| {score_str} | **{name}** ({loc}) | {age} | {price} | {weather} |\n"

    if isinstance(structured_data, list):
        transient = structured_data
    else:
        transient = structured_data.get("transient_events", []) or structured_data.get("events", []) or []

    grouped_transient = {}
    for item in transient:
        name = item.get("name") or item.get("event") or item.get("title", "Unknown")
        if name in grouped_transient:
            existing_day = grouped_transient[name].get("day", "")
            new_day = item.get("day", "")
            if new_day and new_day not in existing_day:
                grouped_transient[name]["day"] = f"{existing_day}, {new_day}"
        else:
            grouped_transient[name] = item

    grouped_transient_list = list(grouped_transient.values())
    fetch_scores_for_items(grouped_transient_list)

    grouped_transient_list.sort(key=lambda x: x.get("score", 0), reverse=True)

    md += "\n### Table 2: Transient / Limited-Time Events (Ranked by Review Score)\n"
    md += "| Score | Event & Location | Target Age(s) | Est. Price | Duration / End Date | Day | Weather Appr. |\n"
    md += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    for item in grouped_transient_list:
        score_str = f"⭐ {item.get('score', 0)}/5" if item.get("score", 0) > 0 else "N/A"
        name = item.get("name") or item.get("event") or item.get("title", "Unknown")
        loc = item.get("location") or item.get("address") or ""
        age = item.get("target_ages") or item.get("age_group") or ""
        price = item.get("price") or item.get("cost") or ""
        duration = item.get("duration") or item.get("end_date") or ""
        day = item.get("day") or item.get("date") or "N/A"
        weather = item.get("weather") or item.get("weather_appropriateness") or ""
        md += f"| {score_str} | **{name}** ({loc}) | {age} | {price} | {duration} | {day} | {weather} |\n"

    return md


def print_to_cli(markdown_content):
    console.print("\n")
    console.print(Markdown(markdown_content))
    console.print("\n")


# ==========================================
# ORCHESTRATOR
# ==========================================


def main(args=None):
    args = args or type('Args', (), {'use_cache': False, 'model': None, 'skip_web': False})()
    init_config()

    if args.model:
        os.environ['OLLAMA_MODEL'] = args.model

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

        # 1. Fetch data (or use cache)
        progress.update(
            task_events, description="[dim]Fetching events...[/dim]")

        if args.use_cache:
            events_str = load_events_cache()
            if events_str:
                print(f"[cache] Using cached events ({len(events_str)} chars)")
            else:
                print("[!] No cached events found, fetching...")
                events_str = fetch_transient_events(dates_str, year, month_name)
                save_events_cache(events_str)
        else:
            events_str = fetch_transient_events(dates_str, year, month_name)
            save_events_cache(events_str)
        print(f"[INFO] Events: {len(events_str)} chars")
        progress.update(
            task_events, description="[dim]✓ Fetched events[/dim]", completed=100
        )

        print("[INFO] Starting venue fetch...")
        progress.update(
            task_venues, description="[dim]Fetching venues...[/dim]")

        if args.use_cache:
            venues_str = load_venues_cache()
            if venues_str:
                print(f"[cache] Using cached venues ({len(venues_str)} chars)")
            else:
                print("[!] No cached venues found, fetching...")
                venues_str = fetch_fixed_venues(year, month_name)
                save_venues_cache(venues_str)
        else:
            venues_str = fetch_fixed_venues(year, month_name)
            save_venues_cache(venues_str)
        print(f"[INFO] Venues: {len(venues_str)} chars")
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
        # Print actual model being used (get_llm_json uses get_best_model("json"))
        from lib.config import get_best_model
        actual_model = get_best_model("json")
        print(f"[DEBUG] Using model: {actual_model}")
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
        # Handle both dict and list responses + normalize Gemma field names
        if isinstance(json_fixed, list):
            fixed_acts = normalize_llm_items(json_fixed)
        else:
            raw_fixed = (
                json_fixed.get("fixed_activities", []) or
                json_fixed.get("year_round_activities", []) or
                []
            ) if json_fixed else []
            fixed_acts = normalize_llm_items(raw_fixed)

        # Normalize transient events too
        if isinstance(json_transient, list):
            transient_items = normalize_llm_items(json_transient)
        else:
            raw_transient = (
                json_transient.get("transient_events", []) or
                json_transient.get("events", []) or
                json_transient.get("limited_time_events", []) or
                []
            ) if json_transient else []
            transient_items = normalize_llm_items(raw_transient)

        final_markdown = build_markdown_tables(
            dates_str, weather_str, {"transient_events": transient_items}, fixed_acts)
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


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Weekend Planner")
    p.add_argument("--use-cache", action="store_true", help="Use cached web results")
    p.add_argument("--model", default=None, help="Model to use")
    p.add_argument("--skip-web", action="store_true", help="Skip web fetch, use cache only")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
