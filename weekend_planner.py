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
from lib.config import get_model_top_keys, get_model_field_mapping, get_model_quirks
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
EXCLUDE_PLACES = WEEKEND_CONFIG.get("exclude_places", [])  # Combined list
CHILDREN = WEEKEND_CONFIG.get("children", [])
CHILDREN_STR = ", ".join([f"{c['age']}yo {c['gender']}" for c in CHILDREN]) if CHILDREN else "{CHILDREN_STR}"
CITY = WEEKEND_CONFIG.get("location", {}).get("city", "Vaughan")
REGION = WEEKEND_CONFIG.get("location", {}).get("region", "Toronto")
AGE_RANGE = f"{min(c['age'] for c in CHILDREN)}-{max(c['age'] for c in CHILDREN)}" if CHILDREN else "4-12"
DATES_STR = "April 24 to April 26"  # Placeholder - actual value computed in main()


MODEL_CONFIG = os.path.expanduser("~/.config/model_eval.json")


# Use consolidated functions from osaurus_lib
from lib.config import Task
MODEL_NAME = os.environ.get(
    "OLLAMA_MODEL", get_best_model(Task.JSON) or "gemma-4-26b-a4b-it-4bit"
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
    def safe_search(q, retries=3):
        for attempt in range(retries):
            try:
                results = list(DDGS().text(q, max_results=8))
                return results
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    import time
                    time.sleep(2 ** attempt)
                else:
                    break
        return []

    try:
        queries = [
            "Ontario Science Centre family workshops April 2026",
            "Toronto Zoo special events April 2026",
            "LEGOLAND Discovery Centre Toronto April May 2026",
            "Royal Ontario Museum ROM family programs April 2026",
            "Vaughan community centres kids April 2026",
        ]

        all_results = []
        for q in queries:
            results = safe_search(q)
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
    """Scrape review score with rate limiting."""
    import time
    for attempt in range(3):
        try:
            time.sleep(0.5)  # Rate limit: max 2 queries/second
            query = f'"{place_name}" rating review 5 stars'
            results = list(DDGS().text(query, max_results=5))
            combined = " ".join([r.get("title", "") + " " + r.get("body", "") for r in results])

            match = re.search(r"([0-4]\.\d)\s*/?\s*5", combined, re.IGNORECASE)
            if match:
                return float(match.group(1))
            match2 = re.search(r"rating[:\s]*([0-4]\.\d)", combined, re.IGNORECASE)
            if match2:
                return float(match2.group(1))
            break  # Success or empty, don't retry
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                break
    return 0.0
    return 0.0


# ==========================================
# PROMPTS & INSTRUCTIONS
# ==========================================


def build_fixed_system_prompt(model: str = None, location: str = None, age_range: str = None):
    from lib.config import get_model_prompt, Task

    exclusion_string = ", ".join(EXCLUDE_PLACES)

    # Use provided values or defaults from config
    location = location or f"{CITY}/{REGION}"
    age_range = age_range or AGE_RANGE

    # Try to get from config first
    config_prompt = get_model_prompt(model, Task.WEEKEND_FIXED) if model else ""
    debug_print(f"[DEBUG] build_fixed_system_prompt: model={model}, location={location}, age_range={age_range}", flush=True)
    if config_prompt:
        # Inject runtime variables
        formatted = config_prompt.format(
            location=location,
            age_range=age_range,
            date_range=DATES_STR,
            exclusions=exclusion_string,
        )
        debug_print(f"[DEBUG] prompt after format (first 200): {formatted[:200]}", flush=True)
        return formatted

    # Fallback to hardcoded
    return f"""
    Output JSON now. Use EXACT schema: {{"fixed_activities": [{{"name": "str", "location": "str", "target_ages": "str", "price": "str", "weather": "str"}}]}}

    Extract 10 popular {location} venues for families with kids ages {age_range}.
    Include location (city only), target_ages, price in CAD, weather.

    MANDATORY default values:
    - target_ages: "{age_range}"
    - price: $18-35 per child or free
    - weather: "indoor"

    Never leave any field empty.
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


def build_transient_system_prompt(model: str = None, location: str = None, age_range: str = None, date_range: str = None):
    from lib.config import get_model_prompt, Task

    # Use provided values or defaults from config
    location = location or f"{CITY}/{REGION}"
    age_range = age_range or AGE_RANGE
    date_range = date_range or DATES_STR

    # Try to get from config first
    config_prompt = get_model_prompt(model, Task.WEEKEND_TRANSIENT) if model else ""
    if config_prompt:
        # Inject runtime variables
        return config_prompt.format(
            location=location,
            age_range=age_range,
            date_range=date_range,
        )

    # Fallback to hardcoded
    return f"""
    Output JSON now. Use EXACT schema: {{"transient_events": [{{"name": "str", "location": "str", "target_ages": "str", "price": "str", "duration": "str", "weather": "str", "day": "str"}}]}}

    Extract {location} family events for {date_range}.

    MANDATORY default values:
    - target_ages: "{age_range}"
    - price: $20-30 per child or free
    - duration: "2-3 hours"
    - weather: "indoor"
    - day: Friday/Saturday/Sunday
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


def get_llm_json(system_prompt, user_prompt, max_retries=5):
    """
    Get JSON from LLM with robust parsing.
    Tries Osaurus server first, then falls back to MLX.
    """
    from lib.osaurus_lib import extract_json

    # Try Osaurus server first
    for attempt in range(1, max_retries + 1):
        target_model = get_best_model(Task.JSON)
        debug_print(f"[llm] Trying Osaurus model: {target_model}")
        from lib.osaurus_lib import apply_model_quirks
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        messages = apply_model_quirks(messages, target_model)

        debug_print(f"[llm] Calling API with {len(messages)} messages, system={len(messages[0]['content'])}, user={len(messages[1]['content'])}", flush=True)

        result = call_llm_api(
            OSAURUS_BASE_URL.rstrip("/"),
            target_model,
            messages,
            temperature=0.1,
            timeout=1800,  # 30 min
            parse_json=True,
        )

        debug_print(f"[llm] API response keys: {result.keys() if isinstance(result, dict) else type(result)}", flush=True)
        debug_print(f"[llm] API response preview: {str(result)[:200]}", flush=True)

        if result and "content" in result:
            try:
                from lib.osaurus_lib import _extract_json_only
                import json
                raw_content = result["content"]
                debug_print(f"[llm] Raw content length: {len(raw_content)}", flush=True)
                debug_print(f"[llm] Raw content preview: {raw_content[:300]}", flush=True)

                cleaned = strip_thinking(raw_content)
                debug_print(f"[llm] After strip_thinking length: {len(cleaned)}", flush=True)
                debug_print(f"[llm] After strip_thinking preview: {cleaned[:300]}", flush=True)

                json_str = _extract_json_only(cleaned)
                if json_str is not None:
                    debug_print(f"[llm] JSON extracted successfully, length: {len(json_str)}", flush=True)
                    return json.loads(json_str)
                else:
                    debug_print(f"[llm] WARNING: _extract_json_only returned None", flush=True)
                    raise ValueError("No valid JSON found")
            except Exception as e:
                debug_print(f"[llm] JSON parse error: {e}", flush=True)
                if attempt == max_retries:
                    debug_print(f"[llm] All retries failed, dumping content", flush=True)
                    panic_dump(result["content"])
        else:
            debug_print(f"[llm] WARNING: No content in result: {result}", flush=True)

        # Wait and restart server between retries
        if attempt < max_retries:
            debug_print(f"[llm] Waiting before retry {attempt + 1}/{max_retries}...", flush=True)
            time.sleep(3)
            debug_print(f"[llm] Restarting server before retry...", flush=True)
            ensure_server()
            time.sleep(2)

    # Fall back to MLX
    mlx_model = find_text_mlx_model(["qwen", "llama", "phi"])
    if mlx_model:
        print(f"[llm] Falling back to MLX: {mlx_model.name}")
        try:
            from lib.osaurus_lib import apply_model_quirks
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            messages = apply_model_quirks(messages, getattr(mlx_model, "name", str(mlx_model)))
            mlx_sys = next((m["content"] for m in messages if m["role"] == "system"), system_prompt)
            mlx_usr = next((m["content"] for m in messages if m["role"] == "user"), user_prompt)

            raw = call_mlx(
                mlx_model, f"System: {mlx_sys}\n\nUser: {mlx_usr}"
            )
            if raw:
                from lib.osaurus_lib import _extract_json_only
                import json
                cleaned = process_mlx_content(raw)
                json_str = _extract_json_only(cleaned)
                if json_str is not None:
                    return json.loads(json_str)
                else:
                    raise ValueError("No valid JSON found")
        except Exception as e:
            print(f"[llm] MLX failed: {e}")

    print("[llm] WARNING: Failed to parse JSON, returning empty result")
    return None


def normalize_llm_items(items, field_mapping=None):
    """Normalize LLM output for different model formats.

    Args:
        items: List of items from LLM
        field_mapping: Optional dict mapping model fields to standard fields
    """
    if not items:
        return items

    normalized = []
    for item in items:
        if isinstance(item, str):
            normalized.append({"name": item})
        elif isinstance(item, dict):
            # Apply model-specific field mapping
            if field_mapping:
                for model_field, standard_field in field_mapping.items():
                    if model_field in item and standard_field not in item:
                        item[standard_field] = item[model_field]
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
    # Fetch scores - this is slow (~5s per item) but needed for ranking
    if fixed:
        debug_print(f"[DEBUG] Fetching scores for {len(fixed)} items...", flush=True)
        fetch_scores_for_items(fixed)

    fixed.sort(key=lambda x: x["score"], reverse=True)

    md += "### Table 1: Fixed / Year-Round Activities (Ranked by Review Score)\n"
    md += "| Score | Activity & Location | Target Age(s) | Estimated Price (CAD) | Weather Appropriateness |\n"
    md += "| :--- | :--- | :--- | :--- | :--- |\n"
    for item in fixed:
        score_str = f"⭐ {item['score']}/5" if item.get("score", 0) > 0 else "N/A"
        name = (item.get("name") or item.get("activity") or item.get("title", "Unknown")).replace("**", "")
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
        name = (item.get("name") or item.get("event") or item.get("title", "Unknown")).replace("**", "")
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

# Debug flag - controlled by --debug argument
DEBUG = False


def debug_print(*args, **kwargs):
    """Debug print - only outputs when DEBUG is True."""
    if DEBUG:
        print(*args, **kwargs)


def main(args=None):
    global DEBUG
    args = args or type('Args', (), {'use_cache': False, 'model': None, 'skip_web': False, 'debug': False})()
    DEBUG = getattr(args, 'debug', False)
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
            "[bold blue]Generating Fixed Activities...[/bold blue]",
            start=False,
            total=None,
        )
        task_transient = progress.add_task(
            "[bold blue]Generating Transient Events...[/bold blue]",
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

        # 1. Fetch data (or use cache) - normally quiet
        progress.update(
            task_events, description="[dim]Fetching events...[/dim]")

        if args.use_cache:
            events_str = load_events_cache()
            if events_str:
                pass  # quiet
            else:
                pass  # quiet
                events_str = fetch_transient_events(dates_str, year, month_name)
                save_events_cache(events_str)
        else:
            events_str = fetch_transient_events(dates_str, year, month_name)
            save_events_cache(events_str)
        progress.update(
            task_events, description="[dim]✓ Fetched events[/dim]", completed=100
        )

        progress.update(
            task_venues, description="[dim]Fetching venues...[/dim]")

        if args.use_cache:
            venues_str = load_venues_cache()
            if venues_str:
                pass  # quiet
            else:
                venues_str = fetch_fixed_venues(year, month_name)
                save_venues_cache(venues_str)
        else:
            venues_str = fetch_fixed_venues(year, month_name)
            save_venues_cache(venues_str)
        progress.update(
            task_venues, description="[dim]✓ Fetched venues[/dim]", completed=100
        )

        # Get model for prompts - use OLLAMA_MODEL env var if set, otherwise use best model
        from lib.config import get_best_model, get_model_field_mapping, Task
        actual_model = os.environ.get("OLLAMA_MODEL") or get_best_model(Task.JSON)
        field_mapping = get_model_field_mapping(actual_model)
        debug_print(f"[DEBUG] Using model: {actual_model}, field_mapping: {field_mapping}")

        # 2. Generate activities via LLM (in main thread to avoid hangs)
        progress.start_task(task_transient)
        sys_transient = build_transient_system_prompt(
            actual_model,
            location=f"{CITY}/{REGION}",
            age_range=AGE_RANGE,
            date_range=dates_str,
        )
        usr_transient = build_transient_user_prompt(
            dates_str, weather_str, events_str)
        debug_print(f"[DEBUG] TRANSIENT user_prompt length: {len(usr_transient)}")
        debug_print(f"[DEBUG] TRANSIENT events preview (first 500):\n{events_str[:500]}", flush=True)
        debug_print(f"[DEBUG] TRANSIENT Using model: {actual_model}")
        debug_print(f"[DEBUG] TRANSIENT system prompt (first 300): {sys_transient[:300]}", flush=True)
        json_transient = get_llm_json(sys_transient, usr_transient) or {}
        debug_print(f"[DEBUG] TRANSIENT raw LLM response: {str(json_transient)[:500]}", flush=True)
        progress.update(
            task_transient,
            description="[dim]✓ Generated Transient Events[/dim]",
            completed=100,
        )

        progress.start_task(task_fixed)
        sys_fixed = build_fixed_system_prompt(
            actual_model,
            location=f"{CITY}/{REGION}",
            age_range=AGE_RANGE,
        )
        usr_fixed = build_fixed_user_prompt(dates_str, weather_str, venues_str)
        json_fixed = get_llm_json(sys_fixed, usr_fixed) or {}
        progress.update(
            task_fixed,
            description="[dim]✓ Generated Fixed Activities[/dim]",
            completed=100,
        )

        # 3. Format and scrape reviews
        debug_print(f"[DEBUG] About to start task_format...", flush=True)
        progress.start_task(task_format)
        debug_print(f"[DEBUG] Step 1: Processing json_fixed...", flush=True)
        debug_print(f"[DEBUG] json_fixed preview: {str(json_fixed)[:200]}", flush=True)

        # Extraction for fixed activities - more permissive
        debug_print(f"[DEBUG] json_fixed preview: {str(json_fixed)[:200]}", flush=True)
        fixed_acts = []
        fixed_keys = get_model_top_keys(actual_model).get("fixed", ["fixed_activities", "year_round_fixed_activities", "venues", "places", "activities", "items"])

        # Direct list
        if isinstance(json_fixed, list) and len(json_fixed) >= 1:
            name_keys = ["name"] + [k for k, v in field_mapping.items() if v == "name"]
            valid_items = [i for i in json_fixed if isinstance(i, dict) and any(i.get(nk) for nk in name_keys)]
            if valid_items:
                fixed_acts = normalize_llm_items(valid_items, field_mapping=field_mapping)

        # Dict keys - use model config keys
        if not fixed_acts and isinstance(json_fixed, dict):
            # Also accept field_mapping keys (activity -> name, etc.)
            name_keys = ["name"] + [k for k, v in field_mapping.items() if v == "name"]
            debug_print(f"[DEBUG] name_keys: {name_keys}", flush=True)
            
            for key in fixed_keys:
                if json_fixed.get(key) and isinstance(json_fixed.get(key), list) and len(json_fixed.get(key)) > 0:
                    raw = json_fixed[key]
                    debug_print(f"[DEBUG] Checking key '{key}': {len(raw)} items", flush=True)
                    valid_items = [i for i in raw if isinstance(i, dict) and any(i.get(nk) for nk in name_keys)]
                    if valid_items:
                        debug_print(f"[DEBUG] Found valid in key '{key}': {len(valid_items)} items", flush=True)
                        fixed_acts = normalize_llm_items(valid_items, field_mapping=field_mapping)
                        break

            # Single object - check for any name key
            if not fixed_acts and any(json_fixed.get(nk) for nk in name_keys):
                debug_print(f"[DEBUG] Single object, wrapping in list", flush=True)
                fixed_acts = normalize_llm_items([json_fixed], field_mapping=field_mapping)

            # Any list
            if not fixed_acts:
                for k, v in json_fixed.items():
                    if isinstance(v, list) and len(v) >= 1:
                        valid_items = [i for i in v if isinstance(i, dict) and i.get("name")]
                        if valid_items:
                            debug_print(f"[DEBUG] Fallback key '{k}': {len(valid_items)} items", flush=True)
                            fixed_acts = normalize_llm_items(valid_items, field_mapping=field_mapping)
                            break
        debug_print(f"[DEBUG] fixed_acts: {len(fixed_acts)} items", flush=True)

        # STRICT extraction for transient events
        debug_print(f"[DEBUG] json_transient preview: {str(json_transient)[:300]}", flush=True)
        transient_items = []
        transient_keys = get_model_top_keys(actual_model).get("transient", ["transient_events", "events", "activities", "recommendations"])

        # First, check if the entire response IS the list (no wrapper)
        if isinstance(json_transient, list) and len(json_transient) >= 2:
            name_keys = ["name"] + [k for k, v in field_mapping.items() if v == "name"]
            alt_name_keys = ["description", "title", "event", "summary"]  # Gemma fallback
            debug_print(f"[DEBUG] Transient keys: {name_keys}", flush=True)
            debug_print(f"[DEBUG] Sample item keys: {list(json_transient[0].keys()) if json_transient else 'none'}", flush=True)
            # Filter out weather data (items with temperature/condition keys)
            filtered = [i for i in json_transient if isinstance(i, dict) and not any(k in i for k in ['temperature', 'condition', 'precipitation'])]
            debug_print(f"[DEBUG] Filtered: {len(filtered)}/{len(json_transient)}", flush=True)
            valid_items = [i for i in filtered if isinstance(i, dict) and (any(i.get(nk) for nk in name_keys) or any(i.get(ank) for ank in alt_name_keys))]
            debug_print(f"[DEBUG] Valid: {len(valid_items)}", flush=True)
            if valid_items:
                debug_print(f"[DEBUG] Direct list: {len(valid_items)} items", flush=True)
                # Normalize: move description->name if missing name
                for item in valid_items:
                    if not item.get("name") and item.get("description"):
                        item["name"] = item.pop("description")
                transient_items = normalize_llm_items(valid_items, field_mapping=field_mapping)
            else:
                # Try different keys (gemma uses 'description' as name)
                alt_keys = ["description", "title", "event", "summary"]
                valid_items = [i for i in filtered if isinstance(i, dict) and any(i.get(ak) for ak in alt_keys)]
                debug_print(f"[DEBUG] Alt valid: {len(valid_items)} with alt keys: {alt_keys}", flush=True)
                # Normalize: move description->name
                for item in valid_items:
                    if not item.get("name") and item.get("description"):
                        item["name"] = item.pop("description")
                if valid_items:
                    transient_items = normalize_llm_items(valid_items, field_mapping=field_mapping)

        # If not, try dict keys - use model config keys
        if not transient_items and isinstance(json_transient, dict):
            for key in transient_keys:
                if json_transient.get(key) and isinstance(json_transient.get(key), list):
                    raw = json_transient[key]
                    valid_items = [i for i in raw if isinstance(i, dict) and i.get("name")]
                    if valid_items:
                        debug_print(f"[DEBUG] Found in key '{key}': {len(valid_items)} items", flush=True)
                        transient_items = normalize_llm_items(valid_items, field_mapping=field_mapping)
                        break

            # Check for gemma-style weekend_forecast transform
            if not transient_items and json_transient.get("weekend_forecast"):
                debug_print(f"[DEBUG] Trying gemma weekend_forecast transform", flush=True)
                forecast = json_transient["weekend_forecast"]
                if isinstance(forecast, dict):
                    all_events = []
                    for day_key, day_data in forecast.items():
                        if isinstance(day_data, dict) and isinstance(day_data.get("events"), list):
                            all_events.extend(day_data["events"])
                    if all_events:
                        valid_items = [i for i in all_events if isinstance(i, dict) and i.get("name")]
                        if valid_items:
                            debug_print(f"[DEBUG] Found in weekend_forecast: {len(valid_items)} items", flush=True)
                            transient_items = normalize_llm_items(valid_items, field_mapping=field_mapping)

            # Single object
            if not transient_items and json_transient.get("name"):
                debug_print(f"[DEBUG] Single object, wrapping in list", flush=True)
                transient_items = normalize_llm_items([json_transient], field_mapping=field_mapping)

            # ANY list with 3+ valid items
            if not transient_items:
                for k, v in json_transient.items():
                    if isinstance(v, list) and len(v) >= 3:
                        valid_items = [i for i in v if isinstance(i, dict) and i.get("name")]
                        if valid_items:
                            debug_print(f"[DEBUG] Fallback key '{k}': {len(valid_items)} items", flush=True)
                            transient_items = normalize_llm_items(valid_items, field_mapping=field_mapping)
                            break

            # ANY list with any items
            if not transient_items:
                for k, v in json_transient.items():
                    if isinstance(v, list) and len(v) >= 2:
                        debug_print(f"[DEBUG] Loose fallback key '{k}': {len(v)} items", flush=True)
                        transient_items = normalize_llm_items(v, field_mapping=field_mapping)
                        break

        debug_print(f"[DEBUG] transient_items: {len(transient_items)} items", flush=True)

        # Validate we have minimum items
        MIN_ITEMS = 5
        has_fixed = len(fixed_acts) >= MIN_ITEMS
        has_transient = len(transient_items) >= MIN_ITEMS

        if not has_fixed or not has_transient:
            print(f"[WARNING] Low item count - Fixed: {len(fixed_acts)}, Transient: {len(transient_items)}")

        final_markdown = build_markdown_tables(
            dates_str, weather_str, {"transient_events": transient_items}, fixed_acts)
        progress.update(
            task_format, description="[dim]✓ Formatted output[/dim]", completed=100
        )

    print_to_cli(final_markdown)

    # Count items in output
    fixed_count = len(fixed_acts) if fixed_acts else 0
    transient_count = len(transient_items) if transient_items else 0

    output_dir = os.path.expanduser("~/Documents/")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(
        output_dir, f"weekend_plan_{dates_str.replace(' ', '_').replace(',', '')}.md"
    )

    with open(filepath, "w") as f:
        f.write(final_markdown)

    elapsed_time = time.time() - start_time

    # Report results - require BOTH tables to have minimum items for success
    if has_fixed and has_transient:
        console.print(
            f"\n[bold green]Success![/bold green] Fixed: {fixed_count}, Transient: {transient_count}")
        console.print(f"Output saved to: {filepath}")
    else:
        console.print(
            f"\n[bold yellow]Partial results:[/bold yellow] Fixed: {fixed_count}/{MIN_ITEMS}, Transient: {transient_count}/{MIN_ITEMS}")
        console.print(f"Output saved to: {filepath}")

    console.print(
        f"[bold dim]Total Execution Time: {elapsed_time / 60:.2f} minutes[/bold dim]"
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Weekend Planner")
    p.add_argument("--use-cache", action="store_true", help="Use cached web results")
    p.add_argument("--model", default=None, help="Model to use")
    p.add_argument("--skip-web", action="store_true", help="Skip web fetch, use cache only")
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = p.parse_args()
    main(args)
