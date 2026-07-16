#!/usr/bin/env python3
import argparse
import os
import time
from pathlib import Path

import lib.tui as tui
from lib import init_config
from lib.config import Task, get_model_field_mapping, get_model_top_keys
from lib.osaurus_lib import get_best_model
from lib.osaurus_server import check_server_or_die
from lib.signal_handling import setup_signals
from lib.tui import STEP, WARN, debug_print
from weekend.config import (
    AGE_RANGE,
    CHILDREN,
    CHILDREN_STR,
    CITY,
    DATES_STR,
    DEBUG_EVENTS_FILE,
    DEBUG_VENUES_FILE,
    EXCLUDE_PLACES,
    MODEL_CONFIG,
    MODEL_NAME,
    OSAURUS_APP,
    OSAURUS_BASE_URL,
    REGION,
    WEEKEND_CONFIG,
    ensure_server,
    is_server_running_ours,
    load_events_cache,
    load_venues_cache,
    load_weekend_config,
    restart_osaurus,
    save_events_cache,
    save_venues_cache,
)
from weekend.data import (
    fetch_fixed_venues,
    fetch_transient_events,
    fetch_weather,
    get_weekend_date_objects,
    get_weekend_dates_string,
    scrape_review_score,
)
from weekend.llm import (
    fetch_scores_for_items,
    generate_weekend_plan,
    get_llm_json,
    normalize_llm_items,
)
from weekend.output import (
    build_markdown_tables,
    print_header,
    print_info,
    print_step,
    print_summary,
    print_to_cli,
    print_warning,
    print_to_cli_gorgeous,
)

__all__ = [
    # weekend_config
    "DEBUG_EVENTS_FILE",
    "DEBUG_VENUES_FILE",
    "load_events_cache",
    "save_events_cache",
    "load_venues_cache",
    "save_venues_cache",
    "load_weekend_config",
    "WEEKEND_CONFIG",
    "EXCLUDE_PLACES",
    "CHILDREN",
    "CHILDREN_STR",
    "CITY",
    "REGION",
    "AGE_RANGE",
    "DATES_STR",
    "MODEL_CONFIG",
    "MODEL_NAME",
    "OSAURUS_BASE_URL",
    "OSAURUS_APP",
    "is_server_running_ours",
    "restart_osaurus",
    "ensure_server",
    # weekend_data
    "get_weekend_date_objects",
    "get_weekend_dates_string",
    "fetch_weather",
    "fetch_transient_events",
    "fetch_fixed_venues",
    "scrape_review_score",
    # weekend_llm
    "get_llm_json",
    "normalize_llm_items",
    "fetch_scores_for_items",
    "generate_weekend_plan",
    # weekend_output
    "build_markdown_tables",
    "print_to_cli",
    "print_header",
    "print_step",
    "print_info",
    "print_warning",
    "print_summary",
    # shim-specific
    "main",
    "parse_args",
]
# CLI parsing and document export constants (Mitchell Hashimoto design)
MIN_ITEMS_THRESHOLD = 5
MIN_FIXED_LIST_LEN = 1
EMPTY_LIST_LIMIT = 0
MIN_TRANSIENT_LIST_LEN = 2
FALLBACK_LIST_LEN_HIGH = 3
FALLBACK_LIST_LEN_LOW = 2
OUTPUT_DIR_PATH = str(Path.home() / "Documents")
PLAN_FILE_PREFIX = "weekend_plan_"
OUTPUT_FILE_SUFFIX = ".md"
FILE_WRITE_MODE = "w"


def _fetch_data(fri, sun, year, month_name, use_cache):
    ensure_server()
    dates_str = get_weekend_dates_string(fri, sun)
    print_info("Bounding Dates", dates_str)

    with tui.status("Fetching weather forecast..."):
        weather_str = fetch_weather(fri, sun)
    weather_clean = weather_str.replace("Daily Forecast:", "").strip().replace("\n", " ")
    print_info("Weather", weather_clean)

    with tui.status("Fetching weekend events..."):
        if use_cache:
            events_str = load_events_cache()
            if not events_str:
                events_str = fetch_transient_events(dates_str, year, month_name)
                save_events_cache(events_str)
        else:
            events_str = fetch_transient_events(dates_str, year, month_name)
            save_events_cache(events_str)

    with tui.status("Fetching fixed venues..."):
        if use_cache:
            venues_str = load_venues_cache()
            if not venues_str:
                venues_str = fetch_fixed_venues(year, month_name)
                save_venues_cache(venues_str)
        else:
            venues_str = fetch_fixed_venues(year, month_name)
            save_venues_cache(venues_str)

    return weather_str, events_str, venues_str, dates_str


def _parse_fixed(json_fixed, actual_model, field_mapping):
    fixed_keys = get_model_top_keys(actual_model).get(
        "fixed",
        [
            "fixed_activities",
            "year_round_fixed_activities",
            "venues",
            "places",
            "activities",
            "items",
        ],
    )
    name_keys = ["name"] + [k for k, v in field_mapping.items() if v == "name"]

    if isinstance(json_fixed, list) and len(json_fixed) >= MIN_FIXED_LIST_LEN:
        valid_items = [
            i for i in json_fixed if isinstance(i, dict) and any(i.get(nk) for nk in name_keys)
        ]
        if valid_items:
            return normalize_llm_items(valid_items, field_mapping=field_mapping)

    if isinstance(json_fixed, dict):
        for key in fixed_keys:
            if (
                json_fixed.get(key)
                and isinstance(json_fixed.get(key), list)
                and len(json_fixed.get(key)) > EMPTY_LIST_LIMIT
            ):
                raw = json_fixed[key]
                valid_items = [
                    i for i in raw if isinstance(i, dict) and any(i.get(nk) for nk in name_keys)
                ]
                if valid_items:
                    debug_print(
                        f"[DEBUG] Found valid in key '{key}': {len(valid_items)} items", flush=True
                    )
                    return normalize_llm_items(valid_items, field_mapping=field_mapping)

        if any(json_fixed.get(nk) for nk in name_keys):
            debug_print("[DEBUG] Single object, wrapping in list", flush=True)
            return normalize_llm_items([json_fixed], field_mapping=field_mapping)

        for k, v in json_fixed.items():
            if isinstance(v, list) and len(v) >= MIN_FIXED_LIST_LEN:
                valid_items = [i for i in v if isinstance(i, dict) and i.get("name")]
                if valid_items:
                    debug_print(f"[DEBUG] Fallback key '{k}': {len(valid_items)} items", flush=True)
                    return normalize_llm_items(valid_items, field_mapping=field_mapping)

    return []


def _parse_transient(json_transient, actual_model, field_mapping):
    name_keys = ["name"] + [k for k, v in field_mapping.items() if v == "name"]
    all_name_keys = name_keys + ["description", "title", "event", "summary", "activity_name"]

    if isinstance(json_transient, list) and len(json_transient) >= MIN_TRANSIENT_LIST_LEN:
        filtered = [
            i
            for i in json_transient
            if isinstance(i, dict)
            and not any(k in i for k in ["temperature", "condition", "precipitation"])
        ]
        if not filtered:
            return []

        def _normalize_with_fallback(items):
            result = []
            for item in items:
                new_item = dict(item)
                if not new_item.get("name"):
                    for alt in ["description", "activity_name", "title"]:
                        if new_item.get(alt):
                            new_item["name"] = new_item.pop(alt)
                            break
                result.append(new_item)
            return result

        valid_items = [
            i for i in filtered if isinstance(i, dict) and any(i.get(nk) for nk in all_name_keys)
        ]
        if valid_items:
            result = _normalize_with_fallback(valid_items)
            return normalize_llm_items(result, field_mapping=field_mapping)

        alt_items = [
            i
            for i in filtered
            if isinstance(i, dict)
            and any(
                i.get(ak) for ak in ["description", "title", "event", "summary", "activity_name"]
            )
        ]
        if alt_items:
            result = _normalize_with_fallback(alt_items)
            return normalize_llm_items(result, field_mapping=field_mapping)

    if isinstance(json_transient, dict):
        transient_keys = get_model_top_keys(actual_model).get(
            "transient", ["transient_events", "events", "activities", "recommendations"]
        )

        for key in transient_keys:
            if json_transient.get(key) and isinstance(json_transient.get(key), list):
                raw = json_transient[key]
                valid_items = [i for i in raw if isinstance(i, dict) and i.get("name")]
                if valid_items:
                    return normalize_llm_items(valid_items, field_mapping=field_mapping)

        forecast = json_transient.get("weekend_forecast")
        if forecast and isinstance(forecast, dict):
            all_events = []
            for day_data in forecast.values():
                if isinstance(day_data, dict) and isinstance(day_data.get("events"), list):
                    all_events.extend(day_data["events"])
            if all_events:
                valid_items = [i for i in all_events if isinstance(i, dict) and i.get("name")]
                if valid_items:
                    return normalize_llm_items(valid_items, field_mapping=field_mapping)

        if json_transient.get("name"):
            return normalize_llm_items([json_transient], field_mapping=field_mapping)

        for k, v in json_transient.items():
            if isinstance(v, list) and len(v) >= FALLBACK_LIST_LEN_HIGH:
                valid_items = [i for i in v if isinstance(i, dict) and i.get("name")]
                if valid_items:
                    return normalize_llm_items(valid_items, field_mapping=field_mapping)

        for k, v in json_transient.items():
            if isinstance(v, list) and len(v) >= FALLBACK_LIST_LEN_LOW:
                return normalize_llm_items(v, field_mapping=field_mapping)

    return []


def main(args=None):
    setup_signals()
    if args is None:
        args = parse_args()

    tui.DEBUG = getattr(args, "debug", False)
    init_config()

    if getattr(args, "host", None):
        os.environ["OLLAMA_BASE_URL"] = args.host
    if getattr(args, "api_key", None):
        os.environ["OLLAMA_API_KEY"] = args.api_key

    use_foundation = getattr(args, "use_foundation", False)
    if use_foundation:
        os.environ["OLLAMA_MODEL"] = "foundation"
        print_header("Using model", "foundation (on-device Apple Foundation Model)")
    else:
        check_server_or_die(os.environ.get("OLLAMA_BASE_URL", OSAURUS_BASE_URL))
        if getattr(args, "model", None):
            os.environ["OLLAMA_MODEL"] = args.model
        model = os.environ.get("OLLAMA_MODEL") or get_best_model(Task.JSON)
        print_header("Using model", model)

    start_time = time.time()
    print_step("Weekend Generator Started")

    fri, sun = get_weekend_date_objects()
    year = fri.strftime("%Y")
    month_name = fri.strftime("%B")
    weather_str, events_str, venues_str, dates_str = _fetch_data(
        fri, sun, year, month_name, args.use_cache
    )

    actual_model = os.environ.get("OLLAMA_MODEL") or get_best_model(Task.JSON)
    field_mapping = get_model_field_mapping(actual_model)

    with tui.status("Generating weekend plan..."):
        json_transient, json_fixed = generate_weekend_plan(
            actual_model,
            weather_str,
            events_str,
            venues_str,
            dates_str,
            location=f"{CITY}/{REGION}",
            age_range=AGE_RANGE,
            date_range=dates_str,
            use_foundation=use_foundation,
        )

    fixed_acts = _parse_fixed(json_fixed, actual_model, field_mapping)
    transient_items = _parse_transient(json_transient, actual_model, field_mapping)

    MIN_ITEMS = MIN_ITEMS_THRESHOLD  # MIN_ITEMS = 5
    has_fixed = len(fixed_acts) >= MIN_ITEMS
    has_transient = len(transient_items) >= MIN_ITEMS

    if not has_fixed or not has_transient:
        print_warning(
            f"Low item count - Fixed: {len(fixed_acts)}, Transient: {len(transient_items)}"
        )

    final_markdown = build_markdown_tables(
        dates_str, weather_str, {"transient_events": transient_items}, fixed_acts
    )
    print_step("Formatted output")

    print_to_cli_gorgeous(dates_str, weather_str, fixed_acts, transient_items)

    output_dir = os.path.expanduser(OUTPUT_DIR_PATH)
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(
        output_dir,
        f"{PLAN_FILE_PREFIX}{dates_str.replace(' ', '_').replace(',', '')}{OUTPUT_FILE_SUFFIX}",
    )
    with open(filepath, FILE_WRITE_MODE) as f:
        f.write(final_markdown)

    elapsed_time = time.time() - start_time

    status = STEP if has_fixed and has_transient else WARN
    print_summary(status, len(fixed_acts), len(transient_items), filepath, elapsed_time)


def parse_args():
    p = argparse.ArgumentParser(description="Weekend Planner")
    p.add_argument("--use-cache", action="store_true", help="Use cached web results")
    p.add_argument("--model", default=None, help="Model to use")
    p.add_argument("--skip-web", action="store_true", help="Skip web fetch, use cache only")
    p.add_argument(
        "--host",
        default=None,
        help="Osaurus/Ollama server URL (default: $OLLAMA_BASE_URL or http://localhost:1337)",
    )
    p.add_argument("--api-key", default=None, help="Bearer token for the LLM API")
    p.add_argument(
        "--use-foundation",
        action="store_true",
        help="Use the on-device Apple Foundation Model instead of Osaurus",
    )
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    return p.parse_args()
