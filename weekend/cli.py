#!/usr/bin/env python3
import os
import time
import argparse


from lib import init_config
from lib.config import get_model_top_keys, get_model_field_mapping, Task
from lib.osaurus_lib import get_best_model

from lib.tui import STEP, WARN, debug_print
import lib.tui as tui
from lib.signal_handling import setup_signals

from weekend.config import (
    DEBUG_EVENTS_FILE, DEBUG_VENUES_FILE,
    load_events_cache, save_events_cache,
    load_venues_cache, save_venues_cache,
    load_weekend_config,
    WEEKEND_CONFIG, EXCLUDE_PLACES, CHILDREN, CHILDREN_STR, CITY, REGION, AGE_RANGE, DATES_STR,
    MODEL_CONFIG, MODEL_NAME, OSAURUS_BASE_URL, OSAURUS_APP,
    is_server_running_ours, restart_osaurus, ensure_server,
)
from weekend.data import (
    get_weekend_date_objects, get_weekend_dates_string,
    fetch_weather, fetch_transient_events, fetch_fixed_venues, scrape_review_score,
)
from weekend.llm import (
    get_llm_json, normalize_llm_items, fetch_scores_for_items,
    generate_weekend_plan,
)
from weekend.output import (
    build_markdown_tables, print_to_cli, print_header, print_step, print_info, print_warning, print_summary,
)

__all__ = [
    # weekend_config
    "DEBUG_EVENTS_FILE", "DEBUG_VENUES_FILE",
    "load_events_cache", "save_events_cache",
    "load_venues_cache", "save_venues_cache",
    "load_weekend_config",
    "WEEKEND_CONFIG", "EXCLUDE_PLACES", "CHILDREN", "CHILDREN_STR", "CITY", "REGION", "AGE_RANGE", "DATES_STR",
    "MODEL_CONFIG", "MODEL_NAME", "OSAURUS_BASE_URL", "OSAURUS_APP",
    "is_server_running_ours", "restart_osaurus", "ensure_server",
    # weekend_data
    "get_weekend_date_objects", "get_weekend_dates_string",
    "fetch_weather", "fetch_transient_events", "fetch_fixed_venues", "scrape_review_score",
    # weekend_llm
    "get_llm_json", "normalize_llm_items", "fetch_scores_for_items", "generate_weekend_plan",
    # weekend_output
    "build_markdown_tables", "print_to_cli", "print_header", "print_step", "print_info", "print_warning", "print_summary",
    # shim-specific
    "main", "parse_args",
]


def _fetch_data(fri, sun, year, month_name, use_cache):
    ensure_server()
    dates_str = get_weekend_dates_string(fri, sun)
    print_info("Bounding Dates", dates_str)

    weather_str = fetch_weather(fri, sun)
    weather_clean = weather_str.replace("Daily Forecast:", "").strip().replace("\n", " ")
    print_info("Weather", weather_clean)

    print_step("Fetching events...")
    if use_cache:
        events_str = load_events_cache()
        if not events_str:
            events_str = fetch_transient_events(dates_str, year, month_name)
            save_events_cache(events_str)
    else:
        events_str = fetch_transient_events(dates_str, year, month_name)
        save_events_cache(events_str)
    print_step("Fetched events")

    print_step("Fetching venues...")
    if use_cache:
        venues_str = load_venues_cache()
        if not venues_str:
            venues_str = fetch_fixed_venues(year, month_name)
            save_venues_cache(venues_str)
    else:
        venues_str = fetch_fixed_venues(year, month_name)
        save_venues_cache(venues_str)
    print_step("Fetched venues")

    return weather_str, events_str, venues_str, dates_str


def _parse_fixed(json_fixed, actual_model, field_mapping):
    fixed_keys = get_model_top_keys(actual_model).get("fixed", ["fixed_activities", "year_round_fixed_activities", "venues", "places", "activities", "items"])
    name_keys = ["name"] + [k for k, v in field_mapping.items() if v == "name"]

    if isinstance(json_fixed, list) and len(json_fixed) >= 1:
        valid_items = [i for i in json_fixed if isinstance(i, dict) and any(i.get(nk) for nk in name_keys)]
        if valid_items:
            return normalize_llm_items(valid_items, field_mapping=field_mapping)

    if isinstance(json_fixed, dict):
        for key in fixed_keys:
            if json_fixed.get(key) and isinstance(json_fixed.get(key), list) and len(json_fixed.get(key)) > 0:
                raw = json_fixed[key]
                valid_items = [i for i in raw if isinstance(i, dict) and any(i.get(nk) for nk in name_keys)]
                if valid_items:
                    debug_print(f"[DEBUG] Found valid in key '{key}': {len(valid_items)} items", flush=True)
                    return normalize_llm_items(valid_items, field_mapping=field_mapping)

        if any(json_fixed.get(nk) for nk in name_keys):
            debug_print(f"[DEBUG] Single object, wrapping in list", flush=True)
            return normalize_llm_items([json_fixed], field_mapping=field_mapping)

        for k, v in json_fixed.items():
            if isinstance(v, list) and len(v) >= 1:
                valid_items = [i for i in v if isinstance(i, dict) and i.get("name")]
                if valid_items:
                    debug_print(f"[DEBUG] Fallback key '{k}': {len(valid_items)} items", flush=True)
                    return normalize_llm_items(valid_items, field_mapping=field_mapping)

    return []


def _parse_transient(json_transient, actual_model, field_mapping):
    name_keys = ["name"] + [k for k, v in field_mapping.items() if v == "name"]
    all_name_keys = name_keys + ["description", "title", "event", "summary", "activity_name"]

    if isinstance(json_transient, list) and len(json_transient) >= 2:
        filtered = [i for i in json_transient if isinstance(i, dict) and not any(k in i for k in ['temperature', 'condition', 'precipitation'])]
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

        valid_items = [i for i in filtered if isinstance(i, dict) and any(i.get(nk) for nk in all_name_keys)]
        if valid_items:
            result = _normalize_with_fallback(valid_items)
            return normalize_llm_items(result, field_mapping=field_mapping)

        alt_items = [i for i in filtered if isinstance(i, dict) and any(i.get(ak) for ak in ["description", "title", "event", "summary", "activity_name"])]
        if alt_items:
            result = _normalize_with_fallback(alt_items)
            return normalize_llm_items(result, field_mapping=field_mapping)

    if isinstance(json_transient, dict):
        transient_keys = get_model_top_keys(actual_model).get("transient", ["transient_events", "events", "activities", "recommendations"])

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
            if isinstance(v, list) and len(v) >= 3:
                valid_items = [i for i in v if isinstance(i, dict) and i.get("name")]
                if valid_items:
                    return normalize_llm_items(valid_items, field_mapping=field_mapping)

        for k, v in json_transient.items():
            if isinstance(v, list) and len(v) >= 2:
                return normalize_llm_items(v, field_mapping=field_mapping)

    return []


def main(args=None):
    setup_signals()
    args = args or type('Args', (), {'use_cache': False, 'model': None, 'skip_web': False, 'debug': False})()
    tui.DEBUG = getattr(args, 'debug', False)
    init_config()

    if args.model:
        os.environ['OLLAMA_MODEL'] = args.model

    model = os.environ.get('OLLAMA_MODEL') or get_best_model(Task.JSON)
    print_header("Using model", model)

    start_time = time.time()
    print_step("Weekend Generator Started")

    fri, sun = get_weekend_date_objects()
    year = fri.strftime("%Y")
    month_name = fri.strftime("%B")
    weather_str, events_str, venues_str, dates_str = _fetch_data(fri, sun, year, month_name, args.use_cache)

    actual_model = os.environ.get("OLLAMA_MODEL") or get_best_model(Task.JSON)
    field_mapping = get_model_field_mapping(actual_model)

    print_step("Generating weekend plan...")
    json_transient, json_fixed = generate_weekend_plan(
        actual_model, weather_str, events_str, venues_str, dates_str,
        location=f"{CITY}/{REGION}",
        age_range=AGE_RANGE,
        date_range=dates_str,
    )
    print_step("Generated weekend plan")

    fixed_acts = _parse_fixed(json_fixed, actual_model, field_mapping)
    transient_items = _parse_transient(json_transient, actual_model, field_mapping)

    MIN_ITEMS = 5
    has_fixed = len(fixed_acts) >= MIN_ITEMS
    has_transient = len(transient_items) >= MIN_ITEMS

    if not has_fixed or not has_transient:
        print_warning(f"Low item count - Fixed: {len(fixed_acts)}, Transient: {len(transient_items)}")

    final_markdown = build_markdown_tables(
        dates_str, weather_str, {"transient_events": transient_items}, fixed_acts)
    print_step("Formatted output")

    print_to_cli(final_markdown)

    output_dir = os.path.expanduser("~/Documents/")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(
        output_dir, f"weekend_plan_{dates_str.replace(' ', '_').replace(',', '')}.md"
    )
    with open(filepath, "w") as f:
        f.write(final_markdown)

    elapsed_time = time.time() - start_time

    status = STEP if has_fixed and has_transient else WARN
    print_summary(status, len(fixed_acts), len(transient_items), filepath, elapsed_time)


def parse_args():
    p = argparse.ArgumentParser(description="Weekend Planner")
    p.add_argument("--use-cache", action="store_true", help="Use cached web results")
    p.add_argument("--model", default=None, help="Model to use")
    p.add_argument("--skip-web", action="store_true", help="Skip web fetch, use cache only")
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    return p.parse_args()
