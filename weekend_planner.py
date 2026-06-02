#!/usr/bin/env python3
import os
import sys
import datetime
import time
import json
import yaml
import re
import concurrent.futures
import argparse
import requests
from pathlib import Path
from ddgs import DDGS


from lib import init_config
from lib.config import get_model_top_keys, get_model_field_mapping, get_model_quirks, Task
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
from lib.mlx_lib import (
    find_text_mlx_model,
    call_mlx,
    process_mlx_content,
)
from lib.tui import STEP, WARN, FAIL

from weekend_config import (
    DEBUG_EVENTS_FILE, DEBUG_VENUES_FILE,
    load_events_cache, save_events_cache,
    load_venues_cache, save_venues_cache,
    load_weekend_config,
    WEEKEND_CONFIG, EXCLUDE_PLACES, CHILDREN, CHILDREN_STR, CITY, REGION, AGE_RANGE, DATES_STR,
    MODEL_CONFIG, MODEL_NAME, OSAURUS_BASE_URL, OSAURUS_APP,
    is_server_running_ours, restart_osaurus, ensure_server,
)
from weekend_data import (
    get_weekend_date_objects, get_weekend_dates_string,
    fetch_weather, fetch_transient_events, fetch_fixed_venues, scrape_review_score,
)
from weekend_prompts import (
    build_fixed_system_prompt, build_fixed_user_prompt,
    build_transient_system_prompt, build_transient_user_prompt,
)
from weekend_llm import (
    get_llm_json, normalize_llm_items, fetch_scores_for_items,
)
from weekend_output import (
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
    # weekend_prompts
    "build_fixed_system_prompt", "build_fixed_user_prompt",
    "build_transient_system_prompt", "build_transient_user_prompt",
    # weekend_llm
    "get_llm_json", "normalize_llm_items", "fetch_scores_for_items",
    # weekend_output
    "build_markdown_tables", "print_to_cli", "print_header", "print_step", "print_info", "print_warning", "print_summary",
    # shim-specific
    "DEBUG", "debug_print", "main", "parse_args",
]

DEBUG = False


def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def main(args=None):
    global DEBUG
    args = args or type('Args', (), {'use_cache': False, 'model': None, 'skip_web': False, 'debug': False})()
    DEBUG = getattr(args, 'debug', False)
    init_config()

    if args.model:
        os.environ['OLLAMA_MODEL'] = args.model

    model = os.environ.get('OLLAMA_MODEL') or get_best_model(Task.JSON)
    print_header("Using model", model)

    start_time = time.time()
    print_step("Weekend Generator Started")
    ensure_server()
    fri, sun = get_weekend_date_objects()
    dates_str = get_weekend_dates_string(fri, sun)
    print_info("Bounding Dates", dates_str)

    weather_str = fetch_weather(fri, sun)
    weather_clean = weather_str.replace("Daily Forecast:", "").strip().replace("\n", " ")
    print_info("Weather", weather_clean)

    year = fri.strftime("%Y")
    month_name = fri.strftime("%B")

    print_step("Fetching events...")
    if args.use_cache:
        events_str = load_events_cache()
        if events_str:
            pass
        else:
            events_str = fetch_transient_events(dates_str, year, month_name)
            save_events_cache(events_str)
    else:
        events_str = fetch_transient_events(dates_str, year, month_name)
        save_events_cache(events_str)
    print_step("Fetched events")

    print_step("Fetching venues...")

    if args.use_cache:
        venues_str = load_venues_cache()
        if venues_str:
            pass
        else:
            venues_str = fetch_fixed_venues(year, month_name)
            save_venues_cache(venues_str)
    else:
        venues_str = fetch_fixed_venues(year, month_name)
        save_venues_cache(venues_str)
    print_step("Fetched venues")

    actual_model = os.environ.get("OLLAMA_MODEL") or get_best_model(Task.JSON)
    field_mapping = get_model_field_mapping(actual_model)
    debug_print(f"[DEBUG] Using model: {actual_model}, field_mapping: {field_mapping}")

    print_step("Generating Transient Events...")
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
    print_step("Generated Transient Events")

    print_step("Generating Fixed Activities...")
    sys_fixed = build_fixed_system_prompt(
        actual_model,
        location=f"{CITY}/{REGION}",
        age_range=AGE_RANGE,
    )
    usr_fixed = build_fixed_user_prompt(dates_str, weather_str, venues_str)
    json_fixed = get_llm_json(sys_fixed, usr_fixed) or {}
    print_step("Generated Fixed Activities")

    debug_print(f"[DEBUG] About to format...", flush=True)
    debug_print(f"[DEBUG] Step 1: Processing json_fixed...", flush=True)
    debug_print(f"[DEBUG] json_fixed preview: {str(json_fixed)[:200]}", flush=True)

    debug_print(f"[DEBUG] json_fixed preview: {str(json_fixed)[:200]}", flush=True)
    fixed_acts = []
    fixed_keys = get_model_top_keys(actual_model).get("fixed", ["fixed_activities", "year_round_fixed_activities", "venues", "places", "activities", "items"])

    if isinstance(json_fixed, list) and len(json_fixed) >= 1:
        name_keys = ["name"] + [k for k, v in field_mapping.items() if v == "name"]
        valid_items = [i for i in json_fixed if isinstance(i, dict) and any(i.get(nk) for nk in name_keys)]
        if valid_items:
            fixed_acts = normalize_llm_items(valid_items, field_mapping=field_mapping)

    if not fixed_acts and isinstance(json_fixed, dict):
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

        if not fixed_acts and any(json_fixed.get(nk) for nk in name_keys):
            debug_print(f"[DEBUG] Single object, wrapping in list", flush=True)
            fixed_acts = normalize_llm_items([json_fixed], field_mapping=field_mapping)

        if not fixed_acts:
            for k, v in json_fixed.items():
                if isinstance(v, list) and len(v) >= 1:
                    valid_items = [i for i in v if isinstance(i, dict) and i.get("name")]
                    if valid_items:
                        debug_print(f"[DEBUG] Fallback key '{k}': {len(valid_items)} items", flush=True)
                        fixed_acts = normalize_llm_items(valid_items, field_mapping=field_mapping)
                        break
    debug_print(f"[DEBUG] fixed_acts: {len(fixed_acts)} items", flush=True)

    debug_print(f"[DEBUG] json_transient preview: {str(json_transient)[:300]}", flush=True)
    transient_items = []
    transient_keys = get_model_top_keys(actual_model).get("transient", ["transient_events", "events", "activities", "recommendations"])

    if isinstance(json_transient, list) and len(json_transient) >= 2:
        name_keys = ["name"] + [k for k, v in field_mapping.items() if v == "name"]
        all_name_keys = name_keys + ["description", "title", "event", "summary", "activity_name"]
        debug_print(f"[DEBUG] Transient name keys: {name_keys}", flush=True)
        debug_print(f"[DEBUG] Sample item keys: {list(json_transient[0].keys()) if json_transient else 'none'}", flush=True)
        filtered = [i for i in json_transient if isinstance(i, dict) and not any(k in i for k in ['temperature', 'condition', 'precipitation'])]
        debug_print(f"[DEBUG] Filtered: {len(filtered)}/{len(json_transient)}", flush=True)
        if not filtered:
            debug_print(f"[DEBUG] All items filtered as weather", flush=True)
            return transient_items
        valid_items = [i for i in filtered if isinstance(i, dict) and any(i.get(nk) for nk in all_name_keys)]
        debug_print(f"[DEBUG] Valid: {len(valid_items)}", flush=True)
        if valid_items:
            debug_print(f"[DEBUG] Direct list: {len(valid_items)} items", flush=True)
            for i, item in enumerate(valid_items):
                new_item = dict(item)
                if not new_item.get("name"):
                    if new_item.get("description"):
                        new_item["name"] = new_item.pop("description")
                    elif new_item.get("activity_name"):
                        new_item["name"] = new_item.pop("activity_name")
                    elif new_item.get("title"):
                        new_item["name"] = new_item.pop("title")
                valid_items[i] = new_item
            transient_items = normalize_llm_items(valid_items, field_mapping=field_mapping)
            debug_print(f"[DEBUG] After normalize: {len(transient_items)} items", flush=True)
        else:
            alt_keys = ["description", "title", "event", "summary", "activity_name"]
            valid_items = [i for i in filtered if isinstance(i, dict) and any(i.get(ak) for ak in alt_keys)]
            debug_print(f"[DEBUG] Alt valid: {len(valid_items)} with alt keys: {alt_keys}", flush=True)
            for i, item in enumerate(valid_items):
                new_item = dict(item)
                if not new_item.get("name"):
                    if new_item.get("description"):
                        new_item["name"] = new_item.pop("description")
                    elif new_item.get("activity_name"):
                        new_item["name"] = new_item.pop("activity_name")
                    elif new_item.get("title"):
                        new_item["name"] = new_item.pop("title")
                valid_items[i] = new_item
            if valid_items:
                transient_items = normalize_llm_items(valid_items, field_mapping=field_mapping)

    if not transient_items and isinstance(json_transient, dict):
        for key in transient_keys:
            if json_transient.get(key) and isinstance(json_transient.get(key), list):
                raw = json_transient[key]
                valid_items = [i for i in raw if isinstance(i, dict) and i.get("name")]
                if valid_items:
                    debug_print(f"[DEBUG] Found in key '{key}': {len(valid_items)} items", flush=True)
                    transient_items = normalize_llm_items(valid_items, field_mapping=field_mapping)
                    break

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

        if not transient_items and json_transient.get("name"):
            debug_print(f"[DEBUG] Single object, wrapping in list", flush=True)
            transient_items = normalize_llm_items([json_transient], field_mapping=field_mapping)

        if not transient_items:
            for k, v in json_transient.items():
                if isinstance(v, list) and len(v) >= 3:
                    valid_items = [i for i in v if isinstance(i, dict) and i.get("name")]
                    if valid_items:
                        debug_print(f"[DEBUG] Fallback key '{k}': {len(valid_items)} items", flush=True)
                        transient_items = normalize_llm_items(valid_items, field_mapping=field_mapping)
                        break

        if not transient_items:
            for k, v in json_transient.items():
                if isinstance(v, list) and len(v) >= 2:
                    debug_print(f"[DEBUG] Loose fallback key '{k}': {len(v)} items", flush=True)
                    transient_items = normalize_llm_items(v, field_mapping=field_mapping)
                    break

    debug_print(f"[DEBUG] transient_items: {len(transient_items)} items", flush=True)

    MIN_ITEMS = 5
    has_fixed = len(fixed_acts) >= MIN_ITEMS
    has_transient = len(transient_items) >= MIN_ITEMS

    if not has_fixed or not has_transient:
        print_warning(f"Low item count - Fixed: {len(fixed_acts)}, Transient: {len(transient_items)}")

    final_markdown = build_markdown_tables(
        dates_str, weather_str, {"transient_events": transient_items}, fixed_acts)
    print_step("Formatted output")

    print_to_cli(final_markdown)

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

    status = STEP if has_fixed and has_transient else WARN
    print_summary(status, fixed_count, transient_count, filepath, elapsed_time)


def parse_args():
    p = argparse.ArgumentParser(description="Weekend Planner")
    p.add_argument("--use-cache", action="store_true", help="Use cached web results")
    p.add_argument("--model", default=None, help="Model to use")
    p.add_argument("--skip-web", action="store_true", help="Skip web fetch, use cache only")
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
