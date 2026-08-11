#!/usr/bin/env python3
import argparse
import os
import re
import time
from pathlib import Path

import lib.tui as tui
from lib import init_config, osaurus_server
from lib.config import Task, get_model_field_mapping
from lib.llm.constants import DEFAULT_HOST, DEFAULT_PORT
from lib.osaurus_lib import get_best_model
from lib.osaurus_models import FALLBACK_MODEL
from lib.signal_handling import setup_signals
from lib.tui import STEP, WARN, die

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
    print_to_cli_gorgeous,
    print_warning,
)
from weekend.parse import _parse_fixed, _parse_transient

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
OUTPUT_DIR_PATH = os.environ.get("WEEKEND_OUTPUT_DIR", str(Path.home() / "Documents"))
PLAN_FILE_PREFIX = "weekend_plan_"
OUTPUT_FILE_SUFFIX = ".md"
FILE_WRITE_MODE = "w"


def _format_weather_display(weather_str: str) -> str:
    lines = weather_str.replace("Daily Forecast:", "").strip().splitlines()
    parts = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            day, rest = line.split(":", 1)
            day = day.strip()[:3]
            rest = rest.strip()
            m = re.search(r"([\d.]+)°C,\s*(.+?)\s*\(([\d.]+)mm\)", rest)
            if m:
                temp = m.group(1)
                cond = m.group(2).lower()
                precip = float(m.group(3))
                label = cond if precip > 0.5 else f"{cond}"
                parts.append(f"{day} {temp}°C ({label})")
            else:
                parts.append(f"{day} {rest}")
    return ", ".join(parts) if parts else weather_str


def _fetch_data(fri, sun, year, month_name, use_cache, skip_web=False):
    """Gather weather, events and venues.

    `use_cache` prefers the caches but refetches when one is empty. `skip_web`
    is stricter: it forbids the web fetch entirely and fails loudly on an empty
    cache, because "offline mode that silently went online" is worse than an
    error — the flag used to be parsed and never read, so `--skip-web` ran the
    full DDGS fetch.
    """
    ensure_server()
    dates_str = get_weekend_dates_string(fri, sun)
    print_info("Bounding Dates", dates_str)

    with tui.status("Fetching weather forecast..."):
        weather_str = fetch_weather(fri, sun)
    weather_clean = _format_weather_display(weather_str)
    print_info("Weather", weather_clean)

    prefer_cache = use_cache or skip_web

    with tui.status("Fetching weekend events..."):
        events_str = load_events_cache() if prefer_cache else ""
        if not events_str:
            if skip_web:
                die("--skip-web was given but the events cache is empty")
            events_str = fetch_transient_events(dates_str, year, month_name)
            save_events_cache(events_str)

    with tui.status("Fetching fixed venues..."):
        venues_str = load_venues_cache() if prefer_cache else ""
        if not venues_str:
            if skip_web:
                die("--skip-web was given but the venues cache is empty")
            venues_str = fetch_fixed_venues(year, month_name)
            save_venues_cache(venues_str)

    # Make the in-window candidates visible WITHOUT removing the rest. Filtering
    # here instead was tried and reverted: it starved the draft and the model
    # invented events. See weekend/supply.py.
    from weekend.supply import in_window_count, prioritise_in_window

    total = len([line for line in events_str.split("\n") if line.strip()])
    in_window = in_window_count(events_str, fri, sun)
    events_str = prioritise_in_window(events_str, fri, sun)
    # The number that explains a thin plan. 20 candidates of which 0 are
    # in-window is a SUPPLY problem, and it is indistinguishable from a model
    # problem unless somebody counts.
    print_info("Candidates", f"{in_window}/{total} mention a date this weekend")
    if total == 0:
        print(f"{WARN} No event candidates were fetched — the transient plan will be empty.")
    if not venues_str.strip():
        print(f"{WARN} No venue candidates were fetched — the fixed plan will be empty.")

    return weather_str, events_str, venues_str, dates_str


def _enforce_constraints(fixed_acts, transient_items, fri, sun, corpus=""):
    """Apply the code-side constraint checks and report every action taken.

    Classes C3/C5/C8: these rules were previously stated only in a prompt, so
    whether they held depended on the model. Enforcing them here makes them
    deterministic, and printing the notes keeps a filtered run honest rather
    than quietly shorter.
    """
    from weekend.enforce import (
        PROMPT_CONSTANTS,
        correct_weather_labels,
        drop_events_outside_window,
        drop_excluded_places,
        drop_unsourced_rows,
        flag_constant_columns,
        reconcile_day_with_dates,
    )

    notes = []
    # Provenance first: a row that traces to nothing we fetched is invention,
    # and there is no point judging an invented row's dates or weather label.
    fixed_acts, n = drop_unsourced_rows(fixed_acts, corpus)
    notes += n
    transient_items, n = drop_unsourced_rows(transient_items, corpus)
    notes += n
    fixed_acts, n = drop_excluded_places(fixed_acts, EXCLUDE_PLACES)
    notes += n
    transient_items, n = drop_excluded_places(transient_items, EXCLUDE_PLACES)
    notes += n
    transient_items, n = drop_events_outside_window(transient_items, fri, sun)
    notes += n
    # Purely checkable: a row must not disagree with its own dates.
    transient_items, n = reconcile_day_with_dates(transient_items, fri, sun)
    notes += n
    fixed_acts, n = correct_weather_labels(fixed_acts)
    notes += n
    transient_items, n = correct_weather_labels(transient_items)
    notes += n

    # Runs LAST, over what survived: a column can only be judged constant once
    # the rows are final. It reports and changes nothing -- see the docstring on
    # why a mechanically-filled column must not become a drop.
    # The configured family range is the one suspect that is not a literal: it
    # is what 5.2 filled every Target Age(s) cell with.
    suspects = {**PROMPT_CONSTANTS, "Target Age(s)": [AGE_RANGE]}
    notes += flag_constant_columns(fixed_acts, suspects)
    notes += flag_constant_columns(transient_items, suspects)

    for note in notes:
        print_step(note)
    if not notes:
        print_step("Constraint checks: nothing dropped or corrected")
    return fixed_acts, transient_items


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
        os.environ["OLLAMA_MODEL"] = FALLBACK_MODEL
        print_header("Using model", f"{FALLBACK_MODEL} (on-device Apple Foundation Model)")
    else:
        if getattr(args, "model", None):
            os.environ["OLLAMA_MODEL"] = args.model
        model = os.environ.get("OLLAMA_MODEL") or get_best_model(Task.JSON)
        osaurus_server.check_server_or_die(
            os.environ.get("OLLAMA_BASE_URL", OSAURUS_BASE_URL), DEFAULT_PORT, model
        )
        print_header("Using model", model)

    start_time = time.time()
    print_step("Weekend Generator Started")

    fri, sun = get_weekend_date_objects()
    year = fri.strftime("%Y")
    month_name = fri.strftime("%B")
    weather_str, events_str, venues_str, dates_str = _fetch_data(
        fri, sun, year, month_name, args.use_cache, args.skip_web
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
            plan_year=fri.year,
        )

    fixed_acts = _parse_fixed(json_fixed, actual_model, field_mapping)
    transient_items = _parse_transient(json_transient, actual_model, field_mapping)

    # Enforce in code what the prompts can only request (classes C3, C5, C8).
    # Every drop and correction is reported, never silent.
    fixed_acts, transient_items = _enforce_constraints(
        fixed_acts, transient_items, fri, sun, corpus=f"{events_str}\n{venues_str}"
    )

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
        help=f"Osaurus/Ollama server URL (default: $OLLAMA_BASE_URL or http://{DEFAULT_HOST}:{DEFAULT_PORT})",
    )
    p.add_argument("--api-key", default=None, help="Bearer token for the LLM API")
    p.add_argument(
        "--use-foundation",
        action="store_true",
        help="Use the on-device Apple Foundation Model instead of Osaurus",
    )
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    return p.parse_args()
