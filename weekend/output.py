from rich.console import Console
from rich.markdown import Markdown

from weekend.llm import fetch_scores_for_items
from weekend.config import AGE_RANGE
from lib.tui import STEP, WARN, debug_print

console = Console(force_terminal=True, force_interactive=True)


def print_header(label, value):
    console.print(f"{label}: {value}")


def print_step(message):
    console.print(f"{STEP} {message}")


def print_info(label, value):
    console.print(f"  {label}: {value}")


def print_warning(message):
    console.print(f"  {WARN}  {message}")


def print_summary(status, fixed_count, transient_count, filepath, elapsed_time):
    console.print(f"\n{status} Weekend plan: {fixed_count} fixed, {transient_count} transient")
    print_info("Saved to", filepath)
    print_info("Time", f"{elapsed_time / 60:.2f} minutes")


def _fmt_score(item):
    score = item.get("score", 0)
    return f"⭐ {score}/5" if score > 0 else ""


def _build_fixed_table(fixed):
    if not fixed:
        return ""
    has_scores = any(item.get("score", 0) > 0 for item in fixed)
    heading = "### Fixed / Year-Round Activities"
    if has_scores:
        heading += " (Ranked by Review Score)"
    heading += "\n"
    if has_scores:
        md = heading + "| Score | Activity & Location | Target Age(s) | Estimated Price (CAD) | Weather Appropriateness |\n| :--- | :--- | :--- | :--- | :--- |\n"
    else:
        md = heading + "| Activity & Location | Target Age(s) | Estimated Price (CAD) | Weather Appropriateness |\n| :--- | :--- | :--- | :--- |\n"
    for item in fixed:
        score_str = _fmt_score(item)
        name = (item.get("name") or item.get("activity") or item.get("title") or item.get("activity_name") or "Unknown").replace("**", "")
        loc = item.get("location") or item.get("address") or ""
        age = item.get("target_ages") or item.get("age_group") or ""
        price = item.get("price") or item.get("cost") or ""
        weather = item.get("weather") or item.get("weather_appropriateness") or ""
        if has_scores:
            md += f"| {score_str} | **{name}** ({loc}) | {age} | {price} | {weather} |\n"
        else:
            md += f"| **{name}** ({loc}) | {age} | {price} | {weather} |\n"
    return md


def _fmt_missing(value):
    return value if value else "—"


def _build_transient_table(grouped_transient_list):
    if not grouped_transient_list:
        return ""
    has_scores = any(item.get("score", 0) > 0 for item in grouped_transient_list)
    heading = "### Transient / Limited-Time Events"
    if has_scores:
        heading += " (Ranked by Review Score)"
    heading += "\n"
    if has_scores:
        md = heading + "| Score | Event & Location | Target Age(s) | Est. Price | Duration / End Date | Day | Weather Appr. |\n| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    else:
        md = heading + "| Event & Location | Target Age(s) | Est. Price | Duration / End Date | Day | Weather Appr. |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    for item in grouped_transient_list:
        score_str = _fmt_score(item)
        name = item.get("name") or item.get("event") or item.get("title") or item.get("event_name") or "Unknown"
        name = name.replace("**", "")
        loc = item.get("location") or item.get("address") or ""
        age = item.get("target_ages") or item.get("age_group") or ""
        price = item.get("price") or item.get("cost") or ""
        duration = item.get("duration") or item.get("end_date") or ""
        day = _fmt_missing(item.get("day") or item.get("dates") or item.get("date"))
        weather = item.get("weather") or item.get("weather_appropriateness") or ""
        if has_scores:
            md += f"| {score_str} | **{name}** ({loc}) | {age} | {price} | {duration} | {day} | {weather} |\n"
        else:
            md += f"| **{name}** ({loc}) | {age} | {price} | {duration} | {day} | {weather} |\n"
    return md


def build_markdown_tables(dates_str, weather_str, structured_data, fixed_activities):
    md = f"# Weekend Plan: {dates_str}\n\n{weather_str}\n\n"

    fixed = fixed_activities
    if fixed:
        debug_print(f"[DEBUG] Fetching scores for {len(fixed)} items...", flush=True)
        fetch_scores_for_items(fixed, weather_str=weather_str, age_range=AGE_RANGE)
    fixed.sort(key=lambda x: x["score"], reverse=True)
    md += _build_fixed_table(fixed)

    if isinstance(structured_data, list):
        transient = structured_data
    else:
        transient = (structured_data.get("transient_events") or
                    structured_data.get("events") or
                    structured_data.get("activities") or
                    [])

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
    fetch_scores_for_items(grouped_transient_list, weather_str=weather_str, age_range=AGE_RANGE)
    grouped_transient_list.sort(key=lambda x: x.get("score", 0), reverse=True)
    md += "\n" + _build_transient_table(grouped_transient_list)

    return md


def print_to_cli(markdown_content):
    console.print("\n")
    console.print(Markdown(markdown_content))
    console.print("\n")
