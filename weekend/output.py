from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from lib.tui import STEP, WARN, console, debug_print
from weekend.config import AGE_RANGE
from weekend.llm import fetch_scores_for_items

# Timing, scoring, and layout constants (Mitchell Hashimoto design)
DEFAULT_SCORE = 0
MAX_RATING_SCALE = 5
SECONDS_PER_MINUTE = 60.0
MISSING_VALUE_PLACEHOLDER = "—"
# Class C6: this number is weekend/scoring.py's internal heuristic over field
# completeness, age overlap and weather fit -- NOT a review score. The heading
# must say what it is; scrape_review_score() is not on the pipeline's call path.
SCORE_LABEL = "Fit Score (computed, not reviews)"
NEWLINE = "\n"


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
    print_info("Time", f"{elapsed_time / SECONDS_PER_MINUTE:.2f} minutes")


def _fmt_score(item):
    score = item.get("score", DEFAULT_SCORE)
    return f"* {score}/{MAX_RATING_SCALE}" if score > DEFAULT_SCORE else ""


def _build_fixed_table(fixed):
    if not fixed:
        return ""
    has_scores = any(item.get("score", DEFAULT_SCORE) > DEFAULT_SCORE for item in fixed)
    heading = "### Fixed / Year-Round Activities"
    if has_scores:
        heading += f" (Ranked by {SCORE_LABEL})"
    heading += "\n"
    if has_scores:
        md = (
            heading
            + "| Score | Activity & Location | Target Age(s) | Estimated Price "
            "(CAD) | Weather Appropriateness |\n| :--- | :--- | :--- | :--- | "
            ":--- |\n"
        )
    else:
        md = (
            heading
            + "| Activity & Location | Target Age(s) | Estimated Price (CAD) | "
            "Weather Appropriateness |\n| :--- | :--- | :--- | :--- |\n"
        )
    for item in fixed:
        score_str = _fmt_score(item)
        name = (
            item.get("name")
            or item.get("activity")
            or item.get("title")
            or item.get("activity_name")
            or "Unknown"
        ).replace("**", "")
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
    return value if value else MISSING_VALUE_PLACEHOLDER


def _fmt_date_range(item):
    """Render an event's real dates, or an honest blank.

    Class C2b/C4: the predecessor printed `item["duration"]` under a column
    headed "Duration / End Date", so a prompt-mandated "2-3 hours" was displayed
    where a date belonged. A duration is not a date and is never shown here.
    """
    start = (item.get("start_date") or "").strip()
    end = (item.get("end_date") or "").strip()
    if start and end and start != end:
        return f"{start} → {end}"
    return _fmt_missing(start or end)


def _build_transient_table(grouped_transient_list):
    if not grouped_transient_list:
        return ""
    has_scores = any(
        item.get("score", DEFAULT_SCORE) > DEFAULT_SCORE for item in grouped_transient_list
    )
    heading = "### Transient / Limited-Time Events"
    if has_scores:
        heading += f" (Ranked by {SCORE_LABEL})"
    heading += "\n"
    if has_scores:
        md = (
            heading
            + "| Score | Event & Location | Target Age(s) | Est. Price | "
            "Dates | Day | Weather Appr. |\n| :--- | :--- | "
            ":--- | :--- | :--- | :--- | :--- |\n"
        )
    else:
        md = (
            heading
            + "| Event & Location | Target Age(s) | Est. Price | Dates "
            "| Day | Weather Appr. |\n| :--- | :--- | :--- | :--- | "
            ":--- | :--- |\n"
        )
    for item in grouped_transient_list:
        score_str = _fmt_score(item)
        name = (
            item.get("name")
            or item.get("event")
            or item.get("title")
            or item.get("event_name")
            or "Unknown"
        )
        name = name.replace("**", "")
        loc = item.get("location") or item.get("address") or ""
        age = _fmt_missing(item.get("target_ages") or item.get("age_group"))
        price = _fmt_missing(item.get("price") or item.get("cost"))
        # C2b: the column is "Dates", so it renders start/end dates -- never a
        # duration standing in for one. A row with no date says so.
        dates = _fmt_date_range(item)
        day = _fmt_missing(item.get("day") or item.get("dates") or item.get("date"))
        weather = _fmt_missing(item.get("weather") or item.get("weather_appropriateness"))
        if has_scores:
            md += (
                f"| {score_str} | **{name}** ({loc}) | {age} | {price} | "
                f"{dates} | {day} | {weather} |\n"
            )
        else:
            md += f"| **{name}** ({loc}) | {age} | {price} | {dates} | {day} | {weather} |\n"
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
        transient = (
            structured_data.get("transient_events")
            or structured_data.get("events")
            or structured_data.get("activities")
            or []
        )

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
    grouped_transient_list.sort(key=lambda x: x.get("score", DEFAULT_SCORE), reverse=True)
    md += NEWLINE + _build_transient_table(grouped_transient_list)

    return md


def print_to_cli(markdown_content):
    console.print(NEWLINE)
    console.print(Markdown(markdown_content))
    console.print(NEWLINE)


def print_to_cli_gorgeous(dates_str, weather_str, fixed_activities, transient_events):
    # Print a beautiful header panel
    console.print()
    panel_text = (
        f"[bold cyan]Weekend Plan: {dates_str}[/bold cyan]\n\n"
        f"[italic white]{weather_str}[/italic white]"
    )
    console.print(Panel(
        panel_text,
        border_style="cyan",
        title="[bold white]ZTools Plan[/bold white]",
        title_align="center"
    ))
    console.print()

    # 1. Render Fixed Activities Table
    if fixed_activities:
        table_fixed = Table(
            title="[bold green]Fixed / Year-Round Activities[/bold green]",
            border_style="dim green"
        )
        has_scores = any(
            item.get("score", DEFAULT_SCORE) > DEFAULT_SCORE for item in fixed_activities
        )
        if has_scores:
            table_fixed.add_column("Score", justify="center", style="yellow")
        table_fixed.add_column("Activity & Location", style="bold white")
        table_fixed.add_column("Ages", justify="center")
        table_fixed.add_column("Price (CAD)", justify="right", style="magenta")
        table_fixed.add_column("Weather Appropriateness", style="italic")

        for item in fixed_activities:
            score = item.get("score", DEFAULT_SCORE)
            score_str = f"* {score}/{MAX_RATING_SCALE}" if score > DEFAULT_SCORE else "—"
            name = (
                item.get("name")
                or item.get("activity")
                or item.get("title")
                or item.get("activity_name")
                or "Unknown"
            ).replace("**", "")
            loc = item.get("location") or item.get("address") or ""
            loc_str = f"{name} ({loc})" if loc else name
            age = item.get("target_ages") or item.get("age_group") or "—"
            price = item.get("price") or item.get("cost") or "—"
            weather = item.get("weather") or item.get("weather_appropriateness") or "—"

            row = []
            if has_scores:
                row.append(score_str)
            row.extend([loc_str, age, price, weather])
            table_fixed.add_row(*row)

        console.print(table_fixed)
        console.print()

    # 2. Render Transient Events Table
    if transient_events:
        table_transient = Table(
            title="[bold purple]Transient / Limited-Time Events[/bold purple]",
            border_style="dim purple"
        )
        has_scores = any(
            item.get("score", DEFAULT_SCORE) > DEFAULT_SCORE for item in transient_events
        )
        if has_scores:
            table_transient.add_column("Score", justify="center", style="yellow")
        table_transient.add_column("Event & Location", style="bold white")
        table_transient.add_column("Ages", justify="center")
        table_transient.add_column("Price", justify="right", style="magenta")
        table_transient.add_column("Dates")
        table_transient.add_column("Day", style="bold blue")
        table_transient.add_column("Weather", style="italic")

        for item in transient_events:
            score = item.get("score", DEFAULT_SCORE)
            score_str = f"* {score}/{MAX_RATING_SCALE}" if score > DEFAULT_SCORE else "—"
            name = (
                item.get("name")
                or item.get("event")
                or item.get("title")
                or item.get("event_name")
                or "Unknown"
            ).replace("**", "")
            loc = item.get("location") or item.get("address") or ""
            loc_str = f"{name} ({loc})" if loc else name
            age = item.get("target_ages") or item.get("age_group") or "—"
            price = item.get("price") or item.get("cost") or "—"
            dates = _fmt_date_range(item)
            day = _fmt_missing(item.get("day") or item.get("dates") or item.get("date"))
            weather = item.get("weather") or item.get("weather_appropriateness") or "—"

            row = []
            if has_scores:
                row.append(score_str)
            row.extend([loc_str, age, price, dates, day, weather])
            table_transient.add_row(*row)

        console.print(table_transient)
        console.print()
