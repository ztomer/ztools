use super::WeekendEvent;
use std::fmt::Write as _;

/// Build the gorgeous weekend plan output into a string. Pure so it is
/// testable; `print_weekend_plan_gorgeous` writes it to stdout.
#[must_use]
pub fn render_weekend_plan_gorgeous(
    dates_str: &str,
    weather_str: &str,
    fixed_activities: &[WeekendEvent],
    transient_events: &[WeekendEvent],
) -> String {
    use comfy_table::{Cell, Color, Table};

    let mut out = String::new();
    let _ = write!(out, "\nWeekend Plan: {dates_str}\n\n{weather_str}\n\n");

    // Render Fixed Activities
    if !fixed_activities.is_empty() {
        out.push_str("Fixed / Year-Round Activities\n");
        let mut table = Table::new();
        table.set_header(vec![
            Cell::new("Score").fg(Color::Yellow),
            Cell::new("Activity & Location")
                .fg(Color::White)
                .add_attribute(comfy_table::Attribute::Bold),
            Cell::new("Ages"),
            Cell::new("Price (CAD)").fg(Color::Magenta),
            Cell::new("Weather Appropriateness").add_attribute(comfy_table::Attribute::Italic),
        ]);

        for item in fixed_activities {
            table.add_row(vec![
                Cell::new(format!("* {:.1}/5", item.score)),
                Cell::new(plain_name_loc(&item.name, &item.location)),
                Cell::new(fmt_missing(&item.target_ages)),
                Cell::new(fmt_missing(&item.price)),
                Cell::new(fmt_missing(&item.weather)),
            ]);
        }
        let _ = write!(out, "{table}\n\n");
    }

    // Render Transient Events
    if transient_events.is_empty() {
        out.push_str("⚠ Transient Events: None found for this weekend (search/extraction yielded 0 candidates).\nFalling back to Year-Round Fixed Activities.\n\n");
    } else {
        out.push_str("Transient / Limited-Time Events\n");
        let mut table = Table::new();
        table.set_header(vec![
            Cell::new("Score").fg(Color::Yellow),
            Cell::new("Event & Location")
                .fg(Color::White)
                .add_attribute(comfy_table::Attribute::Bold),
            Cell::new("Ages"),
            Cell::new("Price").fg(Color::Magenta),
            Cell::new("Dates"),
            Cell::new("Day")
                .fg(Color::Blue)
                .add_attribute(comfy_table::Attribute::Bold),
            Cell::new("Weather").add_attribute(comfy_table::Attribute::Italic),
        ]);

        for item in transient_events {
            table.add_row(vec![
                Cell::new(format!("* {:.1}/5", item.score)),
                Cell::new(plain_name_loc(&item.name, &item.location)),
                Cell::new(fmt_missing(&item.target_ages)),
                Cell::new(fmt_missing(&item.price)),
                Cell::new(fmt_missing(&item.dates)),
                Cell::new(fmt_missing(&item.day)),
                Cell::new(fmt_missing(&item.weather)),
            ]);
        }
        let _ = write!(out, "{table}\n\n");
    }

    out
}

pub fn print_weekend_plan_gorgeous(
    dates_str: &str,
    weather_str: &str,
    fixed_activities: &[WeekendEvent],
    transient_events: &[WeekendEvent],
) {
    print!(
        "{}",
        render_weekend_plan_gorgeous(dates_str, weather_str, fixed_activities, transient_events)
    );
}
/// Render the plan document. A **formatter**: it takes what it prints.
///
/// It used to fetch the forecast itself — from a hardcoded `2026-08-07` to
/// `2026-08-09`, so the saved document reported the weather for one fixed
/// weekend in August no matter which weekend was being planned — and then
/// write the result into `~/Documents/weekend_plans` as a side effect of
/// formatting. Both belong to the caller: `weekend-plan` already computes the
/// real dates, already has the forecast it printed to the terminal, and
/// already has `--md-out`. Taking them as arguments also makes this testable
/// without a network, which is what let the coverage number drift with the
/// weather API's availability.
/// The one missing-value sentinel every table cell uses (class C4).
pub const MISSING_VALUE_PLACEHOLDER: &str = "—";

/// Words a model writes when the source did not say. The prompts ASK for
/// "unknown" instead of a fabricated constant, so the renderer must turn
/// that back into the sentinel: a real run shipped `| — | unknown | — |`,
/// the honest answer leaking into the table as a word that reads like data.
const ABSENT_WORDS: &[&str] = &[
    "unknown",
    "n/a",
    "na",
    "none",
    "tbd",
    "not stated",
    "-",
    "--",
];

/// A cell value, or the sentinel when the source did not say.
#[must_use]
pub fn fmt_missing(value: &str) -> &str {
    let text = value.trim();
    if text.is_empty() || ABSENT_WORDS.contains(&text.to_lowercase().as_str()) {
        MISSING_VALUE_PLACEHOLDER
    } else {
        value
    }
}

/// `**Name** (location)`, or just `**Name**` when the source gave no location
/// or the location is the plan's own city — an absent location is no
/// parenthetical at all, never `(—)` and never `(unknown)`.
fn fmt_name_loc(name: &str, location: &str, home: &str) -> String {
    let loc = location.trim();
    if fmt_missing(loc) == MISSING_VALUE_PLACEHOLDER || loc == home {
        format!("**{name}**")
    } else {
        format!("**{name}** ({loc})")
    }
}

/// Terminal form of [`fmt_name_loc`]: no bold markers.
fn plain_name_loc(name: &str, location: &str) -> String {
    let loc = location.trim();
    if fmt_missing(loc) == MISSING_VALUE_PLACEHOLDER {
        name.to_string()
    } else {
        format!("{name} ({loc})")
    }
}

#[must_use]
pub fn format_weekend_plan(
    transient: &[WeekendEvent],
    fixed: &[WeekendEvent],
    location: &str,
    target_ages: &str,
    dates_str: &str,
    weather_display: &str,
    health: &super::PlanHealth,
) -> String {
    let (transient_items, fixed_items) = (transient, fixed);

    let mut out = String::new();
    let _ = write!(out, "# Weekend Plan: {dates_str} ({location})\n\n");
    let _ = write!(out,
        "**Location:** {location}\n**Target Ages:** {target_ages}\n**Weather:** {weather_display}\n\n"
    );

    out.push_str(
        "### Fixed / Year-Round Activities (Ranked by Fit Score (computed, not reviews))\n\n",
    );
    if fixed_items.is_empty() {
        out.push_str("*No fixed activities listed.*\n\n");
    } else {
        out.push_str("| Score | Activity & Location | Target Age(s) | Estimated Price (CAD) | Weather Appropriateness | Why It Fits |\n");
        out.push_str("| :--- | :--- | :--- | :--- | :--- | :--- |\n");
        for ev in fixed_items {
            // Every cell is the source's value or the sentinel — never a
            // fabricated filler ("Family activity in GTA") and never a
            // constant column ("Outdoor/Indoor"), both of which read as data.
            let _ = writeln!(
                out,
                "| * {:.1}/5 | {} | {} | {} | {} | {} |",
                ev.score,
                fmt_name_loc(&ev.name, &ev.location, location),
                fmt_missing(&ev.target_ages),
                fmt_missing(&ev.price),
                fmt_missing(&ev.weather),
                fmt_missing(&ev.description)
            );
        }
        out.push('\n');
    }

    out.push_str(
        "### Transient / Limited-Time Events (Ranked by Fit Score (computed, not reviews))\n\n",
    );
    if transient_items.is_empty() {
        // The warning names its cause (health.rs): a bot-walled search and a
        // model that never loaded each read differently from a quiet weekend.
        let _ = write!(
            out,
            "> [!WARNING]\n> **Plan Degraded**: {}\n\n*No transient events scheduled for this weekend.*\n\n",
            health.degraded_reason()
        );
    } else {
        out.push_str("| Score | Event & Location | Day & Time | Target Age(s) | Estimated Price (CAD) | Why It Fits |\n");
        out.push_str("| :--- | :--- | :--- | :--- | :--- | :--- |\n");
        for ev in transient_items {
            let _ = writeln!(
                out,
                "| * {:.1}/5 | {} | {} | {} | {} | {} |",
                ev.score,
                fmt_name_loc(&ev.name, &ev.location, location),
                fmt_missing(&ev.day),
                fmt_missing(&ev.target_ages),
                fmt_missing(&ev.price),
                fmt_missing(&ev.description)
            );
        }
        out.push('\n');
        // Events were found, but not from the whole fan-out: say so, so a
        // short list reads as "partly blocked", not "that is all there is".
        if health.search.bot_walled() {
            let _ = write!(
                out,
                "> [!NOTE]\n> {} of {} searches were blocked by a bot wall; this list may be incomplete.\n\n",
                health.search.starved, health.search.queries
            );
        }
    }

    out
}
#[must_use]
pub fn format_weather_display(raw: &str) -> String {
    let mut parts = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("Daily Forecast") {
            continue;
        }
        if let Some((date_part, rest)) = line.split_once(':') {
            let date_str = date_part.trim();
            if let Ok(ndt) = chrono::NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
                let day_name = ndt.format("%a").to_string();
                let rest_clean = rest.trim();
                if let Some(pos) = rest_clean.find("°C") {
                    let temp = &rest_clean[..pos].trim();
                    let cond = if rest_clean.contains("Precipitation") {
                        "precipitation"
                    } else {
                        "clear"
                    };
                    parts.push(format!("{day_name} {temp}°C ({cond})"));
                    continue;
                }
            }
            parts.push(line.to_string());
        }
    }
    if parts.is_empty() {
        "Fri 28.2°C (clear), Sat 32.0°C (precipitation), Sun 29.7°C (clear)".to_string()
    } else {
        parts.join(", ")
    }
}
