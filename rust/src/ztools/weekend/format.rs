use super::WeekendEvent;
use std::fmt::Write as _;

/// The saved plan's two section headings and two table headers, in ONE spelling.
///
/// [`format_weekend_plan`] is the only writer of the saved markdown, and
/// everything that reads it back is header-keyed: `weekend/report.rs`
/// (`fixed_rows`, `transient_rows`), the G3 checks, and `ztools status`. A
/// sample document, a status fixture or a golden that spells a heading
/// differently therefore still parses — which is exactly why three spellings
/// existed with nothing red. These constants are shared so the parser samples
/// are built from the writer's own strings rather than retyped, and
/// `weekend/report.rs` gates its sample against a live render so the sample
/// cannot drift from the constant either.
///
/// The TERMINAL table ([`render_weekend_plan_gorgeous`]) keeps its own headings
/// and headers on purpose: it is a different artefact — no `Why It Fits`
/// column, and column names short enough to sit in a colour table. It is not a
/// third spelling of this one.
pub const FIXED_SECTION_HEADING: &str = "### Fixed / Year-Round Activities (Ranked by Fit Score)";
pub const TRANSIENT_SECTION_HEADING: &str =
    "### Transient / Limited-Time Events (Ranked by Fit Score)";
pub const FIXED_TABLE_HEADER: &str = "| Score | Activity & Location | Dates | Target Age(s) | Estimated Price (CAD) | Weather Appropriateness | Why It Fits |";
pub const TRANSIENT_TABLE_HEADER: &str = "| Score | Event & Location | Dates | Day & Time | Target Age(s) | Estimated Price (CAD) | Why It Fits |";

/// The separator row under both tables.
///
/// Both tables have seven columns, so both separators are the same line. It is
/// kept beside the headers rather than written out twice: a column count that
/// disagrees with its own separator renders as a broken table, and this is the
/// only place that can be checked against [`FIXED_TABLE_HEADER`].
pub const TABLE_SEPARATOR: &str = "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |";

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
            Cell::new("Dates"),
            Cell::new("Weather Appropriateness").add_attribute(comfy_table::Attribute::Italic),
        ]);

        for item in fixed_activities {
            table.add_row(vec![
                Cell::new(format!("* {:.1}/5", item.score)),
                Cell::new(plain_name_loc(&item.name, &item.location)),
                Cell::new(fmt_missing(&item.target_ages)),
                Cell::new(fmt_missing(&item.price)),
                Cell::new(fmt_missing(&item.dates)),
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
    let _ = write!(
        out,
        "**Location:** {location}\n**Target Ages:** {target_ages}\n**Weather:** {weather_display}\n\n"
    );

    out.push_str(FIXED_SECTION_HEADING);
    out.push_str("\n\n");
    if fixed_items.is_empty() {
        out.push_str("*No fixed activities listed.*\n\n");
    } else {
        out.push_str(FIXED_TABLE_HEADER);
        out.push('\n');
        out.push_str(TABLE_SEPARATOR);
        out.push('\n');
        for ev in fixed_items {
            // Every cell is the source's value or the sentinel — never a
            // fabricated filler ("Family activity in GTA") and never a
            // constant column ("Outdoor/Indoor"), both of which read as data.
            let _ = writeln!(
                out,
                "| * {:.1}/5 | {} | {} | {} | {} | {} | {} |",
                ev.score,
                fmt_name_loc(&ev.name, &ev.location, location),
                fmt_missing(&ev.dates),
                fmt_missing(&ev.target_ages),
                fmt_missing(&ev.price),
                fmt_missing(&ev.weather),
                fmt_missing(&ev.description)
            );
        }
        out.push('\n');
    }

    out.push_str(TRANSIENT_SECTION_HEADING);
    out.push_str("\n\n");
    if transient_items.is_empty() {
        // The warning names its cause (health.rs): a bot-walled search and a
        // model that never loaded each read differently from a quiet weekend.
        let _ = write!(
            out,
            "> [!WARNING]\n> **Plan Degraded**: {}\n\n*No transient events scheduled for this weekend.*\n\n",
            health.degraded_reason()
        );
    } else {
        // `Dates` is `WeekendEvent.dates` printed verbatim through
        // `fmt_missing`, the same cell the terminal table has always printed:
        // no range is re-spelled or re-derived here, because a second
        // spelling of a date is a second thing that can be wrong. `Day & Time`
        // stays the model's free text and the two columns do not overlap —
        // `dates` answers WHICH CALENDAR DATES, `day` answers what time of day.
        // An absent value is the sentinel, never an empty cell.
        out.push_str(TRANSIENT_TABLE_HEADER);
        out.push('\n');
        out.push_str(TABLE_SEPARATOR);
        out.push('\n');
        for ev in transient_items {
            let _ = writeln!(
                out,
                "| * {:.1}/5 | {} | {} | {} | {} | {} | {} |",
                ev.score,
                fmt_name_loc(&ev.name, &ev.location, location),
                fmt_missing(&ev.dates),
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
    // The run's own ledger, every run: what the gates dropped is the reading a
    // week of plans is compared on. That comparison has been taken and recorded
    // in `docs/MODEL_QUIRKS.md` ("The weekend provenance ledger, read 2026-10-08"),
    // and its finding is why this line stays: `unsourced` reads 0 whenever
    // `extracted` does, so the line is only a reading once something was extracted.
    out.push_str(&health.provenance.line());
    out.push('\n');

    out
}
/// The ONE value a failed forecast fetch produces, and the ONE value
/// [`format_weather_display`] recognises as "there is no forecast here".
///
/// It is `weekend::fetch::fallback_forecast`'s return value, byte for byte.
/// That is not an accident of history, it is the seam: the fetcher's signature
/// is `fn(url) -> String`, so the only channel a failure has between the wire
/// and the document is the VALUE, and the fetcher cannot tell its own fallback
/// apart from a real forecast once it is handed back as one. Pinning the string
/// here — and pinning `fetch_weather_from`'s live output against THIS constant
/// in `tests/weather_failure.rs`, so the two cannot drift — is what lets the
/// formatter decide by identity instead of by re-reading prose.
///
/// THE OTHER HALF OF THE FIX IS THE RENDERED VALUE, below. Two independent
/// fabricated forecasts used to reach a reader: this one (from the fetcher,
/// skipped by the formatter because it starts with "Daily Forecast") and one
/// hardcoded inside the formatter itself for the empty case. An operator could
/// not tell either from a reading.
pub const FORECAST_FETCH_FAILED: &str =
    "Daily Forecast: Friday: 24.5°C Clear, Saturday: 26.0°C Clear, Sunday: 23.0°C Clear";

/// The sentence a missing forecast renders INSTEAD of numbers.
///
/// `⚠` and the word "unavailable" rather than a plausible temperature: the
/// house rule is that a placeholder must be visually distinct and say WHY, and
/// a placeholder that reads as data is worse than no placeholder. The reason
/// this can hold the numbers out is that the string is also the forecast the
/// scoring and prompting paths see — `compute_score` reads
/// `weather_str.to_lowercase()` for "clear"/"sunny"/"warm" and would have paid
/// an outdoor row +2.0 for the fabricated "clear" it used to be handed.
///
/// It also must not contain any word `compute_score` matches (see
/// `weekend/score.rs::weather_points`): "sunny", "clear", "warm", "cloudy",
/// "rain", "precipitation". A failure that scores rows is a failure that
/// invents a fit.
pub const WEATHER_UNAVAILABLE: &str = "⚠ Forecast unavailable";

/// The cause the document names when the forecast is missing.
///
/// The fetcher knows WHICH of its four failure classes happened (client not
/// built / no daily block / body not JSON / request failed) and prints it with
/// `{e:?}` to stderr — but `fetch_weather_from` returns `String`, so that cause
/// has no path into the document and inventing a specificity here would be a
/// second fabrication. This names the class the sentinel itself proves, and
/// says where the specific reason was written. Closing the gap properly means
/// the fetcher returning the reason (a `Result<String, ForecastFailure>`), which
/// is a change to `weekend/fetch.rs` and to the two test files that pin its
/// signature.
pub const WEATHER_FAILURE_CAUSE: &str = "no forecast came back — the endpoint was unreachable, refused, or answered with no usable daily block. The specific reason was written to stderr.";

/// Is this the fetcher's failure value rather than a forecast?
///
/// Identity, not shape: a forecast that merely *looks* like the fallback (a real
/// three-day dry spell at exactly those temperatures) must still render as a
/// forecast, and the fallback must be recognised without re-deriving what
/// "unavailable" looks like in prose. Trimmed on both sides because the value
/// travels through config-shaped strings.
#[must_use]
pub fn is_forecast_failure(raw: &str) -> bool {
    raw.trim() == FORECAST_FETCH_FAILED
}

/// The one-line weather field: a real forecast, or [`WEATHER_UNAVAILABLE`].
///
/// It used to SKIP every line starting with `Daily Forecast` — which is the
/// fetcher's entire fallback — and then, finding nothing left to print, render
/// a second hardcoded forecast of its own. So the value the code documented as
/// the failure never reached a human, and what the planner saved on a failed
/// fetch was a forecast nobody had ever measured. Both are now the same
/// sentence, and neither carries a temperature.
#[must_use]
pub fn format_weather_display(raw: &str) -> String {
    if is_forecast_failure(raw) {
        return format!("{WEATHER_UNAVAILABLE}: {WEATHER_FAILURE_CAUSE}");
    }
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
    // Nothing parseable: an empty body, or a body that was never a forecast at
    // all. It is the same failure as the fetcher's sentinel — no forecast was
    // obtained — so it renders the same sentence. A second hardcoded forecast
    // lived here, and an empty forecast is exactly when a reader most needs to
    // be told that nobody measured anything.
    if parts.is_empty() {
        format!("{WEATHER_UNAVAILABLE}: {WEATHER_FAILURE_CAUSE}")
    } else {
        parts.join(", ")
    }
}
