//! Parse a rendered weekend plan document back into structured rows.
//!
//! Port of `eval/report_classes.py`'s read side (`parse_tables`,
//! `transient_rows`, `fixed_rows`, `parse_window_from_wk_filename`), the
//! parsers the G3 quality checks judge plans with. One copy is the whole
//! point: if the numbers a status line reports and the numbers the checks
//! assert on could disagree, enforcement drifts from the planner it measures —
//! the project already paid for that once.

use chrono::NaiveDate;
use std::collections::HashMap;

/// One `| ... |` row, keyed by its column header.
pub type ReportRow = HashMap<String, String>;

/// Split a `wk` report into {section heading: [row dicts]}.
///
/// Keys are the raw column headers, so a caller can assert on the header text
/// itself (class C6 cares that a column is *called* "Review Score").
/// Ensure `heading` has a (possibly empty) row list, and hand it back.
fn ensure_section<'a>(
    tables: &'a mut Vec<(String, Vec<ReportRow>)>,
    heading: &str,
) -> &'a mut Vec<ReportRow> {
    for (i, (h, _)) in tables.iter().enumerate() {
        if h == heading {
            return &mut tables[i].1;
        }
    }
    tables.push((heading.to_string(), Vec::new()));
    &mut tables.last_mut().expect("just pushed").1
}

#[must_use]
pub fn parse_tables(text: &str) -> Vec<(String, Vec<ReportRow>)> {
    let mut tables: Vec<(String, Vec<ReportRow>)> = Vec::new();
    let mut heading = String::new();
    let mut header: Vec<String> = Vec::new();

    for raw in text.lines() {
        let line = raw.trim();
        if let Some(rest) = line.strip_prefix('#') {
            // Python uses `line.lstrip("# ")`: every leading `#` and space goes.
            heading = rest.trim_start_matches(['#', ' ']).trim().to_string();
            header.clear();
            continue;
        }
        let Some(cells) = line.strip_prefix('|') else {
            continue;
        };
        let cells: Vec<String> = cells
            .trim_end_matches('|')
            .split('|')
            .map(|c| c.trim().to_string())
            .collect();
        // The `| :--- | :--- |` separator row: every cell is punctuation.
        let joined: String = cells.iter().map(String::as_str).collect();
        if joined.chars().all(|c| matches!(c, ':' | '-' | ' ')) {
            continue;
        }
        if header.is_empty() {
            header = cells;
            ensure_section(&mut tables, &heading);
            continue;
        }
        let mut row = ReportRow::new();
        for (col, value) in header.iter().zip(&cells) {
            row.insert(col.clone(), value.clone());
        }
        ensure_section(&mut tables, &heading).push(row);
    }
    tables
}

/// Rows under the first section whose heading mentions `needle`.
#[must_use]
fn rows_matching(text: &str, needle: &str) -> Vec<ReportRow> {
    for (heading, rows) in parse_tables(text) {
        if heading.to_lowercase().contains(needle) {
            return rows;
        }
    }
    Vec::new()
}

/// Rows under the transient section (titled "Transient / Limited-Time Events").
#[must_use]
pub fn transient_rows(text: &str) -> Vec<ReportRow> {
    rows_matching(text, "transient")
}

/// Rows under the fixed section (titled "Fixed / Year-Round Activities").
#[must_use]
pub fn fixed_rows(text: &str) -> Vec<ReportRow> {
    rows_matching(text, "fixed")
}

/// Parse `weekend_plan_<Month>_<D>_to_<Month>_<D>_<YYYY>.md` -> (start, end).
///
/// Mirrors Python exactly: both dates are parsed with the single `<YYYY>` in
/// the filename. A year-boundary filename (December 31 `to` January 1) is
/// thereby parsed with an impossible start > end tuple. That is a known
/// Python behaviour, not a Rust mistake, so the port keeps it byte-for-byte —
/// a fix belongs to the planner's naming, not to this parser reading a name.
#[must_use]
pub fn parse_window_from_filename(name: &str) -> Option<(NaiveDate, NaiveDate)> {
    let regex =
        regex::Regex::new(r"([A-Z][a-z]+)_(\d{1,2})_to_([A-Z][a-z]+)_(\d{1,2})_(\d{4})").ok()?;
    let caps = regex.captures(name)?;
    let (m1, d1, m2, d2, year) = (
        caps.get(1)?.as_str(),
        caps.get(2)?.as_str(),
        caps.get(3)?.as_str(),
        caps.get(4)?.as_str(),
        caps.get(5)?.as_str(),
    );
    let parse = |month: &str, day: &str, yr: &str| -> Option<NaiveDate> {
        // `%B` is the full month name, exactly as `datetime.strptime(..., "%B %d %Y")`.
        NaiveDate::parse_from_str(&format!("{month} {day} {yr}"), "%B %d %Y").ok()
    };
    Some((parse(m1, d1, year)?, parse(m2, d2, year)?))
}

/// Parse an ISO-8601 `2026-09-13` into a date.
#[must_use]
pub fn parse_iso_date(value: &str) -> Option<NaiveDate> {
    NaiveDate::parse_from_str(value, "%Y-%m-%d").ok()
}

/// Format a date as ISO-8601 `2026-09-13`.
#[must_use]
pub fn iso_date(date: NaiveDate) -> String {
    date.format("%Y-%m-%d").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ztools::weekend::{
        FIXED_SECTION_HEADING, FIXED_TABLE_HEADER, PlanHealth, TABLE_SEPARATOR,
        TRANSIENT_SECTION_HEADING, TRANSIENT_TABLE_HEADER, WeekendEvent, format_weekend_plan,
    };

    /// A saved plan, byte for byte as [`format_weekend_plan`] writes it.
    ///
    /// It was hand-typed, and it drifted: the six-column header and the
    /// `(Ranked by Fit Score (computed, not reviews))` headings it carried were
    /// the PRE-`Dates` spelling, which no writer has produced since the column
    /// was added. Nothing noticed for a month, because the parsers here are
    /// header-keyed — a document with the wrong columns still parses, it just
    /// parses into rows whose keys are columns the writer never emits. A parser
    /// fixture that is not a writer's output is a test that cannot fail for the
    /// reason it exists.
    ///
    /// So the document is the writer's, and
    /// `the_embedded_sample_is_the_document_the_writer_produces` renders the same
    /// inputs and requires the bytes to match. It stays a LITERAL on purpose:
    /// building it by calling the writer would make this file's parser tests
    /// agree with a broken writer by construction.
    const SAMPLE: &str = "\
# Weekend Plan: August 14 to August 16, 2026 (Vaughan)

**Location:** Vaughan
**Target Ages:** 6-12
**Weather:** Fri 28.2°C (clear), Sat 32.0°C (precipitation)

### Fixed / Year-Round Activities (Ranked by Fit Score)

| Score | Activity & Location | Dates | Target Age(s) | Estimated Price (CAD) | Weather Appropriateness | Why It Fits |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| * 2.5/5 | **Air Riderz** (Vaughan, ON) | — | 5 and up | — | outdoor | Indoor inflatable park |
| * 1.9/5 | **Kortright Centre** (Toronto, ON) | Year-Round | 5 and up | — | indoor | Trails and a centre for discovery |

### Transient / Limited-Time Events (Ranked by Fit Score)

| Score | Event & Location | Dates | Day & Time | Target Age(s) | Estimated Price (CAD) | Why It Fits |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| * 4.0/5 | **Maple Syrup Festival** | April 11 to 12 | Saturday 10am-4pm | 6-13 | By donation | Fresh maple, made on site |

_Provenance: 3 extracted, 1 unsourced, 1 outside the window, 1 excluded._
";

    // `Maple Syrup Festival` carries NO location parenthetical while the two
    // fixed rows do, and that is the writer's `fmt_name_loc` rule rather than a
    // slip: an event at the plan's own city has nowhere else to be. It is left
    // in because a fixture that exercises both branches is worth more than one
    // that exercises the easy one — and because "fixing" it back is the exact
    // edit this gate exists to catch.

    fn event(name: &str, location: &str, price: &str, ages: &str, score: f32) -> WeekendEvent {
        WeekendEvent {
            name: name.to_string(),
            location: location.to_string(),
            price: price.to_string(),
            target_ages: ages.to_string(),
            dates: String::new(),
            day: String::new(),
            weather: String::new(),
            description: String::new(),
            is_transient: true,
            score,
            start_date: String::new(),
            end_date: String::new(),
            duration: String::new(),
        }
    }

    /// The inputs [`SAMPLE`] is the output of. Kept beside the literal so the
    /// two are edited together — see the drift gate below.
    fn sample_plan() -> String {
        let mut health = PlanHealth::nominal();
        health.provenance.extracted = 3;
        health.provenance.unsourced = 1;
        health.provenance.outside_window = 1;
        health.provenance.excluded = 1;

        // Location `Vaughan` IS the plan's own city, so this row's location
        // prints with no parenthetical while the two fixed rows keep theirs.
        let mut festival = event(
            "Maple Syrup Festival",
            "Vaughan",
            "By donation",
            "6-13",
            4.0,
        );
        festival.dates = "April 11 to 12".to_string();
        festival.day = "Saturday 10am-4pm".to_string();
        festival.description = "Fresh maple, made on site".to_string();

        // No dates and no price at all: the writer must print the missing-value
        // sentinel in both cells rather than leave them empty.
        let mut air_riderz = event("Air Riderz", "Vaughan, ON", "", "5 and up", 2.5);
        air_riderz.weather = "outdoor".to_string();
        air_riderz.description = "Indoor inflatable park".to_string();

        let mut kortright = event("Kortright Centre", "Toronto, ON", "", "5 and up", 1.9);
        kortright.dates = "Year-Round".to_string();
        kortright.weather = "indoor".to_string();
        kortright.description = "Trails and a centre for discovery".to_string();

        format_weekend_plan(
            &[festival],
            &[air_riderz, kortright],
            "Vaughan",
            "6-12",
            "August 14 to August 16, 2026",
            "Fri 28.2°C (clear), Sat 32.0°C (precipitation)",
            &health,
        )
    }

    /// The drift gate. A shared constant stops the SPELLING from drifting; this
    /// stops the SAMPLE from drifting from the constant, which is the half a
    /// constant alone cannot do — nothing in the type system says a hand-typed
    /// fixture is still what the writer prints.
    #[test]
    fn the_embedded_sample_is_the_document_the_writer_produces() {
        let rendered = sample_plan();
        if rendered == SAMPLE {
            return;
        }
        // Print the first difference and both sides' line counts: a reader needs
        // to tell a moved heading from a dropped row, and that is the whole
        // failure this replaces.
        let first = rendered
            .lines()
            .zip(SAMPLE.lines())
            .position(|(a, b)| a != b)
            .unwrap_or_else(|| rendered.lines().count().min(SAMPLE.lines().count()));
        panic!(
            "the sample plan in this file is not what format_weekend_plan writes.\n  \
             first difference at line {}\n  rendered: {:?}\n  sample:   {:?}\n  \
             lines: rendered {}, sample {}\nA parser fixture that is not a writer's output \
             tests nothing: fix SAMPLE or the inputs beside it, and read the difference \
             before changing either.",
            first + 1,
            rendered.lines().nth(first),
            SAMPLE.lines().nth(first),
            rendered.lines().count(),
            SAMPLE.lines().count(),
        );
    }

    /// The headings and headers the shared constants promise, in the bytes the
    /// sample carries — so the sample cannot quietly go back to the old
    /// six-column spelling under a passing drift gate.
    #[test]
    fn the_sample_carries_the_shared_headings_and_headers() {
        for (name, expected) in [
            ("fixed heading", FIXED_SECTION_HEADING),
            ("transient heading", TRANSIENT_SECTION_HEADING),
            ("fixed header", FIXED_TABLE_HEADER),
            ("transient header", TRANSIENT_TABLE_HEADER),
            ("separator", TABLE_SEPARATOR),
        ] {
            assert!(SAMPLE.contains(expected), "{name} missing from SAMPLE");
        }
        assert!(
            !SAMPLE.contains("(computed, not reviews)"),
            "the old heading text is back in the sample"
        );
    }

    #[test]
    fn parse_tables_adds_the_separator_and_keeps_header_text() {
        let tables = parse_tables(SAMPLE);
        assert_eq!(tables.len(), 2, "{tables:?}");
        let (fixed_heading, fixed) = &tables[0];
        assert!(fixed_heading.contains("Fixed / Year-Round Activities"));
        assert_eq!(fixed.len(), 2);
        // Header text survives, so C6 can assert the column is *called* that.
        assert!(fixed[0].contains_key("Score"));
        assert!(fixed[0].contains_key("Activity & Location"));
        let (transient_heading, transient) = &tables[1];
        assert!(transient_heading.contains("Transient / Limited-Time Events"));
        assert_eq!(transient.len(), 1);
    }

    #[test]
    fn rows_matching_picks_its_own_section_only() {
        assert_eq!(fixed_rows(SAMPLE).len(), 2);
        assert_eq!(transient_rows(SAMPLE).len(), 1);
        // The fixed heading contains neither needle; a missing section is [].
        assert_empty!(rows_matching("## Some Other Section\n", "fixed"));
    }

    #[test]
    fn window_parses_the_python_filename_shape() {
        assert_eq!(
            parse_window_from_filename("weekend_plan_August_14_to_August_16_2026.md"),
            Some((
                NaiveDate::from_ymd_opt(2026, 8, 14).unwrap(),
                NaiveDate::from_ymd_opt(2026, 8, 16).unwrap(),
            ))
        );
        assert_eq!(
            parse_window_from_filename("weekend_plan_December_31_to_January_02_2027.md"),
            // Both dates use the filename's single year, so this is a start>end
            // tuple -- the known year-boundary quirk, kept to match Python.
            Some((
                NaiveDate::from_ymd_opt(2027, 12, 31).unwrap(),
                NaiveDate::from_ymd_opt(2027, 1, 2).unwrap(),
            ))
        );
    }

    #[test]
    fn window_rejects_unparseable_filenames() {
        assert_eq!(parse_window_from_filename("weekend_plan_latest.md"), None);
        assert_eq!(parse_window_from_filename("notes.txt"), None);
        assert_eq!(
            parse_window_from_filename("weekend_plan_August_99_to_August_99_2026.md"),
            None
        );
    }
}
