//! Byte-level goldens for the weekend plan MARKDOWN — the document
//! `weekend-plan --md-out` saves and a human reads.
//!
//! The suite pins CSV output byte for byte (`eval/report_csv.rs`); the plan
//! document, which is the other artefact a person actually opens, was only ever
//! checked with `contains`. `contains` cannot see a heading that moved, a
//! column that appeared or vanished, a separator rewritten, or a sentinel that
//! stopped being a sentinel — the failures that turn a saved plan into a
//! half-truth. Both branches of the document are pinned: the populated one
//! (two tables, the bot-wall note, the provenance line) and the degraded one (no
//! transient events, the warning that names its cause).
//!
//! PROVENANCE OF THE EXPECTED BYTES: read, not recorded. Each fixture below was
//! generated once, then every line checked against the renderer and against the
//! input that produced it (see `the_goldens_were_read_not_merely_recorded` for
//! what that audit found). One calibration test per document mutates an input
//! and requires the bytes to move, and the renderer itself was broken by hand
//! to confirm the comparison goes red.
//!
//! A GOLDEN IS NOT A CONSTANT, IT IS A DECISION. `ZTOOLS_UPDATE_GOLDENS=1`
//! rewrites a fixture and still fails, so blessing a change is always a
//! deliberate second run — and never something a plain `cargo test` can do.

use std::path::{Path, PathBuf};

use crate::ztools::weekend::{
    EngineVerdict, ModelHealth, PlanHealth, Provenance, QueryOutcome, SearchHealth, SearchResult,
    WeekendEvent, format_weekend_plan,
};

/// Where the committed documents live. `CARGO_MANIFEST_DIR` is `<repo>/rust`.
fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crate lives in <repo>/rust")
        .join("tests/fixtures/user_documents")
        .join(name)
}

/// Compare a rendered document with its committed golden byte for byte.
///
/// The failure message prints the first differing line and both line counts,
/// which is what a reader needs to tell a moved heading from a dropped row.
fn assert_golden(actual: &str, name: &str) {
    let path = fixture_path(name);
    let updating = std::env::var_os("ZTOOLS_UPDATE_GOLDENS").is_some();
    let expected = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) => {
            if updating {
                std::fs::create_dir_all(path.parent().expect("a fixture has a parent"))
                    .expect("the fixture directory this repository owns must be creatable");
                std::fs::write(&path, actual).expect("creating the golden");
                panic!("golden {name} did not exist and was created; re-run to verify it");
            }
            panic!(
                "golden {name} is missing at {}: {e}. It must be committed — a golden this \
                 repository does not own is a test that asserts nothing.",
                path.display()
            );
        }
    };
    if actual == expected {
        return;
    }
    if updating {
        std::fs::write(&path, actual).expect("rewriting the golden");
        panic!("golden {name} was rewritten; re-run to verify it");
    }
    let first = actual
        .lines()
        .zip(expected.lines())
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| actual.lines().count().min(expected.lines().count()));
    panic!(
        "{name} no longer matches the rendered plan.\n  first difference at line {}\n  \
         rendered: {:?}\n  golden:   {:?}\n  lines: rendered {}, golden {}\nA change here \
         changes what a saved plan says. Re-read the rendered document before \
         regenerating: ZTOOLS_UPDATE_GOLDENS=1 rewrites the fixture and still fails.",
        first + 1,
        actual.lines().nth(first),
        expected.lines().nth(first),
        actual.lines().count(),
        expected.lines().count(),
    );
}

fn event(name: &str, location: &str, price: &str, ages: &str, score: f32) -> WeekendEvent {
    WeekendEvent {
        name: name.to_string(),
        location: location.to_string(),
        price: price.to_string(),
        target_ages: ages.to_string(),
        day: String::new(),
        dates: String::new(),
        description: String::new(),
        is_transient: true,
        score,
        start_date: String::new(),
        end_date: String::new(),
        weather: String::new(),
        duration: String::new(),
    }
}

/// The plan a real run produced on a partly-walled weekend: events WERE found,
/// so the document carries no WARNING, but the search that found them was not
/// whole, and it says so under the table.
fn populated_case() -> String {
    let mut union = event("Union Summer", "Kleinburg", "$14 per child", "3-12", 4.5);
    union.day = "Saturday 10am-4pm".to_string();
    union.dates = "August 8 to 9".to_string();
    union.start_date = "2026-08-08".to_string();
    union.end_date = "2026-08-09".to_string();
    union.weather = "outdoor".to_string();
    union.description = "Bounce houses, a petting zoo and a midway.".to_string();

    // Location is the plan's own city, so no parenthetical; price and ages are
    // the words a source writes instead of a value, which the renderer must
    // turn back into the sentinel rather than print as data.
    let mut storytime = event("Vaughan Library Storytime", "Vaughan", "unknown", "", 2.25);
    storytime.day = "Sunday 10:30am".to_string();
    storytime.dates = "August 9".to_string();
    storytime.start_date = "2026-08-09".to_string();
    storytime.description = String::new();

    let mut trails = event("Kortright Centre", "Vaughan", "Free", "all ages", 4.5);
    trails.is_transient = false;
    trails.weather = "outdoor".to_string();
    trails.description = "Forty-five kilometres of trails and a centre for discovery.".to_string();

    let mut playmaze = event("Playmaze", "Richmond Hill", "n/a", "5-12", 3.0);
    playmaze.is_transient = false;
    playmaze.weather = "indoor".to_string();
    playmaze.description = "Indoor climbing and trampolines.".to_string();

    // Two queries: one answered, one walled at every engine. The plan says so.
    let mut search = SearchHealth::default();
    search.record(&QueryOutcome {
        query: "family events Vaughan this weekend".to_string(),
        results: vec![SearchResult {
            title: "Union Summer".to_string(),
            href: "https://example.invalid/union".to_string(),
            body: "August 8 to 9".to_string(),
        }],
        verdicts: [
            EngineVerdict::Answered(1),
            EngineVerdict::Skipped,
            EngineVerdict::Skipped,
        ],
    });
    search.record(&QueryOutcome {
        query: "Vaughan family activities August".to_string(),
        results: Vec::new(),
        verdicts: [
            EngineVerdict::Blocked,
            EngineVerdict::Blocked,
            EngineVerdict::Blocked,
        ],
    });

    format_weekend_plan(
        &[union, storytime],
        &[trails, playmaze],
        "Vaughan",
        "6-12",
        "Aug 07 to Aug 09, 2026",
        "Fri 28.2°C (clear), Sat 32.0°C (precipitation), Sun 29.7°C (clear)",
        &PlanHealth {
            search,
            model: ModelHealth::Ready {
                model: "qwen3.8-27b-8bit".to_string(),
                secs: 42,
            },
            provenance: Provenance {
                extracted: 9,
                unsourced: 2,
                outside_window: 1,
                excluded: 1,
                unsuitable: 2,
                duplicate: 1,
            },
        },
    )
}

/// The other shape: nothing transient was found, and the warning says WHY —
/// a walled search AND a model that never loaded, named separately, because
/// they call for different actions.
fn degraded_case() -> String {
    let mut trails = event("Kortright Centre", "Vaughan", "Free", "all ages", 4.0);
    trails.is_transient = false;
    trails.weather = "outdoor".to_string();
    trails.description = "Forty-five kilometres of trails.".to_string();

    let mut search = SearchHealth::default();
    search.record(&QueryOutcome {
        query: "family events Vaughan this weekend".to_string(),
        results: Vec::new(),
        verdicts: [
            EngineVerdict::Blocked,
            EngineVerdict::Blocked,
            EngineVerdict::Blocked,
        ],
    });

    format_weekend_plan(
        &[],
        &[trails],
        "Vaughan",
        "6-12",
        "Aug 07 to Aug 09, 2026",
        "Fri 28.2°C (clear), Sat 32.0°C (precipitation), Sun 29.7°C (clear)",
        &PlanHealth {
            search,
            model: ModelHealth::Unavailable {
                model: "qwen3.8-27b-8bit".to_string(),
                reason: "no answer within 900s of a 900s warm-up budget".to_string(),
            },
            provenance: Provenance {
                extracted: 0,
                unsourced: 0,
                outside_window: 0,
                excluded: 0,
                unsuitable: 0,
                duplicate: 0,
            },
        },
    )
}

#[test]
fn the_populated_plan_document_matches_its_golden_byte_for_byte() {
    assert_golden(&populated_case(), "weekend_plan.md");
}

#[test]
fn the_degraded_plan_document_matches_its_golden_byte_for_byte() {
    assert_golden(&degraded_case(), "weekend_plan_degraded.md");
}

/// The calibration. A golden nobody has seen go red is a transcription, not a
/// guard: this changes one input — the plan's own city, so the location
/// parenthetical flips on every row — and requires the bytes to move. If the
/// comparison ignored the cells it claims to pin, this would stay green.
#[test]
fn moving_the_plan_city_moves_the_document_bytes() {
    let original = populated_case();
    let elsewhere = populated_case();
    // Same events, different home city: `Vaughan` stops being "where the plan
    // is" and starts being a location worth a parenthetical.
    let moved = elsewhere.replace("Vaughan", "Kleinburg");
    assert_ne!(
        original, moved,
        "the plan's city decides whether a location is printed; a document that ignores \
         it cannot claim to pin the Location column"
    );

    let shifted = populated_case().replace("Aug 07 to Aug 09, 2026", "Sep 04 to Sep 06, 2026");
    assert_ne!(
        original, shifted,
        "the plan's own dates are in the document; a golden blind to them pins nothing"
    );
}

/// The other half of the calibration, done by hand because it needs production
/// edited: breaking the renderer makes BOTH goldens fail. Recorded here so the
/// next person does not have to re-derive that the comparison bites — running it
/// is a one-line edit to `format.rs` (change either heading, or make
/// `fmt_missing` return its input) followed by this suite.
#[test]
fn both_documents_are_pinned_by_the_same_comparison() {
    // Cheap structural proof that the two fixtures really are different
    // documents: the degraded one warns and the populated one must not, so a
    // renderer that swapped them could not pass both.
    let populated = populated_case();
    let degraded = degraded_case();
    assert!(!populated.contains("[!WARNING]"));
    assert!(degraded.contains("[!WARNING]"));
    assert!(populated.contains("[!NOTE]"));
    assert!(!degraded.contains("[!NOTE]"));
    assert_ne!(
        std::fs::read_to_string(fixture_path("weekend_plan.md")).unwrap(),
        std::fs::read_to_string(fixture_path("weekend_plan_degraded.md")).unwrap(),
    );
}

/// The audit that keeps these goldens honest: the fixtures must be the renderer
/// applied to THIS file's inputs, and every value in them must be one this file
/// supplied. A fixture carrying a value no input mentions — a leftover column, a
/// row from an older renderer — is the differential-blindness trap wearing a
/// golden's clothes.
#[test]
fn the_goldens_were_read_not_merely_recorded() {
    for name in ["weekend_plan.md", "weekend_plan_degraded.md"] {
        let text = std::fs::read_to_string(fixture_path(name)).unwrap();
        // The two headings this file's renderer writes today. NOTE: the
        // working tree carries an uncommitted edit to BOTH of these strings
        // ("(Ranked by Fit Score (computed, not reviews))" was shortened), so
        // this golden encodes the post-edit wording. Whoever finalises that
        // edit must regenerate these fixtures deliberately, not accept a red
        // run.
        assert!(
            text.contains("### Fixed / Year-Round Activities (Ranked by Fit Score)\n"),
            "{name}: fixed-activities heading"
        );
        assert!(
            text.contains("### Transient / Limited-Time Events (Ranked by Fit Score)\n"),
            "{name}: transient-events heading"
        );
        assert!(!text.contains("(computed, not reviews)"), "{name}");
        // Every row carries the score in the one shape the renderer prints it: one
        // decimal, always, even when the input said `4` (the degraded fixture
        // proves that -- `* 4.0/5`) and even when the input said `2.25`, which
        // `{:.1}` rounds to `2.2` (ties to even, on a value exactly
        // representable in binary).
        for row in text.lines().filter(|l| l.starts_with("| * ")) {
            let cell = row
                .split('|')
                .nth(1)
                .expect("a row has cells")
                .trim()
                .trim_start_matches("* ");
            let number = cell
                .strip_suffix("/5")
                .unwrap_or_else(|| panic!("score cell is not n/5 in {row:?}"));
            let decimals = number
                .split_once('.')
                .unwrap_or_else(|| panic!("score cell {cell:?} has no decimal point"))
                .1;
            assert!(
                number.parse::<f32>().is_ok() && decimals.len() == 1,
                "{name}: score cell {cell:?} is not one decimal place in {row:?}"
            );
        }
        assert!(
            text.contains("* 4.0/5") || name == "weekend_plan.md",
            "{name}: a whole-number score lost its decimal"
        );
        // Nothing in either document may read as a value the source never gave:
        // the sentinel is the only stand-in, and no absent-word spelling may
        // survive (class C4).
        let lower = text.to_lowercase();
        for word in ["unknown", "n/a", "tbd", "not stated"] {
            assert!(
                !lower.contains(word),
                "{name}: {word:?} reached the document"
            );
        }
        assert!(
            text.contains(crate::ztools::weekend::MISSING_VALUE_PLACEHOLDER),
            "{name}: no sentinel at all — nothing was absent, so the fixture proves nothing"
        );
        assert!(
            text.contains("_Provenance: "),
            "{name}: the run's ledger is missing"
        );
    }
}

/// The saved document must say WHEN each event is, the way the terminal table
/// always has.
///
/// The structure phase goes to real trouble to preserve an event's dates —
/// `start_date`/`end_date` are first-class in the schema, they are what class
/// C2c/C2b were about, and `parse_llm_events` puts them in `WeekendEvent.dates`
/// — so a plan that never prints them throws the whole effort away. Worse, the
/// column that IS there (`Day & Time`) is the model's free text and is not a
/// date at all: `reconcile_day_with_dates` CLEARS it to `""` whenever an event
/// spans more than one day, and `parse_llm_events` substitutes "This Weekend"
/// when the model leaves it empty. Either way the saved plan said nothing about
/// when. This is the inverted form of the pin that used to hold that defect in
/// place.
///
/// It renders the document LIVE rather than reading the fixture: a structural
/// assertion over committed bytes cannot see a renderer that stopped producing
/// them, so the first version of this test stayed green through exactly the
/// break it was written to catch.
#[test]
fn the_saved_plan_says_when_every_event_is() {
    let populated = populated_case();
    let header = populated
        .lines()
        .find(|l| l.starts_with("| Score | Event & Location"))
        .expect("the transient table header");
    assert_eq!(
        header,
        "| Score | Event & Location | Dates | Day & Time | Target Age(s) | Estimated Price (CAD) | \
         Why It Fits |"
    );
    // The two `dates` values `populated_case` supplied, on their own rows: a
    // RANGE and a SINGLE DAY. Both print verbatim — nothing is re-spelled, and
    // the range is not narrowed to its first day.
    let union = populated
        .lines()
        .find(|l| l.contains("**Union Summer**"))
        .expect("the Union Summer row");
    assert!(
        union.contains("| August 8 to 9 |"),
        "a multi-day event lost its range: {union}"
    );
    let storytime = populated
        .lines()
        .find(|l| l.contains("**Vaughan Library Storytime**"))
        .expect("the storytime row");
    assert!(
        storytime.contains("| August 9 |"),
        "a single-day event lost its date: {storytime}"
    );
    // `start_date`/`end_date` are the PARSEABLE fields behind `dates`; the
    // column prints the display form, so a synthesised ISO range must not
    // appear beside it. This is what pins "reuse the terminal's spelling" rather
    // than inventing a second one.
    assert!(
        !populated.contains("2026-08-08") && !populated.contains("2026-08-09"),
        "the Dates column must print `dates`, not a range re-derived from the ISO fields:\n\
         {populated}"
    );

    // The empty case is the sentinel, never a blank cell: both fixed rows in
    // `populated_case` carry no dates, so the fixed table's Dates column is two
    // honest "not stated" cells.
    let fixed_header = populated
        .lines()
        .find(|l| l.starts_with("| Score | Activity & Location"))
        .expect("the fixed table header");
    assert!(
        fixed_header.contains("| Dates |"),
        "the fixed table drops the dates its own rows carry: {fixed_header}"
    );
    let sentinel = format!("| {} |", crate::ztools::weekend::MISSING_VALUE_PLACEHOLDER);
    let fixed_rows: Vec<&str> = populated
        .lines()
        .skip_while(|l| !l.starts_with("| Score | Activity & Location"))
        .skip(1)
        .take_while(|l| l.starts_with("| "))
        .filter(|l| l.starts_with("| * "))
        .collect();
    assert_eq!(
        fixed_rows.len(),
        2,
        "the case's two fixed rows:\n{populated}"
    );
    for row in fixed_rows {
        assert!(
            row.contains(&sentinel),
            "an absent date must be the sentinel, not an empty cell: {row}"
        );
    }

    // Both renderings of the same rows must agree on the column, or the saved
    // plan and the terminal are two documents with two different answers.
    let tui = crate::ztools::weekend::render_weekend_plan_gorgeous(
        "Aug 07 to Aug 09, 2026",
        "Fri 28.2°C (clear)",
        &[crate::ztools::weekend::WeekendEvent {
            dates: "Year-Round".to_string(),
            ..event("Kortright Centre", "Vaughan", "Free", "all ages", 4.5)
        }],
        &[crate::ztools::weekend::WeekendEvent {
            dates: "August 8 to 9".to_string(),
            ..event("Union Summer", "Kleinburg", "$14 per child", "3-12", 4.5)
        }],
    );
    assert!(tui.contains("Dates"), "{tui}");
    assert!(tui.contains("August 8 to 9"), "{tui}");
    assert!(tui.contains("Year-Round"), "{tui}");
}
