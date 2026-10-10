//! Same-event merging, on the rows the 2026-10-10 replay of the Thanksgiving
//! corpus actually structured (name, location, dates, price and ages
//! verbatim), plus the near misses that must stay two rows.

use super::*;

fn row(
    name: &str,
    location: &str,
    start: &str,
    end: &str,
    price: &str,
    ages: &str,
) -> WeekendEvent {
    WeekendEvent {
        name: name.into(),
        location: location.into(),
        price: price.into(),
        target_ages: ages.into(),
        day: String::new(),
        dates: start.into(),
        description: String::new(),
        is_transient: true,
        score: 0.0,
        start_date: start.into(),
        end_date: end.into(),
        weather: String::new(),
        duration: String::new(),
    }
}

fn names(rows: &[WeekendEvent]) -> Vec<&str> {
    rows.iter().map(|r| r.name.as_str()).collect()
}

/// THE CLASS, from the replay: one workshop under two names, the second only
/// adding where it is. One row survives, the richer one, and nothing it did
/// not know is lost.
#[test]
fn the_same_event_twice_is_one_row() {
    let mut first = row(
        "Robotics Workshop For Kids",
        "Aloft by Marriott Vaughan Mills, Vaughan, ON",
        "2026-10-10",
        "2026-10-10",
        "Free",
        "7-14yrs",
    );
    first.description = "Family-friendly hands-on robotics workshop for kids in Vaughan.".into();
    let mut second = row(
        "Robotics Workshop For Kids at Vaughan Mills",
        "Vaughan, ON",
        "2026-10-10",
        "2026-10-10",
        "Free",
        "7-14yrs",
    );
    second.duration = "from 11:00 AM".into();
    let (kept, notes) = merge_duplicate_events(vec![first, second], 2026);
    assert_eq!(
        names(&kept),
        vec!["Robotics Workshop For Kids"],
        "{notes:?}"
    );
    assert_eq!(
        kept[0].location,
        "Aloft by Marriott Vaughan Mills, Vaughan, ON"
    );
    assert_eq!(
        kept[0].duration, "from 11:00 AM",
        "the twin's value is kept"
    );
    assert_eq!(notes.len(), 1);
    assert!(
        notes[0].contains("Robotics Workshop For Kids at Vaughan Mills"),
        "{notes:?}"
    );
}

/// The richer row wins whichever comes first, and the market whose name only
/// adds its street is the same market on the same day.
#[test]
fn the_richer_twin_is_kept_whatever_the_order() {
    let poor = row(
        "Markham Farmers' Market",
        "Markham",
        "2026-10-10",
        "",
        "unknown",
        "unknown",
    );
    let mut rich = row(
        "Main Street Markham Farmers Market",
        "Main Street Markham, Markham",
        "2026-10-10",
        "2026-10-10",
        "Free",
        "",
    );
    rich.description = "Produce, local arts & crafts".into();
    let (kept, _) = merge_duplicate_events(vec![poor, rich], 2026);
    assert_eq!(names(&kept), vec!["Main Street Markham Farmers Market"]);
    assert_eq!(kept[0].price, "Free");
}

/// Near misses that are two events: the same name in two places, the same
/// name on two days that do not overlap (the sources disagree; neither is
/// verifiably wrong), and two different events at one mall on one weekend.
#[test]
fn different_events_are_never_merged() {
    let rows = vec![
        row(
            "Fall Harvest Market",
            "Sugar Beach, Toronto",
            "2026-10-09",
            "2026-10-12",
            "",
            "",
        ),
        row(
            "Fall Harvest Market",
            "Country Heritage Park, Milton",
            "2026-10-10",
            "",
            "",
            "",
        ),
        row(
            "Markham Farmers Market",
            "Main Street Markham",
            "2026-10-10",
            "",
            "",
            "",
        ),
        row(
            "Markham Farmers Market",
            "Main Street Markham",
            "2026-10-11",
            "",
            "",
            "",
        ),
        row(
            "Thanksgiving Family Festival",
            "Erin Mills Town Centre, Mississauga",
            "2026-10-10",
            "2026-10-11",
            "",
            "",
        ),
        row(
            "Kids Craft & Play",
            "Erin Mills Town Centre, Mississauga",
            "2026-10-10",
            "2026-10-11",
            "",
            "",
        ),
    ];
    let (kept, notes) = merge_duplicate_events(rows.clone(), 2026);
    assert_eq!(kept.len(), rows.len(), "{notes:?}");
    assert_empty!(notes);
}

/// A row with no date, or no venue, cannot contradict its twin on either: an
/// undated "Pumpkins After Dark" is the dated one.
#[test]
fn an_unknown_date_or_venue_does_not_split_a_twin() {
    let rows = vec![
        row(
            "Pumpkins After Dark",
            "Milton",
            "2026-10-09",
            "2026-10-12",
            "",
            "2-12yrs",
        ),
        row("Pumpkins After Dark", "", "", "", "", ""),
    ];
    let (kept, notes) = merge_duplicate_events(rows, 2026);
    assert_eq!(kept.len(), 1, "{notes:?}");
    assert_eq!(kept[0].target_ages, "2-12yrs");
}
