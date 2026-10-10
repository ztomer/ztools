//! The three suitability gates, each tested at the CLASS level: the rows are
//! the ones the Thanksgiving 2026 plan actually shipped, plus the near misses
//! a gate must not catch.

use super::*;

fn row(name: &str, location: &str, target_ages: &str, description: &str) -> WeekendEvent {
    WeekendEvent {
        name: name.into(),
        location: location.into(),
        price: String::new(),
        target_ages: target_ages.into(),
        day: "Sunday".into(),
        dates: "2026-10-11".into(),
        description: description.into(),
        is_transient: true,
        score: 0.0,
        start_date: "2026-10-11".into(),
        end_date: String::new(),
        weather: String::new(),
        duration: String::new(),
    }
}

const FAMILY: &[u32] = &[14, 11, 6];

fn names(rows: &[WeekendEvent]) -> Vec<&str> {
    rows.iter().map(|r| r.name.as_str()).collect()
}

#[test]
fn ranges_are_read_from_age_phrases_not_from_any_number() {
    assert_eq!(
        stated_range("Baby and Me (birth to 12 months)"),
        Some((0, 1))
    );
    assert_eq!(stated_range("Lap time, 18-36 months"), Some((1, 3)));
    assert_eq!(
        stated_range("Family Fun Outdoors (birth to 6 years old)"),
        Some((0, 6))
    );
    assert_eq!(stated_range("Robotics for ages 7-14"), Some((7, 14)));
    assert_eq!(
        stated_range("sessions for ages 3-5 and ages 9-12"),
        Some((3, 12))
    );
    assert_eq!(stated_range("Trivia night, 19+"), Some((19, OPEN_END)));
    assert_eq!(stated_range("Climbing, ages 5 and up"), Some((5, OPEN_END)));
    assert_eq!(stated_range("Free for kids under 5"), Some((0, 4)));
    // Times, prices and dates are not ages.
    assert_eq!(stated_range("Saturday 10am-2pm, $5, Oct 10-12"), None);
}

#[test]
fn audience_words_mean_ages_unless_children_are_welcome_too() {
    assert_eq!(stated_range("Baby Adventures Storytime"), Some((0, 1)));
    assert_eq!(
        stated_range("Cercle de discussion / French Meetup for Adults"),
        Some((18, OPEN_END))
    );
    assert_eq!(stated_range("Toastmasters for Teens"), Some((13, 19)));
    assert_eq!(
        stated_range("Mud Kitchen at EarlyON Fred Armstrong Parkette"),
        Some((0, 6))
    );
    assert_eq!(stated_range("Seniors coffee social"), Some((60, OPEN_END)));
    assert_eq!(stated_range("Young Adult book club"), Some((12, 18)));
    // Welcoming words win over an audience word.
    assert_eq!(stated_range("Pumpkin patch for kids and adults"), None);
    assert_eq!(stated_range("Family baby-sign class"), None);
    // Word starts only, and a price is not an audience.
    assert_eq!(stated_range("Canteen open all day"), None);
    assert_eq!(stated_range("Admission $10/adult"), None);
    assert_eq!(stated_range("Senior Kindergarten open house"), None);
}

#[test]
fn a_declared_field_reads_bare_numbers() {
    assert_eq!(declared_range("6-12"), Some((6, 12)));
    assert_eq!(declared_range("7-14 yrs"), Some((7, 14)));
    assert_eq!(declared_range("6,10"), Some((6, 10)));
    assert_eq!(declared_range("5+"), Some((5, OPEN_END)));
    assert_eq!(declared_range("Adults"), Some((18, OPEN_END)));
    assert_eq!(declared_range("All Ages"), None);
    assert_eq!(declared_range("unknown"), None);
}

/// The class: a row whose stated audience excludes every child is dropped,
/// whether the audience was in the field, the name or the description — and a
/// row that states nothing is kept, because unknown is not unsuitable.
#[test]
fn rows_that_fit_none_of_the_children_are_dropped_from_any_source() {
    let rows = vec![
        row(
            "Baby and Me (birth to 12 months) at EarlyON Woodbridge Centre",
            "",
            "",
            "",
        ),
        row(
            "Cercle de discussion / French Meetup for Adults",
            "",
            "",
            "",
        ),
        row("Baby Adventures Storytime", "", "", ""),
        row("Story hour", "", "0-3", ""),
        row("Wine walk", "", "", "A guided tasting, 19+ only"),
        row("Toastmasters for Teens", "", "", ""),
        row(
            "Songs, Rhymes, and Stories (birth to 6 years old)",
            "",
            "",
            "",
        ),
        row("Sing and Sign", "", "", ""),
        row("Robotics Workshop For Kids", "", "7-14 yrs", ""),
    ];
    let (kept, notes) = drop_unsuitable_for_ages(rows, FAMILY);
    assert_eq!(
        names(&kept),
        vec![
            "Toastmasters for Teens",
            "Songs, Rhymes, and Stories (birth to 6 years old)",
            "Sing and Sign",
            "Robotics Workshop For Kids",
        ]
    );
    assert_eq!(notes.len(), 5, "{notes:?}");
    assert!(
        notes[1].contains("for ages 18+") && notes[1].contains("(14,11,6)"),
        "{}",
        notes[1]
    );
    assert!(notes[3].contains("for ages 0-3"), "{}", notes[3]);
}

#[test]
fn every_stated_range_must_admit_the_child() {
    // The field says 6-12 but the name says babies: the sources disagree and
    // no child satisfies both, so the row goes.
    let contradictory = row("Baby Sensory Hour", "", "6-12", "");
    assert_eq!(children_who_fit(&contradictory, FAMILY), Some(0));
    assert_eq!(
        children_who_fit(&row("Sing and Sign", "", "", ""), FAMILY),
        None
    );
    // With no family there is nothing to judge against: nothing is dropped.
    let (kept, notes) = drop_unsuitable_for_ages(vec![contradictory], &[]);
    assert_eq!((kept.len(), notes.len()), (1, 0));
}

#[test]
fn listing_page_titles_are_recognised_as_a_class() {
    for title in [
        "Vaughan Events This Weekend & Things to Do - Oct 2026",
        "Things to Do in Toronto This Weekend",
        "What's On in Vaughan",
        "Family Events in Vaughan",
        "Upcoming Events | City of Vaughan",
        "Kids Activities (October 2026)",
        "Top 10 Fall Fairs",
    ] {
        assert!(is_listing_page_title(title), "{title}");
    }
    for venue in [
        "Kortright Centre for Conservation",
        "Aloft by Marriott Vaughan Mills, Vaughan, ON",
        "Heroes World, Richmond Hill, ON",
        "Vaughan, Ontario",
        "Maple Leaf Gardens",
    ] {
        assert!(!is_listing_page_title(venue), "{venue}");
    }
}

/// The places of `conf/weekend.toml [region]` the real titles below name.
fn places() -> Vec<String> {
    [
        "vaughan",
        "toronto",
        "markham",
        "richmond hill",
        "woodbridge",
        "mississauga",
    ]
    .map(String::from)
    .to_vec()
}

/// A NAME that names no event: every word of it is a place, a listing word, a
/// season or holiday, or a date. Each title below reached a plan's transient
/// rows on 2026-10-10 (the parent run, then a replay of its corpus); the first
/// two passed the phrase list, because neither carries "things to do".
/// The events beside them -- several as generic-sounding -- must stand: what
/// separates them is one word that is not listing vocabulary ("fair",
/// "market", "festival", "pumpkins"), and a singular event noun is not one.
#[test]
fn a_name_that_names_no_event_is_a_listing_page() {
    let places = places();
    for title in [
        "Richmond Hill, ON",
        "Best Thanksgiving Events and Fall Festivals Near Toronto 2026",
        "Toronto Events",
        "Family Fun Toronto",
        "Toronto Thanksgiving Long Weekend",
        "Richmond Hill Activities",
        "8 Things to Do in Toronto This Thanksgiving Long Weekend",
        "Thanksgiving Weekend Toronto 2026: 15 Things to Do",
        // A followed page's category labels, drafted as events on the replay.
        "This Week",
        "All Categories",
        "Free Events",
        "Music & Concerts",
        "Arts & Culture",
        "Community",
        "Family & Kids",
    ] {
        assert!(names_no_event(title, &places), "{title} names no event");
    }
    for event in [
        "Woodbridge Fall Fair",
        "Pumpkins After Dark",
        "Markham Farmers' Market",
        "Markham Farmers Market",
        "Thanksgiving Family Festival",
        "Fall Harvest Market",
        "Kids Craft & Play",
        "Robotics Workshop For Kids",
        "RHGA Member Gallery Show and Sale",
        "Halloween Haunt at Canada's Wonderland",
        "Screemers",
        "Toastmasters for Teens",
        "Sing and Sign",
        "Little Explorers Storytime",
        "Community Harvest Festival",
        "Food Truck Festival",
    ] {
        assert!(!names_no_event(event, &places), "{event} is an event");
    }
    assert!(
        !names_no_event("", &places),
        "an empty name is not a listing title"
    );

    // The gate drops such a row and keeps the events.
    let rows = vec![
        row("Richmond Hill, ON", "Richmond Hill, ON", "", ""),
        row("Woodbridge Fall Fair", "Woodbridge", "", ""),
        row(
            "Best Thanksgiving Events and Fall Festivals Near Toronto 2026",
            "Toronto",
            "",
            "",
        ),
        row("Pumpkins After Dark", "Milton", "", ""),
    ];
    let (kept, notes, dropped) = reject_listing_page_titles(rows, &places);
    assert_eq!(
        names(&kept),
        vec!["Woodbridge Fall Fair", "Pumpkins After Dark"]
    );
    assert_eq!((dropped, notes.len()), (2, 2), "{notes:?}");
    assert_eq!(
        kept[0].location, "Woodbridge",
        "a place is a venue, not a listing"
    );
}

#[test]
fn a_listing_title_is_cleared_as_a_venue_and_dropped_as_an_event() {
    let rows = vec![
        row(
            "Chess Club",
            "Vaughan Events This Weekend & Things to Do - Oct 2026",
            "",
            "",
        ),
        row("Things to Do in Vaughan This Weekend", "Vaughan", "", ""),
        row(
            "Robotics Workshop",
            "Aloft by Marriott Vaughan Mills",
            "",
            "",
        ),
    ];
    let (kept, notes, dropped) = reject_listing_page_titles(rows, &places());
    assert_eq!(names(&kept), vec!["Chess Club", "Robotics Workshop"]);
    assert_eq!(
        kept[0].location, "",
        "the page title must not stand as a venue"
    );
    assert_eq!(kept[1].location, "Aloft by Marriott Vaughan Mills");
    assert_eq!(dropped, 1);
    assert_eq!(notes.len(), 2, "{notes:?}");
}

#[test]
fn a_transient_row_that_is_only_a_fixed_venue_is_dropped() {
    let fixed = vec![row("Kortright Centre for Conservation", "Vaughan", "", "")];
    let transient = vec![
        row(
            "Kortright Centre for Conservation",
            "Vaughan, Ontario",
            "",
            "",
        ),
        row("Kortright Centre", "", "", ""),
        row("Kortright Maple Syrup Festival", "", "", ""),
        row("Chess Club", "", "", ""),
    ];
    let (kept, notes) = drop_duplicates_of_fixed(transient, &fixed);
    assert_eq!(
        names(&kept),
        vec!["Kortright Maple Syrup Festival", "Chess Club"]
    );
    assert_eq!(notes.len(), 2);
    assert!(notes[0].contains("fixed venue 'Kortright Centre for Conservation'"));
}
