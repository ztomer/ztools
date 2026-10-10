//! Refine is laid back over the draft: nothing leaves without a reason. The
//! draft and refine answers are a replay's, verbatim
//! (`tests/fixtures/weekend/2026-10-10_refine_replay.txt`).

use super::*;

const REPLAY: &str =
    include_str!("../../../../tests/fixtures/weekend/2026-10-10_refine_replay.txt");

fn section(name: &str) -> &'static str {
    let start = REPLAY
        .find(&format!("=== {name}\n"))
        .map(|i| i + name.len() + 5)
        .expect("the fixture has the section");
    let rest = &REPLAY[start..];
    rest.find("\n=== ").map_or(rest, |end| &rest[..end])
}

fn names(rows: &str) -> Vec<String> {
    extracted_rows(rows).iter().map(|r| name_of(r)).collect()
}

/// THE CLASS, on the replay: the draft had 17 entries, refine answered 16 and
/// said nothing about the one it left out. That row comes back, and the
/// operator is told; the 16 refine kept are refine's own rows.
#[test]
fn a_row_refine_leaves_out_without_a_reason_is_restored() {
    let (draft, answer) = (section("draft"), section("refine"));
    assert_eq!(
        (extracted_rows(draft).len(), extracted_rows(answer).len()),
        (17, 16)
    );
    let (rows, notes) = merge_refined(draft, Some(answer));
    let out = names(&rows);
    for name in names(draft) {
        assert!(out.contains(&name), "{name} was lost at refine: {notes:?}");
    }
    assert_eq!(out.len(), 17);
    assert_eq!(
        notes,
        vec![
            "refine left out 'Thanksgiving Weekend Toronto 2026: 15 Things to Do' without a \
             reason; kept from the draft"
                .to_string()
        ]
    );
}

/// The 2026-10-10 run's loss, on the replay's answer: refine's answer without
/// the Sugar Beach market. It is in the window, sourced and for families; it
/// is restored.
#[test]
fn the_sugar_beach_market_cannot_vanish_at_refine() {
    let answer: String = section("refine")
        .lines()
        .filter(|l| !l.starts_with("Fall Harvest Market | Sugar Beach"))
        .collect::<Vec<_>>()
        .join("\n");
    let (rows, notes) = merge_refined(section("draft"), Some(&answer));
    assert!(
        names(&rows).contains(&"Fall Harvest Market".to_string()),
        "{notes:?}"
    );
    assert!(
        notes.iter().any(|n| n.contains("'Fall Harvest Market'")),
        "{notes:?}"
    );
}

/// A removal WITH a reason is the model's judgement and stands, recorded; a
/// removal line with no reason is no reason, and the row stays. A merge is a
/// removal like any other.
#[test]
fn a_reasoned_removal_stands_and_is_recorded() {
    let draft = "Toronto Events | Toronto, ON | October 9, 2026 | unknown | unknown | listing\n\
                 Very Toronto | Toronto, ON | October 9-12, 2026 | unknown | unknown | a website\n\
                 Robotics Workshop For Kids | Aloft by Marriott Vaughan Mills | Oct 10 | Free | 7-14 | robots\n\
                 Robotics Workshop For Kids at Vaughan Mills | Vaughan, ON | Oct 10 | Free | 7-14 | robots";
    let answer = "Robotics Workshop For Kids | Aloft by Marriott Vaughan Mills | Oct 10 | Free | 7-14 | robots\n\
                  DROPPED | Toronto Events | a listing page's title, not an event\n\
                  DROPPED | Robotics Workshop For Kids at Vaughan Mills | merged into Robotics Workshop For Kids\n\
                  DROPPED | Very Toronto |\n\
                  DROPPED | Robotics Workshop For Kids | Duplicate of \"Robotics Workshop For Kids\" entry";
    let (rows, notes) = merge_refined(draft, Some(answer));
    assert_eq!(
        names(&rows),
        vec!["Robotics Workshop For Kids", "Very Toronto"],
        "{notes:?}"
    );
    assert!(notes.contains(
        &"refine dropped 'Toronto Events': a listing page's title, not an event".to_string()
    ));
    assert!(
        notes
            .iter()
            .any(|n| n.starts_with("refine left out 'Very Toronto'")),
        "{notes:?}"
    );
    assert!(
        !rows.contains(DROPPED),
        "a removal line is not a row to structure"
    );
    // The replay's shape: a "removal" of a row the answer kept is no removal.
    assert!(
        !notes
            .iter()
            .any(|n| n.starts_with("refine dropped 'Robotics Workshop For Kids'")),
        "{notes:?}"
    );
    assert_eq!(notes.len(), 3, "{notes:?}");
}

/// An answer that is all commentary keeps the whole draft; no answer at all
/// keeps the draft text as it was.
#[test]
fn an_answer_without_rows_keeps_the_draft() {
    let draft = section("draft");
    let (rows, notes) = merge_refined(draft, Some("I have refined the list as requested."));
    assert_eq!(names(&rows), names(draft));
    assert_eq!(notes.len(), 17);
    assert_eq!(merge_refined(draft, None), (draft.to_string(), Vec::new()));
}
