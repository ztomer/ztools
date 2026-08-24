//! Constant-column (mandated-placeholder) detection.

use crate::ztools::weekend::{flag_constant_columns, WeekendEvent};
use std::collections::HashMap;

use super::support::event;

fn suspects(ages: &str) -> HashMap<String, Vec<String>> {
    HashMap::from([
        (
            "Estimated Price".to_string(),
            vec!["$18-35".to_string(), "18-35".to_string()],
        ),
        ("Duration".to_string(), vec!["2-3 hours".to_string()]),
        ("Target Age(s)".to_string(), vec![ages.to_string()]),
    ])
}

fn rows_with(n: usize, field: &str, value: &str) -> Vec<WeekendEvent> {
    (0..n)
        .map(|i| {
            let mut ev = event(&format!("event {i}"), "Vaughan");
            match field {
                "target_ages" => ev.target_ages = value.into(),
                "price" => ev.price = value.into(),
                "duration" => ev.duration = value.into(),
                _ => {}
            }
            ev
        })
        .collect()
}

/// 5.2's actual failure: every row carrying the configured family range.
#[test]
fn the_shipped_target_age_constant_is_flagged() {
    let notes = flag_constant_columns(&rows_with(5, "target_ages", "6-13"), &suspects("6-13"));
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert!(notes[0].contains("Target Age(s)"), "{notes:?}");
    assert!(notes[0].contains("6-13"), "{notes:?}");
    // The note must say the rows were kept, or a reader will assume a drop.
    assert!(notes[0].to_lowercase().contains("kept"), "{notes:?}");
}

/// The other two shipped instances: Duration and Estimated Price.
#[test]
fn the_shipped_duration_and_price_constants_are_flagged() {
    let mut rows = rows_with(4, "duration", "2-3 hours");
    for ev in rows.iter_mut() {
        ev.price = "$18-35".into();
    }
    let notes = flag_constant_columns(&rows, &suspects("6-13"));
    let joined = notes.join(" ");
    assert!(joined.contains("Duration"), "{notes:?}");
    assert!(joined.contains("Estimated Price"), "{notes:?}");
}

/// A constant column that is NOT a configured value is ordinary, not C4.
#[test]
fn a_constant_that_is_not_a_configured_value_is_not_flagged() {
    assert!(
        flag_constant_columns(&rows_with(4, "target_ages", "all ages"), &suspects("6-13"))
            .is_empty()
    );
}

/// A column that varies is not flagged -- a shared word is not the defect.
#[test]
fn a_column_that_varies_is_not_flagged() {
    let mut rows = rows_with(3, "target_ages", "6-13");
    rows[1].target_ages = "all ages".into();
    rows[2].target_ages = "8+".into();
    assert!(flag_constant_columns(&rows, &suspects("6-13")).is_empty());
}

/// An empty cell is an honest 'unknown'. The guard is load-bearing only when a
/// suspect list itself contains the empty string.
#[test]
fn empty_cells_are_not_a_constant_column() {
    assert!(flag_constant_columns(&rows_with(4, "target_ages", ""), &suspects("6-13")).is_empty());
    let mut pathological = suspects("6-13");
    pathological.insert(
        "Target Age(s)".to_string(),
        vec!["6-13".to_string(), "".to_string()],
    );
    assert!(flag_constant_columns(&rows_with(4, "target_ages", ""), &pathological).is_empty());
}

/// A single row is trivially constant; flagging it would be noise.
#[test]
fn one_row_is_never_a_constant_column() {
    assert!(
        flag_constant_columns(&rows_with(1, "target_ages", "6-13"), &suspects("6-13")).is_empty()
    );
    assert!(flag_constant_columns(&[], &suspects("6-13")).is_empty());
}

/// Matching is case and space insensitive: "6-13 " and "2-3 Hours" are the same
/// answer wearing different clothes.
#[test]
fn matching_is_case_and_space_insensitive() {
    let mut rows = rows_with(3, "target_ages", " 6-13 ");
    for ev in rows.iter_mut() {
        ev.duration = "2-3 Hours".into();
    }
    let joined = flag_constant_columns(&rows, &suspects("6-13")).join(" ");
    assert!(joined.contains("Target Age(s)"), "{joined}");
    assert!(joined.contains("Duration"), "{joined}");
}
