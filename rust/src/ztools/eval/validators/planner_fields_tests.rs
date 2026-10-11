use serde_json::json;

use super::*;

/// Each closed field is judged by the planner's own function: a value it
/// keeps costs nothing, "unknown" and empty cost nothing, anything it throws
/// away is named.
#[test]
fn a_field_the_planner_discards_is_named_and_an_honest_unknown_is_not() {
    let clean = json!({"name": "Puppet Show", "price": "$5", "duration": "2 hours",
        "for_families": "yes", "weather": "indoor", "place_kind": "library"});
    assert_eq!(field_defects(&clean), Vec::<String>::new());
    let unknown = json!({"name": "x", "price": "unknown", "duration": "",
        "for_families": "unknown", "weather": "unknown", "place_kind": "unknown"});
    assert_eq!(field_defects(&unknown), Vec::<String>::new());
    let garbage = json!({"name": "x", "price": "18", "duration": "outdoor",
        "for_families": "all", "weather": "sunny", "place_kind": "outdoor"});
    assert_eq!(
        field_defects(&garbage),
        [
            "price=18",
            "duration=outdoor",
            "for_families=all",
            "weather=sunny",
            "place_kind=outdoor"
        ]
    );
}

/// The share is fields kept times rows returned; the reasons say which.
#[test]
fn the_share_counts_discarded_fields_and_rows_that_never_came_back() {
    let rows = vec!["Spring Festival".to_string(), "Puppet Show".to_string()];
    let one = json!({"transient_events": [{"name": "Spring Festival",
        "place_kind": "outdoor", "for_families": "all"}]});
    // 3 of 5 fields kept, 1 of 2 rows back: 30.
    let (score, why) = validate_planner_fields(&one, &rows);
    assert_eq!(score, 30, "{why}");
    assert!(
        why.contains("2 of 5 fields") && why.contains("1 of 2 input rows"),
        "{why}"
    );
    // Without rows to expect, only the fields count.
    assert_eq!(validate_planner_fields(&one, &[]).0, 60);
    // No list is the parse gate's business.
    assert_eq!(validate_planner_fields(&json!([]), &rows).0, 100);
}
