use super::*;
use serde_json::json;

fn v(xs: &[&str]) -> Vec<String> {
    xs.iter().map(|s| (*s).to_string()).collect()
}

#[test]
fn recall_is_the_share_of_planted_items_that_came_back() {
    let items = v(&["חיפה", "212", "186", "4.5%"]);
    assert_eq!(
        validate_recall("חיפה 212 186 4.5%", &items, &[], &[]).0,
        100
    );
    assert_eq!(validate_recall("חיפה 212", &items, &[], &[]).0, 50);
    assert_eq!(validate_recall("שום דבר", &items, &[], &[]).0, 0);
    assert_eq!(validate_recall("", &items, &[], &[]).0, 0);
}

#[test]
fn the_attribution_form_is_thirty_points_and_wants_the_time() {
    let items = v(&["212"]);
    let handles = v(&["dr_amir", "noa_cohen"]);
    let both = "- 212 מקרים (@dr_amir | 09:31)\n- עוד (@noa_cohen | 08:12)";
    assert_eq!(validate_recall(both, &items, &handles, &[]).0, 100);
    let one = "- 212 מקרים (@dr_amir | 09:31)\n- עוד @noa_cohen";
    assert_eq!(validate_recall(one, &items, &handles, &[]).0, 85);
    let none = "- 212 מקרים לפי dr_amir";
    assert_eq!(validate_recall(none, &items, &handles, &[]).0, 70);
    // A handle in the form but with no time is not the form.
    assert_eq!(
        validate_recall("212 (@dr_amir)", &items, &handles, &[]).0,
        70
    );
}

#[test]
fn a_forbidden_token_costs_fifteen_and_the_score_floors_at_zero() {
    let items = v(&["212"]);
    let forbidden = v(&["Hartley", "9000"]);
    assert_eq!(
        validate_recall("212 Hartley", &items, &[], &forbidden).0,
        85
    );
    assert_eq!(
        validate_recall("212 Hartley 9000", &items, &[], &forbidden).0,
        70
    );
    assert_eq!(
        validate_recall("Hartley 9000", &items, &[], &forbidden).0,
        0
    );
}

#[test]
fn json_fields_count_by_key_and_value_wherever_the_object_sits() {
    let expected = vec![
        ("title".to_string(), "גן".to_string()),
        ("price_nis".to_string(), "120".to_string()),
        ("date".to_string(), "2027-03-14".to_string()),
        ("title".to_string(), "ים".to_string()),
    ];
    let full = json!({"events": [
        {"title": "הגן של רוזה", "price_nis": 120, "date": "2027-03-14"},
        {"title": "שיר ים"}
    ]});
    assert_eq!(validate_json_fields(&full, &expected).0, 100);
    let half = json!([{"title": "הגן של רוזה", "price_nis": "119"}]);
    assert_eq!(validate_json_fields(&half, &expected).0, 25);
    assert_eq!(validate_json_fields(&json!("not json"), &expected).0, 0);
    // The right value under the wrong key is not a hit.
    assert_eq!(
        validate_json_fields(&json!({"venue": "120"}), &expected).0,
        0
    );
}
