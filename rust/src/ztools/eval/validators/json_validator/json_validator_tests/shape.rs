//! Item shape, name matching, and source-grounding extraction.

use serde_json::json;

use crate::ztools::eval::validators::json_validator::*;

#[test]
fn test_extract_list_from_dict_prefers_known_keys_in_order() {
    let data = json!({"items": [1], "events": [1, 2]});
    assert_eq!(extract_list_from_dict(&data).len(), 1);
}

#[test]
fn test_extract_list_from_dict_recurses_into_nested_objects_first() {
    let data = json!({"meta": {"info": {"results": [7, 8]}}, "tags": [1, 2, 3, 4]});
    let found = extract_list_from_dict(&data);
    assert_eq!(found, vec![json!(7), json!(8)]);
}

#[test]
fn test_extract_list_from_dict_falls_back_to_largest_array() {
    let data = json!({"nested": {"x": 5}, "notes": ["a"], "tags": ["x", "y", "z"]});
    assert_eq!(
        extract_list_from_dict(&data),
        vec![json!("x"), json!("y"), json!("z")]
    );
}

#[test]
fn test_extract_list_from_dict_handles_non_objects_and_arrays() {
    assert_eq!(extract_list_from_dict(&json!([1, 2])), vec![json!(1), json!(2)]);
    assert!(extract_list_from_dict(&json!("plain")).is_empty());
    assert!(extract_list_from_dict(&json!(42)).is_empty());
    assert!(extract_list_from_dict(&json!(null)).is_empty());
    assert!(extract_list_from_dict(&json!({"a": 1, "b": "two"})).is_empty());
}

#[test]
fn test_is_valid_list_item_matrix() {
    assert!(is_valid_list_item(&json!("Zoo day")));
    assert!(!is_valid_list_item(&json!("   ")));
    assert!(!is_valid_list_item(&json!(5)));
    assert!(!is_valid_list_item(&json!([1])));
    assert!(is_valid_list_item(&json!({"name": "High Park"})));
    assert!(is_valid_list_item(&json!({"path": "docs/readme"})));
    assert!(!is_valid_list_item(&json!({"name": "", "title": "  "})));
    assert!(!is_valid_list_item(&json!({"other": "x", "desc": ""})));
}

#[test]
fn test_has_item_details_matrix() {
    assert!(!has_item_details(&json!("not an object")));
    assert!(!has_item_details(&json!(null)));
    // no name field but >=2 entries counts as detailed
    assert!(has_item_details(&json!({"time": "3pm", "cost": "$2"})));
    assert!(!has_item_details(&json!({"place": "Here"})));
    // no name field at all and fewer than 2 entries: not detailed
    assert!(!has_item_details(&json!({"time": "3pm"})));
    // name plus a populated detail field
    assert!(has_item_details(&json!({"name": "X", "price": "$5"})));
    // name present but its detail-field siblings are empty: falls back to map size
    assert!(has_item_details(&json!({"name": "X", "desc": ""})));
    assert!(!has_item_details(&json!({"name": "X"})));
}

#[test]
fn test_norm_name_and_tokens() {
    assert_eq!(_norm_name(" High-Park  Zoo! "), "highpark zoo");
    let tokens = _name_tokens("The Big-Foot and 12 Zoo-goers!");
    assert_eq!(
        tokens,
        ["big", "foot", "goers", "zoo"].into_iter().map(String::from).collect()
    );
}

#[test]
fn test_names_match_containment_and_token_overlap() {
    // containment arm
    assert!(_names_match("Central Park", "park"));
    // >=2 shared tokens arm
    assert!(_names_match("Maple Grove Park", "Maple Grove Arena"));
    // empty normalized names never match
    assert!(!_names_match("!!!", "Central Park"));
    assert!(!_names_match("Central Park", "***"));
    // one side has no usable tokens after stopword/length filtering
    assert!(!_names_match("The Who", "Clap Your Hands Say Yeah"));
    // no containment and no long-token anchor
    assert!(!_names_match("Art Cafe", "Cafe Bar"));
    assert!(!_names_match("Waterfront Market", "Marketplace Lofts"));
}

#[test]
fn test_names_match_longest_token_arms() {
    // longest token of A found inside B
    assert!(_names_match("Modern Gallery Annex", "Gallery Bistro"));
    // longest token of B found inside A (symmetric arm)
    assert!(_names_match("Gallery Bistro", "Modern Gallery Annex"));
}

#[test]
fn test_check_source_extraction_guards() {
    let items = vec![json!({"name": "Alpha Beta"})];
    assert_eq!(check_source_extraction(&[], "some source"), 0.0);
    assert_eq!(check_source_extraction(&items, ""), 0.0);
    // every source word is a stopword or under 3 chars -> no terms to match
    assert_eq!(check_source_extraction(&items, "of the a an to"), 0.0);
}

#[test]
fn test_check_source_extraction_object_string_and_other_items() {
    // object item matches via term overlap
    let objects = vec![json!({"name": "Alpha Beta Gamma"})];
    assert_eq!(
        check_source_extraction(&objects, "alpha beta gamma delta report"),
        1.0
    );
    // string item matches via terms...
    let strings = vec![json!("alpha beta")];
    assert_eq!(check_source_extraction(&strings, "alpha beta zone"), 1.0);
    // ...and a non-string item falls back to raw-text containment ("other" arm)
    let others = vec![json!(12345)];
    assert_eq!(check_source_extraction(&others, "route 12345 north"), 1.0);
}

#[test]
fn test_check_source_extraction_skips_empty_text_and_counts_partials() {
    let items = vec![json!({"name": ""}), json!({"name": "Alpha Beta"})];
    let ratio = check_source_extraction(&items, "alpha beta gamma");
    assert!((ratio - 0.5).abs() < 1e-9);
}

#[test]
fn test_check_source_extraction_primary_name_fallbacks() {
    // term overlap is zero ("who" is a stopword) but the normalized primary is in the source
    let who = vec![json!({"name": "The Who"})];
    assert_eq!(check_source_extraction(&who, "tickets for the who tribute night"), 1.0);
    // falls back through event fields when name is absent; single shared term
    // is below the >=2 threshold so the normalized-name containment decides
    let titled = vec![json!({"event": "Winterfolk"})];
    assert_eq!(check_source_extraction(&titled, "the winterfolk lineup"), 1.0);
    let fenice = vec![json!({"title": "La Fenice"})];
    assert_eq!(
        check_source_extraction(&fenice, "an evening at la fenice opera house"),
        1.0
    );
    // normalized search shorter than 4 chars is not searched at all
    let short = vec![json!({"name": "Oh!"})];
    assert_eq!(check_source_extraction(&short, "oh what a night"), 0.0);
}
