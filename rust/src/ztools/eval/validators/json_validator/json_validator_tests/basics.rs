//! Sanity tests: empty/valid input for validate_json and validate_detailed_json.

use serde_json::json;

use crate::ztools::eval::validators::defects::{
    GENERIC_LOCATION_MAX_SCORE, NEAR_DUPLICATE_MAX_SCORE,
};
use crate::ztools::eval::validators::json_validator::*;

#[test]
fn test_validate_json_empty() {
    let (score, reason) = validate_json(&json!({}), "");
    assert_eq!(score, 0);
    assert_eq!(reason, "no items found");
}

#[test]
fn test_validate_json_valid_list() {
    let data = json!({
        "activities": [
            {"name": "High Park Zoo", "location": "1873 Bloor St W", "price": "Free"},
            {"name": "Ripley's Aquarium", "location": "288 Bremner Blvd", "price": "$45"},
            {"name": "Ontario Science Centre", "location": "770 Don Mills Rd", "price": "$22"},
            {"name": "Royal Ontario Museum", "location": "100 Queens Park", "price": "$26"},
            {"name": "Riverdale Farm", "location": "201 Winchester St", "price": "Free"},
            {"name": "Toronto Zoo", "location": "2000 Meadowvale Rd", "price": "$30"},
            {"name": "Centerville Theme Park", "location": "Centre Island", "price": "$40"},
            {"name": "Allan Gardens", "location": "19 Horticultural Ave", "price": "Free"}
        ]
    });
    let (score, _) = validate_json(&data, "");
    assert_eq!(score, 75); // structure (20) + count (25) + validity (30) = 75 (no source given)
}

#[test]
fn test_validate_detailed_json_detects_generic_location() {
    let data = json!({
        "fixed_activities": [
            {"name": "Activity 1", "location": "Indoor venue", "target_ages": "all", "price": "Free"},
            {"name": "Activity 2", "location": "Outdoor location", "target_ages": "all", "price": "Free"},
            {"name": "Activity 3", "location": "Indoor space", "target_ages": "all", "price": "Free"},
            {"name": "Activity 4", "location": "Venue", "target_ages": "all", "price": "Free"},
            {"name": "Activity 5", "location": "Place", "target_ages": "all", "price": "Free"}
        ]
    });
    let (score, reason) = validate_detailed_json(&data, "");
    assert!(score <= GENERIC_LOCATION_MAX_SCORE);
    assert!(
        reason.contains("generic placeholders"),
        "reason was: {reason}"
    );
}

#[test]
fn test_validate_detailed_json_detects_near_duplicates_with_acronym() {
    let data = json!({
        "fixed_activities": [
            {"name": "Royal Ontario Museum", "location": "Toronto", "price": "$20"},
            {"name": "The ROM", "location": "Toronto", "price": "$20"},
            {"name": "Ontario Science Centre", "location": "Toronto", "price": "$20"},
            {"name": "The OSC", "location": "Toronto", "price": "$20"},
            {"name": "Toronto Zoo", "location": "Toronto", "price": "$20"}
        ]
    });
    let (score, reason) = validate_detailed_json(&data, "");
    assert!(score <= NEAR_DUPLICATE_MAX_SCORE);
    assert!(
        reason.contains("repeat an earlier venue"),
        "reason was: {reason}"
    );
}
