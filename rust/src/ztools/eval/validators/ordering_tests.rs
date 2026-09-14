//! Cross-validator ordering: no validator may rank bad output above good.
//!
//! Port of `test_validator_ordering.py`. Every scoring bug this repo has hit
//! was an ordering failure wearing a plausible number, so these assert the
//! ORDER, not the number — they survive rescoring but fail the moment a
//! validator inverts.

use super::*;
use serde_json::json;

const GOOD_SUMMARY: &str = "## Executive Summary\nFunding and model releases dominated the week, with inference cost recurring\nacross threads. Participants reported benchmarks and confirmed pricing.\n\n## Funding\n- Series B closed at $40M (@TechCrunch | Mar 15 08:00)\n- Follow-on announced for infrastructure (@benedictevans | Mar 15 09:30)\n\n## Models\n- Lower latency confirmed (@simonw | Mar 16 11:05)\n- Early evaluation numbers shared (@karpathy | Mar 16 14:20)\n";
const BAD_SUMMARY: &str = "stuff happened. things were said. no idea who or when.";

const ORDER_SOURCE: &str = "Spring Festival at Vaughan Mills on Saturday, free admission, all ages, outdoor. Winter Market at Dufferin Grove on Sunday, $10 entry, ages 6-12, indoor.";

#[test]
fn summary_ranks_good_above_bad() {
    let (good, _) = validate_summary(GOOD_SUMMARY, "");
    let (bad, _) = validate_summary(BAD_SUMMARY, "");
    assert!(good > bad, "{good} vs {bad}");
}

#[test]
fn filename_ranks_good_above_bad() {
    let (good, _) = validate_filename("quarterly_revenue_report", "");
    let (bad, _) = validate_filename("Here is the filename: IMG 1234.PNG", "");
    assert!(good > bad, "{good} vs {bad}");
}

#[test]
fn file_summary_ranks_good_above_bad() {
    let good = json!([
        {"path": "lib/config_loader.py", "desc": "Parses TOML config and validates required keys"},
        {"path": "lib/api_client.py", "desc": "Sends chat requests and handles retries"},
        {"path": "lib/report.py", "desc": "Renders scorecards into markdown and writes them"},
        {"path": "lib/extract.py", "desc": "Extracts JSON from noisy model output"},
    ]);
    let bad = json!([
        {"path": "lib/config_loader.py", "desc": "a python script"},
        {"path": "lib/api_client.py", "desc": "a config file"},
        {"path": "lib/report.py", "desc": "report"},
        {"path": "lib/extract.py", "desc": "extract"},
    ]);
    let (good_score, _) = crate::ztools::eval::validate::validate_file_summary(&good.to_string());
    let (bad_score, _) = crate::ztools::eval::validate::validate_file_summary(&bad.to_string());
    assert!(good_score > bad_score, "{good_score} vs {bad_score}");
}

#[test]
fn grounded_json_outscores_invented_by_more_than_double() {
    let grounded = json!([
        {"name": "Spring Festival", "location": "Vaughan Mills", "day": "Saturday",
         "price": "Free", "target_ages": "All", "weather": "outdoor"}
    ]);
    let invented = json!([
        {"name": "Moon Rave", "location": "Atlantis", "day": "Tuesday",
         "price": "$999", "target_ages": "40-50", "weather": "underwater"}
    ]);
    let (grounded_score, _) = validate_detailed_json(&grounded, ORDER_SOURCE);
    let (invented_score, _) = validate_detailed_json(&invented, ORDER_SOURCE);
    assert!(
        invented_score * 2 < grounded_score,
        "{invented_score} vs {grounded_score}"
    );
}
