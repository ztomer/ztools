//! The file-summary scorer's controls, moved here for the 500-line cap.
//
// The two ladder rungs and the generic/echo counters are the port's shape and
// are unchanged. The grounding controls are new: they are the evidence that a
// name-based guess and a description written from the file no longer earn the
// same score.

use super::*;

#[test]
fn empty_data() {
    assert_eq!(validate_file_summary(""), (0, "empty response".to_string()));
}

#[test]
fn empty_list() {
    assert_eq!(validate_file_summary("[]"), (0, "no items".to_string()));
}

#[test]
fn few_files_flagged() {
    let (score, msg) = validate_file_summary(r#"[{"path": "a.py", "desc": "does stuff"}]"#);
    assert!(msg.contains("only"));
    assert!(score <= 25);
}

#[test]
fn all_detailed_scores_100() {
    let data = r#"[
        {"path": "a.py", "desc": "parses config files and loads settings"},
        {"path": "b.py", "desc": "validates JSON output format"},
        {"path": "c.py", "desc": "fetches data from external API"},
        {"path": "d.py", "desc": "handles error processing logic"}
    ]"#;
    let (score, msg) = validate_file_summary(data);
    assert_eq!(score, 100);
    assert_eq!(msg, "");
}

#[test]
fn no_content_details() {
    let data = r#"[
        {"path": "a.py", "desc": "some file"},
        {"path": "b.py", "desc": "another file"}
    ]"#;
    let (score, msg) = validate_file_summary(data);
    assert_eq!(score, 25);
    assert!(msg.contains("no content details"));
}

#[test]
fn dict_input_scores_85() {
    let data = r#"{"main.py": "parses input data", "utils.py": "validates output"}"#;
    let (score, msg) = validate_file_summary(data);
    assert_eq!(score, 85);
    assert_eq!(msg, "");
}

#[test]
fn generic_description_counted() {
    // "a python script" is generic; the others are content verbs.
    let data = r#"[
        {"path": "a.py", "desc": "a python script"},
        {"path": "b.py", "desc": "parses config files and loads settings"},
        {"path": "c.py", "desc": "validates JSON output format"},
        {"path": "d.py", "desc": "fetches data from external API"}
    ]"#;
    let (score, msg) = validate_file_summary(data);
    // 3/4 detailed >= 0.5 -> 85
    assert_eq!(score, 85);
    assert!(msg.contains("1 generic description(s)"));
}

#[test]
fn filename_echo_counted() {
    let data = r#"[
        {"path": "config_loader.py", "desc": "config loader"},
        {"path": "b.py", "desc": "validates JSON output format"},
        {"path": "c.py", "desc": "fetches data from external API"},
        {"path": "d.py", "desc": "handles error processing logic"}
    ]"#;
    let (score, msg) = validate_file_summary(data);
    assert_eq!(score, 85);
    assert!(msg.contains("1 filename-only description(s)"));
}

#[test]
fn string_with_headers_scores_20() {
    let content =
        "## Main module\nhandles configuration and api calls\n## Utils\nvalidation helpers";
    let (score, msg) = validate_file_summary(content);
    assert_eq!(score, 20);
    assert!(msg.contains("no headers"));
}

#[test]
fn string_too_short() {
    let (_, msg) = validate_file_summary("hello");
    assert!(msg.contains("no headers"));
}

#[test]
fn non_dict_items_skipped_but_counted() {
    let data = r#"[
        "string item", 42, null,
        {"path": "a.py", "desc": "parses config files"},
        {"path": "b.py", "desc": "validates JSON output format"},
        {"path": "c.py", "desc": "fetches data from external API"},
        {"path": "d.py", "desc": "handles error processing logic"}
    ]"#;
    let (score, _) = validate_file_summary(data);
    // 4 detailed / 7 total >= 0.5 -> 85
    assert_eq!(score, 85);
}

#[test]
fn scalar_json_input_uses_the_raw_string_scorer() {
    // A bare JSON number parses but is neither array nor object.
    let (score, msg) = validate_file_summary("42");
    assert_eq!(score, 20, "clamped floor for a headerless short string");
    assert!(msg.contains("no headers"));
}

#[test]
fn items_missing_path_or_desc_are_skipped_but_still_counted() {
    let data = r#"[
        {"path": "", "desc": "parses config files"},
        {"path": "b.py", "desc": ""},
        {"path": "c.py", "desc": "validates JSON output format"},
        {"path": "d.py", "desc": "fetches data from external API"},
        {"path": "e.py", "desc": "handles error processing logic"}
    ]"#;
    let (score, _) = validate_file_summary(data);
    // num_files=5 (skipped items still count), detailed=3 -> 3*2 >= 5 -> 85
    assert_eq!(score, 85);
}

#[test]
fn list_score_ladder_middle_rungs() {
    // 7 files, only 2 detailed: misses the 85 gate, hits the >=2 rung -> 70.
    let data_70 = r#"[
        {"path": "a.py", "desc": "parses config files"},
        {"path": "b.py", "desc": "validates JSON output format"},
        {"path": "c.py", "desc": "some file"},
        {"path": "d.py", "desc": "another file"},
        {"path": "e.py", "desc": "more filler text here"},
        {"path": "f.py", "desc": "yet another entry"},
        {"path": "g.py", "desc": "and one more"}
    ]"#;
    let (score, _) = validate_file_summary(data_70);
    assert_eq!(score, 70);

    // 4 files, only 1 detailed: -> 50.
    let data_50 = r#"[
        {"path": "a.py", "desc": "parses config files"},
        {"path": "b.py", "desc": "some file"},
        {"path": "c.py", "desc": "another file"},
        {"path": "d.py", "desc": "one more"}
    ]"#;
    let (score, _) = validate_file_summary(data_50);
    assert_eq!(score, 50);
}

#[test]
fn dict_entries_that_cannot_be_summarized_are_skipped() {
    // Empty key, non-string value and empty summary all skip; the rest counts.
    let data = r#"{"": "parses input data", "utils.py": 42, "notes.md": "", "main.py": "validates output"}"#;
    let (score, msg) = validate_file_summary(data);
    assert_eq!(score, 40);
    assert_eq!(msg, "");
}

#[test]
fn dict_with_no_usable_summaries_scores_the_floor() {
    let data = r#"{"utils.py": 42, "notes.md": ""}"#;
    let (score, msg) = validate_file_summary(data);
    assert_eq!(score, 25);
    assert!(msg.contains("no content details"));
}

#[test]
fn dict_score_ladder_middle_rungs() {
    // 5 entries, 2 detailed: -> 55.
    let data_55 = r#"{
        "a.py": "parses data",
        "b.py": "validates output",
        "c.py": "stuff",
        "d.py": "things",
        "e.py": "junk"
    }"#;
    let (score, _) = validate_file_summary(data_55);
    assert_eq!(score, 55);

    // 2 entries, 1 detailed: -> 70.
    let data_70 = r#"{"main.py": "parses input data", "utils.py": "nothing much"}"#;
    let (score, _) = validate_file_summary(data_70);
    assert_eq!(score, 70);
}

#[test]
fn long_markdown_body_scores_headers_plus_length() {
    let mut body = String::from("## Sections\n\n");
    body.push_str(&"word ".repeat(60));
    let (score, msg) = validate_file_summary(&body);
    assert_eq!(score, 40, "20 headers + 20 length");
    assert_eq!(msg, "");
    assert!(body.chars().count() >= 200);
}
