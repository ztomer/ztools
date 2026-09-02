//! Eval validators. Ported from `eval/validate.py`: the file-summary scorer
//! that detects filename inference and generic filler instead of real content
//! detail.

use regex::Regex;

/// A description that is essentially the file's own name re-spaced ("config
/// loader" for config_loader.py) is filename inference, not file reading, which
/// is precisely what this task exists to detect.
const GENERIC_DESC_RE: &str = r"^(a|an|the)?\s*(python|shell|bash|config(uration)?|test|helper|utility|source)?\s*(script|file|module|class|program|code)\.?$";

const TEXT_HEADERS_RE: &str = r"(?m)^#{2,}\s+\w+";

const CONTENT_VERBS: &[&str] = &[
    "parse",
    "validat",
    "evaluat",
    "extract",
    "load",
    "save",
    "read",
    "write",
    "fetch",
    "send",
    "process",
    "handle",
    "config",
    "setting",
    "option",
    "parameter",
    "api",
    "client",
    "model",
    "llm",
];

/// A description adds nothing beyond the filename itself when it restates the
/// stem (word-boundary match) and contributes at most 25 extra characters.
fn is_filename_echo(path: &str, desc_lower: &str) -> bool {
    let stem: String = std::path::Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .chars()
        .map(|c| if c == '_' || c == '-' { ' ' } else { c })
        .collect();
    let stem = stem.trim().to_lowercase();

    // Short stems ("a", "cli") match as substrings inside ordinary prose, so
    // require a real word-boundary match on a stem long enough to be meaningful.
    if stem.chars().count() < 4 {
        return false;
    }
    let re = Regex::new(&format!(r"\b{}\b", regex::escape(&stem)))
        .expect("filename-echo regex is static");
    if !re.is_match(desc_lower) {
        return false;
    }
    desc_lower.chars().count() <= stem.chars().count() + 25
}

fn has_text_headers(text: &str) -> bool {
    Regex::new(TEXT_HEADERS_RE)
        .expect("static regex")
        .is_match(text)
}

/// Validate file-summary quality: checks for ACTUAL content detail, not
/// filename inference. Port of `validate_file_summary`.
///
/// Input is the model's raw output text; it is parsed the way the Python
/// caller hands it over: an already-JSON list/dict goes through the structured
/// branches, anything else through the header-heuristic branch.
pub fn validate_file_summary(raw: &str) -> (u8, String) {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return (0, "empty response".to_string());
    }

    let parsed: Option<serde_json::Value> = serde_json::from_str(trimmed).ok();
    match parsed {
        Some(serde_json::Value::Array(items)) => validate_list(&items),
        Some(serde_json::Value::Object(map)) => validate_parsed(map),
        Some(_) => validate_raw_string(trimmed),
        None => validate_raw_string(trimmed),
    }
}

fn validate_list(items: &[serde_json::Value]) -> (u8, String) {
    let mut failures: Vec<String> = Vec::new();
    let generic_desc = Regex::new(GENERIC_DESC_RE).expect("static regex");
    let num_files = items.len();
    if num_files < 4 {
        failures.push(format!("only {num_files} files"));
    }

    let mut detailed_count = 0usize;
    let mut generic_count = 0usize;
    let mut echo_count = 0usize;

    for item in items {
        let Some(obj) = item.as_object() else {
            continue;
        };
        let path = obj.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let desc = obj
            .get("desc")
            .and_then(|v| v.as_str())
            .or_else(|| obj.get("summary").and_then(|v| v.as_str()))
            .unwrap_or("");
        if path.is_empty() || desc.is_empty() {
            continue;
        }
        let desc_lower = desc.to_lowercase();

        let desc_stripped = desc_lower.trim();
        if generic_desc.is_match(desc_stripped) {
            generic_count += 1;
            continue;
        }
        if is_filename_echo(path, desc_stripped) {
            echo_count += 1;
            continue;
        }

        if CONTENT_VERBS.iter().any(|kw| desc_lower.contains(kw)) {
            detailed_count += 1;
        }
    }

    if num_files == 0 {
        return (0, "no items".to_string());
    }

    let score = if detailed_count * 10 >= num_files * 8 {
        100
    } else if detailed_count * 2 >= num_files {
        85
    } else if detailed_count >= 2 {
        70
    } else if detailed_count >= 1 {
        50
    } else {
        failures.push("no content details".to_string());
        25
    };

    if generic_count > 0 {
        failures.push(format!("{generic_count} generic description(s)"));
    }
    if echo_count > 0 {
        failures.push(format!("{echo_count} filename-only description(s)"));
    }

    (score.min(100), failures.join("; "))
}

fn validate_parsed(map: serde_json::Map<String, serde_json::Value>) -> (u8, String) {
    let mut failures: Vec<String> = Vec::new();
    let num_files = map.len();
    let mut detailed_count = 0usize;

    for (filepath, summary) in map.iter() {
        if filepath.is_empty() {
            continue;
        }
        let Some(summary_str) = summary.as_str() else {
            continue;
        };
        if summary_str.is_empty() {
            continue;
        }
        let summary_lower = summary_str.to_lowercase();
        if CONTENT_VERBS.iter().any(|kw| summary_lower.contains(kw)) {
            detailed_count += 1;
        }
    }

    let score = if detailed_count * 10 >= num_files * 8 {
        85
    } else if detailed_count * 2 >= num_files {
        70
    } else if detailed_count >= 2 {
        55
    } else if detailed_count >= 1 {
        40
    } else {
        25
    };

    if detailed_count == 0 {
        failures.push("no content details".to_string());
    }

    (score.min(100), failures.join("; "))
}

fn validate_raw_string(data_str: &str) -> (u8, String) {
    let mut failures: Vec<String> = Vec::new();
    let mut score: u8 = 0;
    if has_text_headers(data_str) {
        score += 20;
    }
    if data_str.chars().count() >= 200 {
        score += 20;
    }
    if score < 40 {
        failures.push("no headers".to_string());
    }
    (score.clamp(20, 100), failures.join("; "))
}

#[cfg(test)]
mod tests {
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
}
