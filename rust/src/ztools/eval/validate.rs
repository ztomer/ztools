//! Eval validators. Ported from `eval/validate.py`: the file-summary scorer
//! that detects filename inference and generic filler instead of real content
//! detail.
//!
//! WHAT CHANGED ON 2026-10-05, and why it is not a knob. The scorer counted
//! descriptions containing one of [`CONTENT_VERBS`] and bucketed the ratio. That
//! measures SENTENCE SHAPE, and the prompt it graded told the model to rely only
//! on file CONTENT — which the prompt never sent, so the only description
//! available was one guessed from the name. A name guess like "parses the
//! manifest and loads metadata" carries two verbs and scored full marks; a
//! truthful sentence about `rust/src/manifest.rs` ("expands a leading `~` against
//! the home directory") carries none and scored nothing. The metric and the
//! instruction pointed in opposite directions.
//!
//! So for the rows [`grounding`] has truth for, "detailed" now means GROUNDED:
//! the description is checked against what that file actually says. Rows with no
//! pinned truth — the synthetic fixtures, `Check::FileSummary` on any other
//! repo's paths — keep the verb heuristic unchanged, because for those there is
//! nothing better to grade against and a stricter rule would score every
//! existing caller zero.

use regex::Regex;

#[path = "validate_grounding.rs"]
pub mod grounding;

/// A description that is essentially the file's own name re-spaced ("config
/// loader" for `config_loader.py`) is filename inference, not file reading, which
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

pub(crate) fn has_text_headers(text: &str) -> bool {
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
#[must_use]
pub fn validate_file_summary(raw: &str) -> (u8, String) {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return (0, "empty response".to_string());
    }

    let parsed: Option<serde_json::Value> = serde_json::from_str(trimmed).ok();
    match parsed {
        Some(serde_json::Value::Array(items)) => validate_list(&items),
        Some(serde_json::Value::Object(map)) => validate_parsed(&map),
        Some(_) | None => validate_raw_string(trimmed),
    }
}

/// The score ladder, shared by the list and dict branches.
///
/// `judged` is the number of descriptions that earned the row: GROUNDED for a
/// row with pinned truth, verb-bearing otherwise. An echo or a generic filler
/// never reaches it, which is how `echo_count`/`generic_count` reduce the score
/// rather than only annotating it.
const fn ladder(judged: usize, num_files: usize) -> (u8, bool) {
    if judged * 10 >= num_files * 8 {
        (100, false)
    } else if judged * 2 >= num_files {
        (85, false)
    } else if judged >= 2 {
        (70, false)
    } else if judged >= 1 {
        (50, false)
    } else {
        (25, true)
    }
}

/// The score ladder for a list answer: `detailed` was GROUNDED for a row with
/// pinned truth, verb-bearing otherwise.
///
/// The dict shape keeps its OWN ladder, lower at every rung. That is the ported
/// shape and it is not negotiable here: the dict form is the weaker ANSWER FORMAT
/// (paths as keys, no description field), and collapsing its rungs into the list
/// ones would have moved `dict_score_ladder_middle_rungs` from 55 to 70 for
/// reasons that have nothing to do with grounding.
const fn dict_ladder(judged: usize, num_files: usize) -> (u8, bool) {
    if judged * 10 >= num_files * 8 {
        (85, false)
    } else if judged * 2 >= num_files {
        (70, false)
    } else if judged >= 2 {
        (55, false)
    } else if judged >= 1 {
        (40, false)
    } else {
        (25, true)
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
    let mut grounded_count = 0usize;
    let mut ungrounded_count = 0usize;

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

        match grounding::facts_for(path) {
            Some(facts) => {
                if grounding::grounded(facts, &desc_lower) {
                    grounded_count += 1;
                } else {
                    ungrounded_count += 1;
                }
            }
            None => {
                if CONTENT_VERBS.iter().any(|kw| desc_lower.contains(kw)) {
                    detailed_count += 1;
                }
            }
        }
    }

    if num_files == 0 {
        return (0, "no items".to_string());
    }

    let (score, no_detail) = ladder(grounded_count + detailed_count, num_files);
    if no_detail {
        failures.push("no content details".to_string());
    }
    if generic_count > 0 {
        failures.push(format!("{generic_count} generic description(s)"));
    }
    if echo_count > 0 {
        failures.push(format!("{echo_count} filename-only description(s)"));
    }
    if ungrounded_count > 0 {
        failures.push(format!(
            "{ungrounded_count} description(s) not supported by the file's content"
        ));
    }

    (score.min(100), failures.join("; "))
}

fn validate_parsed(map: &serde_json::Map<String, serde_json::Value>) -> (u8, String) {
    let mut failures: Vec<String> = Vec::new();
    let num_files = map.len();
    let mut detailed_count = 0usize;
    let mut ungrounded_count = 0usize;

    for (filepath, summary) in map {
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
        match grounding::facts_for(filepath) {
            Some(facts) => {
                if grounding::grounded(facts, &summary_lower) {
                    detailed_count += 1;
                } else {
                    ungrounded_count += 1;
                }
            }
            None => {
                if CONTENT_VERBS.iter().any(|kw| summary_lower.contains(kw)) {
                    detailed_count += 1;
                }
            }
        }
    }

    // The dict shape keeps its own ceiling: it is the weaker ANSWER FORMAT, and
    // that is a property of the format rather than of the grounding.
    let (score, no_detail) = dict_ladder(detailed_count, num_files);
    if no_detail {
        failures.push("no content details".to_string());
    }
    if ungrounded_count > 0 {
        failures.push(format!(
            "{ungrounded_count} description(s) not supported by the file's content"
        ));
    }

    (score, failures.join("; "))
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
#[path = "validate_tests.rs"]
mod tests;
