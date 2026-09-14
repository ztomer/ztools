//! Mixed-signal validators and the two factual probes.
//!
//! Port of the remainder of `lib/validators/text_validator_mixed.py`
//! (`validate_mixed_summary`, `validate_mixed_file_summary`,
//! `validate_mixed_filename`, `validate_factual_accuracy`,
//! `validate_factual_coverage`); the leak / schema / contradiction third of
//! that file lives in [`super::faithfulness`]. The mixed prompts carry a
//! SIGNAL block and, after a literal `NOISE` marker, a block of decoys; the
//! score is recall of the signal against exclusion of the decoys.
//!
//! The two thresholds are CHOSEN, not measured, and each fails in its own
//! safe direction (see the Python module's calibration note): a missed fact
//! under-credits a good summary, a missed falsehood scores a hoax-repeating
//! summary as clean, so coverage is lenient and falsehood detection is not.

use std::collections::HashSet;

use regex::Regex;
use serde_json::Value;
use std::sync::LazyLock;

use super::text_match::{identifying_tokens, phrase_overlap, tokenize};

/// Fraction of a key fact's identifying tokens that must appear for the fact
/// to count as covered (calibrated: 100 / 77 / 16 / 0 on the four controls).
pub const COVERAGE_TOKEN_RATIO: f64 = 0.5;
/// Distinctive tokens of a falsehood that prove it was repeated — a COUNT, so
/// the longest falsehoods are not the hardest to detect.
pub const FALSEHOOD_TOKEN_HITS: usize = 2;

static SENDER_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[@(\w+)\s*\|").unwrap());
static NON_ALNUM_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[^a-z0-9 ]").unwrap());
static NUMBERED_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\d+\.\s*(.+)$").unwrap());

const COMMON: &[&str] = &[
    "about", "above", "after", "again", "their", "there", "these", "those", "would", "could",
    "should", "which", "while", "where", "when", "what", "with", "from", "this", "that", "then",
    "than", "they", "them", "have", "been", "were", "will", "your", "said", "says", "into", "over",
    "also",
];

/// Split a mixed prompt into (signal, noise) on the `NOISE` marker.
fn split_signal_noise(source_text: &str) -> (&str, &str) {
    source_text.split_once("NOISE").unwrap_or((source_text, ""))
}

/// `@sender` handles from `[@Sender | time]:` tweet lines, lowercased.
#[must_use]
pub fn extract_tweet_senders(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| SENDER_RE.captures(line))
        .map(|c| format!("@{}", c[1].to_lowercase()))
        .collect()
}

/// The text after the `- label:` prefix of every noise bullet.
fn parse_noise_entries(noise_part: &str) -> Vec<String> {
    noise_part
        .lines()
        .map(str::trim)
        .filter_map(|line| line.strip_prefix("- "))
        .map(|content| {
            let content = content.trim();
            content
                .split_once(':')
                .map_or(content, |(_, rest)| rest)
                .trim()
                .to_string()
        })
        .filter(|text| !text.is_empty())
        .collect()
}

fn alnum_words(text: &str) -> Vec<String> {
    NON_ALNUM_RE
        .replace_all(&text.to_lowercase(), " ")
        .split_whitespace()
        .map(str::to_string)
        .collect()
}

/// True when at least two distinctive (4+ char) tokens of a noise entry
/// appear in the text.
fn entry_hit(entry: &str, text: &str) -> bool {
    let toks: HashSet<String> = alnum_words(entry)
        .into_iter()
        .filter(|t| t.len() >= 4)
        .collect();
    if toks.is_empty() {
        return false;
    }
    toks.iter().filter(|t| text.contains(t.as_str())).count() >= 2
}

fn as_text(data: &Value) -> String {
    match data {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn is_empty(data: &Value) -> bool {
    match data {
        Value::Null => true,
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
        Value::Bool(b) => !b,
        Value::Number(n) => n.as_f64() == Some(0.0),
    }
}

/// Score tweet-summary filtering by NOISE EXCLUSION.
///
/// Models summarise by content rather than by echoing senders, so
/// contamination is detected by phrase-matching the noise entries. Signal
/// coverage over distinctive (5+ char) content tokens is reported, and a
/// summary that covers none of it is floored at 30.
#[must_use]
pub fn validate_mixed_summary(data: &Value, source_text: &str) -> (i64, String) {
    if is_empty(data) {
        return (0, "empty response".to_string());
    }
    let summary = as_text(data).to_lowercase();
    let (signal_part, noise_part) = split_signal_noise(source_text);
    let noise_entries = parse_noise_entries(noise_part);
    let noise_hits = noise_entries
        .iter()
        .filter(|e| entry_hit(e, &summary))
        .count();
    let noise_total = noise_entries.len();

    let mut score: i64 = 100;
    let mut failures = Vec::new();
    if noise_total > 0 {
        score -= super::super::scoring_math::pct_round(noise_hits, noise_total);
        if noise_hits > 0 {
            failures.push(format!("included {noise_hits}/{noise_total} noise items"));
        }
    }
    let sig_toks: HashSet<String> = alnum_words(signal_part)
        .into_iter()
        .filter(|t| t.len() >= 5 && !COMMON.contains(&t.as_str()))
        .collect();
    if !sig_toks.is_empty() {
        let covered = sig_toks
            .iter()
            .filter(|t| summary.contains(t.as_str()))
            .count();
        failures.push(format!("signal coverage {covered}/{}", sig_toks.len()));
        if covered == 0 {
            score = score.min(30);
        }
    }
    (score, failures.join("; "))
}

/// Absolute paths listed one per line (`/lib/parser.py ...`).
fn extract_file_paths(text: &str) -> Vec<String> {
    text.lines()
        .map(str::trim)
        .filter(|line| line.starts_with('/') && (line.contains('.') || line.contains('/')))
        .filter_map(|line| line.split_whitespace().next())
        .map(str::to_string)
        .collect()
}

/// The paths a file-summary answer accounts for, whatever its shape:
/// markdown `## path` headers, a `{path: desc}` object, or a list of
/// `{path|file: ..}` items / bare strings.
fn file_summary_paths(data: &Value) -> Vec<String> {
    match data {
        Value::String(text) => text
            .lines()
            .map(str::trim)
            .filter_map(|line| line.strip_prefix("##"))
            .map(|h| h.trim().trim_end_matches(':').trim().to_string())
            .filter(|h| !h.is_empty())
            .collect(),
        Value::Object(map) => {
            if map.values().all(Value::is_string) {
                return map.keys().cloned().collect();
            }
            let first_list = map.values().find(|v| v.is_array()).cloned();
            first_list.map_or_else(Vec::new, |list| file_summary_paths(&list))
        }
        Value::Array(items) => items
            .iter()
            .filter_map(|item| match item {
                Value::Object(m) => m
                    .get("path")
                    .or_else(|| m.get("file"))
                    .and_then(|v| v.as_str())
                    .filter(|p| !p.is_empty())
                    .map(str::to_string),
                Value::String(s) => Some(s.clone()),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn prefix_overlap(a: &str, b: &str) -> bool {
    a.contains(b) || b.contains(a)
}

/// Score file-summary filtering: real files summarised, noise files excluded.
/// A string answer that parses as JSON is judged as JSON.
#[must_use]
pub fn validate_mixed_file_summary(data: &Value, source_text: &str) -> (i64, String) {
    if is_empty(data) {
        return (0, "empty response".to_string());
    }
    let parsed: Option<Value> = match data {
        Value::String(s) => serde_json::from_str::<Value>(s).ok(),
        _ => None,
    };
    let data = parsed.as_ref().unwrap_or(data);
    let (signal_part, noise_part) = split_signal_noise(source_text);
    let lower =
        |v: Vec<String>| -> Vec<String> { v.into_iter().map(|p| p.to_lowercase()).collect() };
    let signal_paths = lower(extract_file_paths(signal_part));
    let noise_paths = lower(extract_file_paths(noise_part));
    let output_paths = lower(file_summary_paths(data));
    if output_paths.is_empty() {
        return (0, "no file entries in output".to_string());
    }
    let tp = signal_paths
        .iter()
        .filter(|sp| output_paths.iter().any(|op| prefix_overlap(sp, op)))
        .count();
    let recall = if signal_paths.is_empty() {
        1.0
    } else {
        super::super::scoring_math::ratio(tp, signal_paths.len())
    };
    let fp = output_paths
        .iter()
        .filter(|op| noise_paths.iter().any(|np| prefix_overlap(np, op)))
        .count();
    let precision = super::super::scoring_math::ratio(output_paths.len() - fp, output_paths.len());
    let score = super::super::scoring_math::pct_floor_mean(recall, precision);
    let mut failures = Vec::new();
    if fp > 0 {
        failures.push(format!("included {fp}/{} noise files", noise_paths.len()));
    }
    if !signal_paths.is_empty() && tp < signal_paths.len() {
        failures.push(format!(
            "missed {}/{} real files",
            signal_paths.len() - tp,
            signal_paths.len()
        ));
    }
    (score, failures.join("; "))
}

/// Numbered signal snippets before `NOISE`, bullet noise entries after.
fn extract_mixed_filenames(source_text: &str) -> (Vec<String>, Vec<String>) {
    let (signal_part, noise_part) = split_signal_noise(source_text);
    let signal = signal_part
        .lines()
        .map(str::trim)
        .filter_map(|line| NUMBERED_RE.captures(line))
        .map(|c| c[1].trim().to_string())
        .collect();
    (signal, parse_noise_entries(noise_part))
}

fn name_overlap(a: &str, b: &str) -> bool {
    let norm = |s: &str| -> HashSet<String> {
        alnum_words(s)
            .into_iter()
            .filter(|t| t.len() >= 3)
            .collect()
    };
    !norm(a).is_disjoint(&norm(b))
}

/// Score filename-extraction filtering: signal snippets renamed, noise excluded.
#[must_use]
pub fn validate_mixed_filename(data: &Value, source_text: &str) -> (i64, String) {
    if is_empty(data) {
        return (0, "empty response".to_string());
    }
    let outputs: Vec<String> = match data {
        Value::Object(_) => vec![data.to_string()],
        Value::Array(items) => items.iter().filter(|x| !is_empty(x)).map(as_text).collect(),
        other => vec![as_text(other)],
    };
    let (signal, noise) = extract_mixed_filenames(source_text);
    if signal.is_empty() {
        return (0, "no signal snippets in source".to_string());
    }
    let tp = signal
        .iter()
        .filter(|sig| outputs.iter().any(|out| name_overlap(sig, out)))
        .count();
    let recall = super::super::scoring_math::ratio(tp, signal.len());
    let fp = outputs
        .iter()
        .filter(|out| noise.iter().any(|n| name_overlap(n, out)))
        .count();
    let precision = if outputs.is_empty() {
        0.0
    } else {
        super::super::scoring_math::ratio(outputs.len() - fp, outputs.len())
    };
    let score = super::super::scoring_math::pct_floor_mean(recall, precision);
    let mut failures = Vec::new();
    if fp > 0 {
        failures.push(format!("included {fp}/{} noise-derived names", noise.len()));
    }
    if tp < signal.len() {
        failures.push(format!(
            "missed {}/{} signal snippets",
            signal.len() - tp,
            signal.len()
        ));
    }
    (score, failures.join("; "))
}

/// Multi-falsehood faithfulness probe: `100 - 100 * parroted / planted`.
/// A falsehood counts as parroted when it appears verbatim or when
/// [`FALSEHOOD_TOKEN_HITS`] of its identifying tokens do.
#[must_use]
pub fn validate_factual_accuracy(output: &str, falsehood_phrases: &[String]) -> (i64, String) {
    if output.is_empty() || falsehood_phrases.is_empty() {
        return (100, String::new());
    }
    let out = output.to_lowercase();
    let present: HashSet<String> = tokenize(&out).into_iter().collect();
    let found: Vec<&String> = falsehood_phrases
        .iter()
        .filter(|phrase| {
            let p = phrase.to_lowercase();
            let p = p.trim();
            if p.is_empty() {
                return false;
            }
            if out.contains(p) {
                return true;
            }
            let ptoks: HashSet<String> = identifying_tokens(p).into_iter().collect();
            !ptoks.is_empty()
                && ptoks.iter().filter(|t| present.contains(*t)).count() >= FALSEHOOD_TOKEN_HITS
        })
        .collect();
    if found.is_empty() {
        return (100, String::new());
    }
    let pct =
        (100 - super::super::scoring_math::pct_floor(found.len(), falsehood_phrases.len())).max(0);
    (
        pct,
        format!(
            "parrots {}/{} falsehoods: {:?}",
            found.len(),
            falsehood_phrases.len(),
            found[0]
        ),
    )
}

/// Fact-coverage scoring on identifying tokens, not substrings, because the
/// prompt orders the model to reword.
#[must_use]
pub fn validate_factual_coverage(output: &str, key_facts: &[String]) -> (i64, String) {
    if output.is_empty() {
        return (0, "empty response".to_string());
    }
    if key_facts.is_empty() {
        return (100, String::new());
    }
    let out = output.to_lowercase();
    let found = key_facts
        .iter()
        .filter(|fact| phrase_overlap(fact, &out) >= COVERAGE_TOKEN_RATIO)
        .count();
    let pct = super::super::scoring_math::pct_floor(found, key_facts.len());
    let failures = if pct < 30 {
        format!("covered {found}/{} key facts", key_facts.len())
    } else {
        String::new()
    };
    (pct, failures)
}

#[cfg(test)]
#[path = "mixed_text_tests.rs"]
mod tests;
