//! Reading prompt instructions and contract bounds from the prompt itself.
//!
//! Port of `lib/validators/prompt_contract.py`.

use regex::Regex;
use std::sync::LazyLock;

static REQUESTED_COUNT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:find|list|return|output|provide)\s+(\d+)(?:\s*(?:-|to)\s*(\d+))?").unwrap()
});

pub fn requested_item_count(source_text: &str) -> Option<usize> {
    let caps = REQUESTED_COUNT_RE.captures(source_text)?;
    caps.get(1).and_then(|m| m.as_str().parse().ok())
}

pub fn parse_signal_noise(source_text: &str) -> (Vec<String>, Vec<String>) {
    if !source_text.contains("NOISE") {
        return (Vec::new(), Vec::new());
    }
    let parts: Vec<&str> = source_text.splitn(2, "NOISE").collect();
    (extract_bullets(parts[0]), extract_bullets(parts[1]))
}

fn extract_bullets(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(stripped) = trimmed.strip_prefix("- ") {
            let content = stripped.trim();
            let name = content.split(':').next().unwrap_or("").trim();
            if !name.is_empty() {
                out.push(name.to_string());
            }
        }
    }
    out
}
