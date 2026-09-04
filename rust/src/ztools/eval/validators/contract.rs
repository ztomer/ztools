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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_requested_item_count_matches_all_verbs_and_ranges() {
        assert_eq!(
            requested_item_count("please find 5 great activities"),
            Some(5)
        );
        assert_eq!(requested_item_count("list 3 to 7 nearby events"), Some(3));
        assert_eq!(requested_item_count("RETURN 12 results now"), Some(12));
        assert_eq!(requested_item_count("output 4 ideas"), Some(4));
        assert_eq!(requested_item_count("provide 9-11 options"), Some(9));
    }

    #[test]
    fn test_requested_item_count_rejects_missing_counts() {
        assert_eq!(requested_item_count("no counts requested here"), None);
        assert_eq!(requested_item_count("find many activities"), None);
        assert_eq!(requested_item_count(""), None);
    }

    #[test]
    fn test_parse_signal_noise_without_marker_is_empty() {
        let (signal, noise) = parse_signal_noise("just a prompt with no sections");
        assert!(signal.is_empty());
        assert!(noise.is_empty());
    }

    #[test]
    fn test_parse_signal_noise_splits_on_marker_and_cleans_bullets() {
        let src = "Find these:\n\
                   - Alpha: because of reasons\n\
                   - Beta\n\
                   ignore this prose line\n\
                   NOISE\n\
                   - Gamma: known wrong item\n\
                   - \n\
                   -\n\
                   trailing text";
        let (signal, noise) = parse_signal_noise(src);
        assert_eq!(signal, vec!["Alpha", "Beta"]);
        assert_eq!(noise, vec!["Gamma"]);
    }
}
