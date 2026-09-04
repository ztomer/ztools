//! Matching known statements inside paraphrased text.
//!
//! Port of `lib/validators/text_match.py`. Pure text transformation and overlap calculation.

use regex::Regex;
use std::collections::HashSet;
use std::sync::LazyLock;

static TOKEN_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[^a-z0-9$%.+]+").unwrap());

static ACRONYM_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b[A-Z]{2,}\b").unwrap());

/// Split text into comparable tokens, keeping the marks that carry meaning.
///
/// '$', '%', '+' and an interior '.' stay attached, because '$75K', '40%',
/// '1000+' and '2.5' are each one thing; a trailing '.' is dropped.
pub fn tokenize(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let raw = TOKEN_RE.replace_all(&lower, " ");
    raw.split_whitespace()
        .map(|tok| tok.trim_matches('.').to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

/// The tokens that identify `phrase`, for matching it inside a paraphrase.
///
/// - any token of 4+ characters
/// - any token containing a digit
/// - any acronym recognized from the phrase's OWN capitalization
pub fn identifying_tokens(phrase: &str) -> Vec<String> {
    let acronyms: HashSet<String> = ACRONYM_RE
        .find_iter(phrase)
        .map(|m| m.as_str().to_lowercase())
        .collect();

    tokenize(phrase)
        .into_iter()
        .filter(|t| t.len() >= 4 || t.chars().any(|c| c.is_ascii_digit()) || acronyms.contains(t))
        .collect()
}

/// Fraction of `phrase`'s identifying tokens present in `out_lower`.
pub fn phrase_overlap(phrase: &str, out_lower: &str) -> f64 {
    let tokens = identifying_tokens(phrase);
    if tokens.is_empty() {
        let stripped = phrase.to_lowercase().trim().to_string();
        if !stripped.is_empty() && out_lower.contains(&stripped) {
            return 1.0;
        }
        return 0.0;
    }
    let present: HashSet<String> = tokenize(out_lower).into_iter().collect();
    let hits = tokens.iter().filter(|t| present.contains(*t)).count();
    hits as f64 / tokens.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_preserves_money_and_percentages() {
        let tokens = tokenize("Revenue rose 40% to $75K. Output was 2.5x");
        assert_eq!(
            tokens,
            vec!["revenue", "rose", "40%", "to", "$75k", "output", "was", "2.5x"]
        );
    }

    #[test]
    fn test_identifying_tokens_extracts_acronyms_and_digits() {
        let tokens = identifying_tokens("TD Bank announced GPT-5 and Llama 4 at 1000+ venues");
        assert!(tokens.contains(&"td".to_string()));
        assert!(tokens.contains(&"bank".to_string()));
        assert!(tokens.contains(&"gpt".to_string()));
        assert!(tokens.contains(&"5".to_string()));
        assert!(tokens.contains(&"llama".to_string()));
        assert!(tokens.contains(&"4".to_string()));
        assert!(tokens.contains(&"1000+".to_string()));
        assert!(tokens.contains(&"venues".to_string()));
        assert!(!tokens.contains(&"at".to_string()));
    }

    #[test]
    fn test_phrase_overlap_smooth_scoring() {
        let phrase = "Canadian GDP grows 0.5% in Q3";
        let output = "The Canadian GDP grew 0.5% in Q3 according to analysts.";
        let overlap = phrase_overlap(phrase, output);
        assert_eq!(overlap, 0.8); // 4 out of 5 tokens match (canadian, gdp, 0.5%, q3)
    }
}
