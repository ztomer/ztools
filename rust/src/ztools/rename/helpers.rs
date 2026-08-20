//! Pure text-cleaning helpers for the image renamer.
//!
//! Ported from `rename/helpers.py`. These are the reference semantics the Rust
//! side must agree with; the `bin/ab_test` fixtures pin both sides to the same
//! verdicts.

use regex::Regex;

/// Generic LLM outputs that are not real filenames. The bases alone, plus each
/// base combined with each image extension.
const GENERIC_BASES: &[&str] = &[
    "text", "file", "image", "unnamed", "output", "filename", "none", "screenshot", "document",
    "note", "empty", "blank",
];

const GENERIC_EXTENSIONS: &[&str] = &[
    "txt", "png", "jpg", "jpeg", "gif", "bmp", "tiff", "tif", "webp",
];

/// Whether a proposed name is generic filler that identifies nothing. Matches
/// `_GENERIC_NAMES` in `helpers.py`: the bases, plus every `base_ext` pair.
pub fn is_generic_name(name: &str) -> bool {
    let n = name.trim().to_lowercase();
    if GENERIC_BASES.contains(&n.as_str()) {
        return true;
    }
    for base in GENERIC_BASES {
        for ext in GENERIC_EXTENSIONS {
            if n == format!("{base}_{ext}") {
                return true;
            }
        }
    }
    false
}

/// Fold arbitrary text into a concise snake_case filename.
///
/// Exactly the Python steps: drop every char that is not a word char,
/// whitespace or hyphen; collapse hyphen/whitespace runs to a single `_`;
/// strip leading/trailing `_`; lowercase; truncate at `max_length` without
/// leaving a trailing `_`. Empty result is `unnamed`.
pub fn clean_filename(text: &str, max_length: usize) -> String {
    let mut kept = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '_' || ch.is_whitespace() || ch == '-' {
            kept.push(ch);
        }
    }

    let mut out = String::new();
    let mut last_was_under = true;
    for ch in kept.chars() {
        if ch == '-' || ch.is_whitespace() {
            if !last_was_under {
                out.push('_');
                last_was_under = true;
            }
        } else {
            out.extend(ch.to_lowercase());
            last_was_under = false;
        }
    }

    let trimmed = out.trim_matches('_');
    if trimmed.is_empty() {
        return "unnamed".to_string();
    }
    if trimmed.len() > max_length {
        let cut: String = trimmed.chars().take(max_length).collect();
        cut.trim_end_matches('_').to_string()
    } else {
        trimmed.to_string()
    }
}

/// Strip a conversational prefix like "here is the filename:" or "renamed to:"
/// from model output. Port of `_strip_instruction_prefix` (regex, IGNORECASE).
pub fn strip_instruction_prefix(content: &str) -> String {
    let re = Regex::new(
        r"(?i)^\s*(?:(?:here(?: is|'s)?(?: a| the)?|the|suggested|renamed? to)?\s*(?:filename|file|name|output|result|response)?(?:\s+is)?\s*:\s*)",
    )
    .expect("instruction-prefix regex is static");
    re.replace(content, "").trim().to_string()
}

/// Whether extracted text looks like real narrative worth naming from.
///
/// Port of `is_meaningful_text`: a word-count heuristic with a single-word
/// acronym guard, an all-caps filename guard, and a `min_word_count` floor on
/// words that are longer than 2 chars and contain a letter.
pub fn is_meaningful_text(text: &str, min_word_count: usize) -> bool {
    let text = text.trim();

    let words: Vec<&str> = text.split_whitespace().collect();

    if words.len() == 1 && text.len() > 8 {
        // A single long alphanumeric token whose first two chars are upper case
        // is a hash or an ID, not a description.
        if text.chars().all(|c| c.is_alphanumeric())
            && text.chars().take(2).all(|c| c.is_uppercase())
        {
            return false;
        }
    }

    // A long all-caps token with no spaces is a filename fragment, not prose.
    if text.len() > 4
        && !text.contains(' ')
        && text.chars().any(char::is_uppercase)
        && !text.chars().any(char::is_lowercase)
    {
        return false;
    }

    let word_like = words
        .iter()
        .filter(|w| w.len() > 2 && w.chars().any(|c| c.is_alphabetic()))
        .count();
    word_like >= min_word_count
}

/// Whether text is a hash, handle, code or other non-human token.
///
/// Port of `is_non_human_readable`. Empty and short strings count as
/// non-human-readable: there is nothing to name from.
pub fn is_non_human_readable(text: &str) -> bool {
    let text = text.trim();
    if text.is_empty() {
        return true;
    }
    if text.len() < 3 {
        return true;
    }

    // ^HF[A-Za-z0-9]{7,}$ / ^HH[A-Za-z0-9]{7,}$ -- huggingface model ids.
    if (text.starts_with("HF") || text.starts_with("HH"))
        && text.len() >= 9
        && text.chars().skip(2).all(|c| c.is_ascii_alphanumeric())
    {
        return true;
    }

    // @handle without underscores.
    if text.starts_with('@') && !text.contains('_') && text.len() > 1 {
        return true;
    }

    // Short all-caps codes.
    if text.len() <= 3 && text.chars().any(char::is_uppercase) {
        return true;
    }

    // Uppercase token containing digits and no spaces (e.g. "ABC123XYZ").
    if !text.contains(' ')
        && text.chars().any(char::is_uppercase)
        && text.chars().any(|c| c.is_ascii_digit())
    {
        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_names_are_flagged() {
        assert!(is_generic_name("image"));
        assert!(is_generic_name("unnamed"));
        assert!(is_generic_name("screenshot_png"));
        assert!(!is_generic_name("image.jpg"));
        assert!(!is_generic_name("white_goose_grass"));
    }

    #[test]
    fn clean_filename_matches_python_contract() {
        assert_eq!(clean_filename("Hello World! 2026", 50), "hello_world_2026");
        assert_eq!(clean_filename("   ", 50), "unnamed");
        assert_eq!(clean_filename("Special @#$% Symbols!", 30), "special_symbols");
        let long = "This is an extremely long title that exceeds the maximum length constraint for filenames";
        assert_eq!(clean_filename(long, 20), "this_is_an_extremely");
    }

    #[test]
    fn strip_instruction_prefix_matches_python_contract() {
        assert_eq!(strip_instruction_prefix("Here is the filename: tax_return_2026.pdf"),
                   "tax_return_2026.pdf");
        assert_eq!(strip_instruction_prefix("The file is: meeting_notes_v1.png"),
                   "meeting_notes_v1.png");
        assert_eq!(strip_instruction_prefix("suggested name: invoice"),
                   "invoice");
        assert_eq!(strip_instruction_prefix("renamed to: screenshot"),
                   "screenshot");
        // No prefix: unchanged apart from trimming.
        assert_eq!(strip_instruction_prefix("  plain content  "), "plain content");
    }

    #[test]
    fn meaningfulness_matches_python_contract() {
        assert!(is_meaningful_text("Receipt from Apple Store", 2));
        assert!(is_meaningful_text("Red Car", 2));
        assert!(!is_meaningful_text("2026-08-06 14:30:00", 2));
        assert!(!is_meaningful_text("a", 2));
        assert!(!is_meaningful_text("IMG 9999", 2));
        assert!(!is_meaningful_text("SCREENSHOT2026", 2));
    }

    #[test]
    fn non_human_readable_matches_python_contract() {
        assert!(is_non_human_readable("HFa8f9c1b3d9e4f2a7b0c8d1e3f5a9b7c1"));
        assert!(is_non_human_readable("@somename"));
        assert!(is_non_human_readable("ABC"));
        assert!(is_non_human_readable("ABC123XYZ"));
        assert!(is_non_human_readable(""));
        // A plain lowercase hex string is NOT flagged by this check alone --
        // the meaningfulness check catches it instead (port parity).
        assert!(!is_non_human_readable("a8f9c1b3d9e4f2a7b0c8d1e3f5a9b7c1"));
    }
}