//! The file-summary prompt's content block: the listed files' own text, bounded.
//!
//! Exists because the prompt asks for a description that follows only from the
//! file's CONTENT and, until this, sent seventeen paths and no content at all.
//! The bound is stated in the parent ([`FILE_SUMMARY_EXCERPT_LINES`] /
//! [`FILE_SUMMARY_EXCERPT_BYTES`]) and pinned by
//! `the_rendered_prompt_stays_inside_its_stated_budget`.
//!
//! EVERY LINE IS PREFIXED. The block is spliced into the mixed prompt ABOVE its
//! noise marker, where `validate_mixed_file_summary` reads signal paths by
//! "line starts with `/`". A raw excerpt line beginning `/` — a comment, a
//! TOML value, a path in prose — would be counted as a signal path the model was
//! never asked about and quietly depress the mixed variant's recall for every
//! model. The `| ` prefix makes that unreachable by construction rather than by
//! careful editing.

use anyhow::{Context, Result};
use std::fmt::Write as _;

use super::{
    FILE_SUMMARY_CONTENTS_SLOT, FILE_SUMMARY_EXCERPT_BYTES, FILE_SUMMARY_EXCERPT_LINES,
    FILE_SUMMARY_FILE_LIST,
};

/// The listed paths, one per row, whitespace-trimmed.
fn rows() -> impl Iterator<Item = &'static str> {
    FILE_SUMMARY_FILE_LIST
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
}

/// The first [`FILE_SUMMARY_EXCERPT_LINES`] lines of `text`, and no more than
/// [`FILE_SUMMARY_EXCERPT_BYTES`] bytes of it.
///
/// Bytes are counted as UTF-8 char boundaries are crossed, so a file whose
/// first 1200 bytes end mid-codepoint drops that character rather than
/// producing invalid text.
#[must_use]
pub fn excerpt(text: &str) -> (String, usize, usize) {
    let total_lines = text.lines().count();
    let mut out = String::new();
    let mut bytes = 0usize;
    let mut shown = 0usize;
    for line in text.lines().take(FILE_SUMMARY_EXCERPT_LINES) {
        let cost = line.len() + 1;
        if !out.is_empty() && bytes + cost > FILE_SUMMARY_EXCERPT_BYTES {
            break;
        }
        out.push_str(line);
        out.push('\n');
        bytes += cost;
        shown += 1;
    }
    (out, shown, total_lines)
}

/// The content block: every listed file, fenced, with the truncation said out
/// loud.
///
/// # Errors
///
/// When a listed row cannot be read. The alternative — skipping the row and
/// sending the path anyway — is precisely the defect this block exists to fix,
/// so a missing file is a hard error naming the row rather than a prompt that
/// cannot be followed.
pub fn excerpts() -> Result<String> {
    let mut out = String::from(
        "FILE CONTENTS (EXCERPTS). These are the HEADS of the files named above, not the whole \
         files. Describe each file from the excerpt in front of you and from nothing \
         else -- never from its name.\n\n",
    );
    for row in rows() {
        let text = std::fs::read_to_string(row).with_context(|| format!("listed file {row}"))?;
        let (body, shown, total) = excerpt(&text);
        let note = if shown < total {
            format!(" (showing {shown} of {total} lines)")
        } else {
            String::new()
        };
        writeln!(out, "--- BEGIN {row}{note} ---").expect("writing to a String cannot fail");
        for line in body.lines() {
            out.push_str("| ");
            out.push_str(line);
            out.push('\n');
        }
        writeln!(out, "--- END {row} ---").expect("writing to a String cannot fail");
        out.push('\n');
    }
    Ok(out)
}

/// `template` with its [`FILE_SUMMARY_CONTENTS_SLOT`] replaced by [`excerpts`].
///
/// # Errors
///
/// As [`excerpts`]; also when the template carries no slot, which would ship a
/// prompt that lists files and still explains none of them.
pub fn render(template: &str) -> Result<String> {
    Ok(substitute(template, &excerpts()?))
}

/// `template` with its [`FILE_SUMMARY_CONTENTS_SLOT`] replaced by `block`.
///
/// Both file-summary prompts carry the same slot and the same file list, so
/// [`render_both`] builds the block ONCE and calls this twice. Reading seventeen
/// files once per prompt is thirty-four reads for one byte-identical block, and
/// `roster()` is on the path of every eval run.
#[must_use]
pub fn substitute(template: &str, block: &str) -> String {
    template.replace(FILE_SUMMARY_CONTENTS_SLOT, block)
}

/// Both templates rendered from ONE content block.
///
/// # Errors
///
/// As [`render`]: a listed file that cannot be read, or a template that lost its
/// slot. Both are hard errors on purpose — see the module header.
pub fn render_both(plain: &str, mixed: &str) -> Result<(String, String)> {
    for (name, template) in [("plain", plain), ("mixed", mixed)] {
        if !template.contains(FILE_SUMMARY_CONTENTS_SLOT) {
            anyhow::bail!(
                "the {name} file-summary prompt lost its {FILE_SUMMARY_CONTENTS_SLOT} slot"
            );
        }
    }
    let block = excerpts()?;
    Ok((substitute(plain, &block), substitute(mixed, &block)))
}
