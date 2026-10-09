//! The file-summary prompt's content block: the listed files' own text, bounded.
//!
//! Exists because the prompt asks for a description that follows only from the
//! file's CONTENT and, until this, sent seventeen paths and no content at all.
//! The bound is stated in the parent ([`FILE_SUMMARY_EXCERPT_LINES`] /
//! [`FILE_SUMMARY_EXCERPT_BYTES`]) and pinned by
//! `the_rendered_prompt_stays_inside_its_stated_budget`.
//!
//! EVERY LINE IS PREFIXED. The block is spliced into the mixed prompt ABOVE its
//! noise marker, where `validate_mixed_file_summary` reads signal paths out of
//! each row-shaped line (see `validators::mixed_text::extract_file_paths`: a
//! trimmed line with no interior whitespace, not starting `-`, `|` or `#`, and
//! carrying a `.` or a `/`). A raw excerpt line beginning `/` — a comment, a
//! TOML value, a path in prose — would be counted as a signal path the model was
//! never asked about and quietly depress the mixed variant's recall for every
//! model. The `| ` prefix makes that unreachable by construction rather than by
//! careful editing.
//!
//! THE ROOT IS A PARAMETER, AND ONLY THE LIVE WRAPPERS PICK ONE. The rows are
//! repo-relative (see the parent's header), so reading them means resolving a
//! checkout first. That resolution is [`live_root`], and it is the ONE impure
//! thing in this module: everything else is a pure function of `(root,
//! template)`, which is what lets a fixture checkout drive a render and pin that
//! the prompt carries the FIXTURE's bytes rather than whatever repository
//! happens to hold the binary. The live wrappers are thin on purpose —
//! [`excerpts`], [`render`] and [`render_both`] keep the signatures the roster
//! and the pins already call and add nothing but `live_root()`.

use anyhow::{Context, Result};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use super::{
    FILE_SUMMARY_CONTENTS_SLOT, FILE_SUMMARY_EXCERPT_BYTES, FILE_SUMMARY_EXCERPT_LINES,
    FILE_SUMMARY_FILE_LIST,
};
use crate::manifest;

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
/// When a listed row cannot be read under `root`. The alternative — skipping the
/// row and sending the path anyway — is precisely the defect this block exists
/// to fix, so a missing file is a hard error naming the row rather than a prompt
/// that cannot be followed. The row is named RELATIVE and the root is named with
/// it, because the row is the constant and the root is what differed between the
/// two machines that reported this.
pub fn excerpts_from(root: &Path) -> Result<String> {
    let mut out = String::from(
        "FILE CONTENTS (EXCERPTS). These are the HEADS of the files named above, not the whole \
         files. Describe each file from the excerpt in front of you and from nothing \
         else -- never from its name.\n\n",
    );
    for row in rows() {
        let text = std::fs::read_to_string(root.join(row))
            .with_context(|| format!("listed file {row} in {}", root.display()))?;
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

/// `template` with its [`FILE_SUMMARY_CONTENTS_SLOT`] replaced by the block
/// [`excerpts_from`] reads out of `root`.
///
/// # Errors
///
/// As [`excerpts_from`]; also when the template carries no slot, which would ship
/// a prompt that lists files and still explains none of them.
pub fn render_from(root: &Path, template: &str) -> Result<String> {
    Ok(substitute(template, &excerpts_from(root)?))
}

/// `template` with its [`FILE_SUMMARY_CONTENTS_SLOT`] replaced by `block`.
///
/// Both file-summary prompts carry the same slot and the same file list, so
/// `render_both_from` builds the block ONCE and calls this twice. Reading
/// seventeen files once per prompt is thirty-four reads for one byte-identical
/// block, and `roster()` is on the path of every eval run.
#[must_use]
pub fn substitute(template: &str, block: &str) -> String {
    template.replace(FILE_SUMMARY_CONTENTS_SLOT, block)
}

/// Both templates rendered from ONE content block read out of `root`.
///
/// # Errors
///
/// As [`render_from`]: a listed file that cannot be read, or a template that lost
/// its slot. Both are hard errors on purpose — see the module header.
pub fn render_both_from(root: &Path, plain: &str, mixed: &str) -> Result<(String, String)> {
    for (name, template) in [("plain", plain), ("mixed", mixed)] {
        if !template.contains(FILE_SUMMARY_CONTENTS_SLOT) {
            anyhow::bail!(
                "the {name} file-summary prompt lost its {FILE_SUMMARY_CONTENTS_SLOT} slot"
            );
        }
    }
    let block = excerpts_from(root)?;
    Ok((substitute(plain, &block), substitute(mixed, &block)))
}

/// The checkout the LIVE prompts are rendered from.
///
/// THE DERIVATION, and why it is not `.`. The rows are repo-relative, so a
/// renderer has to pick a checkout; picking the working directory would be a guess
/// that happens to be right for `cargo test` in a repo and wrong for an installed
/// binary and for every CI runner, and a render that silently reads the WRONG
/// checkout grades a model against files it was never shown. So this asks
/// [`manifest::checkout_roots_from`] — the same derivation every other shipped
/// default in the crate uses, which starts from the running executable and falls
/// back to the home checkout — and takes the FIRST candidate that actually holds
/// every listed row.
///
/// THE LAST CANDIDATE IS THE CHECKOUT THIS BINARY WAS BUILT FROM, and that is not
/// a fifth fallback invented here; it is the third thing that is true about where
/// a binary's own source lives. Two of this crate's documented build layouts
/// break the other two candidates on their own: the shared build dir under
/// `CARGO_HOME` (`<cargo home>/build/<hash>/`) sits OUTSIDE any checkout, so
/// nothing is above the executable, and a machine with no home checkout has no
/// home fallback either. `CARGO_MANIFEST_DIR` is a compile-time constant, so it is also the one
/// candidate no environment redirect can move — which matters far more than it
/// sounds, because `ZTOOLS_EXE` and `$HOME` are both sandboxed per-test and cargo
/// runs tests in parallel: a roster built while a peer's `TestEnv` held the lock
/// would otherwise resolve to a directory that does not exist. Measured 2026-10-08
/// — without it, seventeen tests in this crate's suite (roster, pins, validator
/// and task-loader) failed whenever a peer's sandbox happened to be live, and all
/// pass with it.
///
/// EVERY ROW, NOT ONE. A root that holds `README.md` but not
/// `conf/models/qwen.toml` is a different checkout, an unrelated `conf/`, or a
/// partial worktree; accepting it would render a prompt naming eighteen files and
/// explaining seventeen, which is exactly the silent hole the hard error in
/// [`excerpts_from`] exists to refuse.
///
/// # Errors
///
/// When no candidate holds every row, naming every candidate tried. `ZTOOLS_EXE`
/// is the seam that decides this in a test, exactly as it does for every other
/// checkout-derived default in the crate; [`live_root_from`] is the twin that
/// takes the decision as arguments instead of reading the environment.
///
/// # Returns
///
/// The checkout root to resolve the rows against.
pub fn live_root() -> Result<PathBuf> {
    live_root_from(
        manifest::running_exe().as_deref(),
        dirs::home_dir().as_deref(),
        built_from().as_deref(),
    )
}

/// [`live_root`] against explicit inputs: the executable to walk, the home to
/// fall back to, and the checkout this binary was built from.
///
/// The twin exists for the reason every other one in this crate does: a rule
/// about WHICH checkout wins needs a fixture it controls on both sides. Written
/// against the live default it would pin the BUILD LAYOUT — the test binary sits
/// in `~/.cargo/build/<hash>/`, so the derived candidate is absent and the answer
/// is decided by `$HOME`, which on this machine happens to hold a checkout and on
/// a CI runner does not. `None` for any argument means "no such candidate", not
/// "use the live one"; [`live_root`] is the live one-liner.
///
/// # Errors
///
/// When no candidate holds every row, naming every candidate tried.
pub fn live_root_from(
    exe: Option<&Path>,
    home: Option<&Path>,
    built_from: Option<&Path>,
) -> Result<PathBuf> {
    let rows: Vec<&str> = rows().collect();
    let mut tried = manifest::checkout_roots_from(exe, home);
    if let Some(built) = built_from {
        // Appended, never prepended: the install-from derivation and the home
        // fallback are the operator's answer, and the build checkout is this
        // binary's own. `checkout_roots_from` de-duplicated its two candidates;
        // this one is compared rather than assumed distinct, because for a
        // binary built and run in place they are the same directory.
        if !tried.iter().any(|root| root == built) {
            tried.push(built.to_path_buf());
        }
    }
    let candidates = if tried.is_empty() {
        // No executable to walk, no home directory to fall back to and no build
        // checkout: nothing, said out loud rather than as an empty list, which
        // reads like a formatting bug.
        String::from("<no candidate: no executable, no home directory, no build checkout>")
    } else {
        tried
            .iter()
            .map(|root| root.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };
    tried
        .into_iter()
        .find(|root| rows.iter().all(|row| root.join(row).is_file()))
        .with_context(|| format!("no checkout of the file-summary rows; tried {candidates}"))
}

/// `<repo>`, from `CARGO_MANIFEST_DIR` (which is `<repo>/rust`) — the checkout
/// this binary was compiled in.
fn built_from() -> Option<PathBuf> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
}

/// [`excerpts_from`] against [`live_root`].
///
/// # Errors
///
/// As [`excerpts_from`], or when no checkout holds every row (see [`live_root`]).
pub fn excerpts() -> Result<String> {
    excerpts_from(&live_root()?)
}

/// [`render_from`] against [`live_root`].
///
/// # Errors
///
/// As [`render_from`], or when no checkout holds every row (see [`live_root`]).
pub fn render(template: &str) -> Result<String> {
    render_from(&live_root()?, template)
}

/// [`render_both_from`] against [`live_root`].
///
/// # Errors
///
/// As [`render_both_from`], or when no checkout holds every row (see
/// [`live_root`]).
pub fn render_both(plain: &str, mixed: &str) -> Result<(String, String)> {
    render_both_from(&live_root()?, plain, mixed)
}
