//! Path helpers for the ztools crate.
//!
//! Extracted from `routines/src/manifest.rs` when the port moved into its own
//! crate; the ztools modules only ever used `expand_tilde`, so that is all
//! that came over.

use std::path::PathBuf;

/// Expand a leading `~` so paths can be written portably in config.
#[must_use]
pub fn expand_tilde(p: &str) -> PathBuf {
    // Either half missing -- not a `~/` path, or no home directory -- leaves
    // the path exactly as written rather than half-expanded.
    p.strip_prefix("~/")
        .and_then(|rest| dirs::home_dir().map(|h| h.join(rest)))
        .unwrap_or_else(|| PathBuf::from(p))
}
