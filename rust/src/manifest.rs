//! Path helpers for the ztools crate.
//!
//! Extracted from `routines/src/manifest.rs` when the port moved into its own
//! crate; the ztools modules only ever used `expand_tilde`, so that is all
//! that came over.

use std::path::PathBuf;

/// Expand a leading `~` so paths can be written portably in config.
pub fn expand_tilde(p: &str) -> PathBuf {
    match p.strip_prefix("~/") {
        Some(rest) => match dirs::home_dir() {
            Some(h) => h.join(rest),
            None => PathBuf::from(p),
        },
        None => PathBuf::from(p),
    }
}