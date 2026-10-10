//! Answers the quality gate refused, kept where the operator can read them.
//!
//! A rejection reason says WHAT rule fired ("44 of 44 bullets lack the
//! citation"), never what the model wrote instead -- and without the answer
//! the next fix is a guess. So each rejected answer is written beside the
//! store, in `rejected/`, a directory the store's readers never descend into,
//! and the reason names the file. Only the newest [`KEEP`] are kept.

use std::fs;
use std::path::{Path, PathBuf};

/// How many rejected answers are kept; older ones are deleted.
pub const KEEP: usize = 20;

/// Write `answer` from `model` under `store/rejected/`, prune to [`KEEP`],
/// and return the path. `None` when it cannot be written: keeping the
/// evidence must never turn a rejection into a different failure.
pub fn keep(store: &Path, stamp: &str, model: &str, answer: &str, why: &str) -> Option<PathBuf> {
    let dir = store.join("rejected");
    fs::create_dir_all(&dir).ok()?;
    let safe: String = model
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let path = dir.join(format!("{stamp}_{safe}.md"));
    fs::write(&path, format!("<!-- rejected: {why} -->\n\n{answer}")).ok()?;
    let mut kept: Vec<PathBuf> = fs::read_dir(&dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "md"))
        .collect();
    // Stamps lead the name, so lexical order is age order.
    kept.sort();
    for old in kept.iter().rev().skip(KEEP) {
        let _ = fs::remove_file(old);
    }
    Some(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rejected_answer_is_kept_with_its_reason_and_the_oldest_pruned() {
        let store = tempfile::tempdir().unwrap();
        for i in 0..KEEP + 3 {
            keep(store.path(), &format!("2026-10-10_{i:04}"), "m", "x", "r").unwrap();
        }
        let path = keep(
            store.path(),
            "2026-10-11_0000",
            "raptor/v0:5",
            "the answer",
            "no cites",
        )
        .unwrap();
        assert_eq!(path.file_name().unwrap(), "2026-10-11_0000_raptor_v0_5.md");
        let body = fs::read_to_string(&path).unwrap();
        assert!(body.starts_with("<!-- rejected: no cites -->"), "{body}");
        assert!(body.ends_with("the answer"), "{body}");
        let left = fs::read_dir(store.path().join("rejected")).unwrap().count();
        assert_eq!(left, KEEP);
        assert!(!store.path().join("rejected/2026-10-10_0000_m.md").exists());
    }
}
