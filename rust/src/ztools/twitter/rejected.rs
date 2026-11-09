//! Answers the quality gate refused, kept where the operator can read them.
//!
//! A rejection reason says WHAT rule fired ("44 of 44 bullets lack the
//! citation"), never what the model wrote instead -- and without the answer
//! the next fix is a guess. So each rejected answer is written beside the
//! store, in `rejected/`, a directory the store's readers never descend into,
//! and the reason names the file. Only the newest [`KEEP`] are kept.
//!
//! The tweets the model was SHOWN are kept beside it, as `<same name>.json`, a
//! file `twitter-summarize --json` reads back. Without them a refusal could be
//! read but never re-run: on 2026-11-09 sixteen kept answers had no input to
//! measure a fix against.

use std::fs;
use std::path::{Path, PathBuf};

/// How many rejected answers are kept; older ones are deleted.
pub const KEEP: usize = 20;

/// Write `answer` from `model` under `store/rejected/`, prune to [`KEEP`],
/// and return the path. `None` when it cannot be written: keeping the
/// evidence must never turn a rejection into a different failure.
pub fn keep(
    store: &Path,
    stamp: &str,
    model: &str,
    answer: &str,
    why: &str,
    shown: &[super::Tweet],
) -> Option<PathBuf> {
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
    if let Ok(json) = serde_json::to_string_pretty(shown) {
        let _ = fs::write(path.with_extension("json"), json);
    }
    for ext in ["md", "json"] {
        prune(&dir, ext);
    }
    Some(path)
}

/// Delete all but the newest [`KEEP`] files ending `.ext` in `dir`.
fn prune(dir: &Path, ext: &str) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut kept: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == ext))
        .collect();
    // Stamps lead the name, so lexical order is age order.
    kept.sort();
    for old in kept.iter().rev().skip(KEEP) {
        let _ = fs::remove_file(old);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rejected_answer_is_kept_with_its_reason_and_the_oldest_pruned() {
        let store = tempfile::tempdir().unwrap();
        for i in 0..KEEP + 3 {
            keep(
                store.path(),
                &format!("2026-10-10_{i:04}"),
                "m",
                "x",
                "r",
                &[],
            )
            .unwrap();
        }
        let path = keep(
            store.path(),
            "2026-10-11_0000",
            "raptor/v0:5",
            "the answer",
            "no cites",
            &[],
        )
        .unwrap();
        assert_eq!(path.file_name().unwrap(), "2026-10-11_0000_raptor_v0_5.md");
        let body = fs::read_to_string(&path).unwrap();
        assert!(body.starts_with("<!-- rejected: no cites -->"), "{body}");
        assert!(body.ends_with("the answer"), "{body}");
        let left = fs::read_dir(store.path().join("rejected")).unwrap().count();
        assert_eq!(left, 2 * KEEP, "the newest {KEEP} answers and their inputs");
        assert!(!store.path().join("rejected/2026-10-10_0000_m.md").exists());
    }

    /// The input is kept beside the answer, in the form `--json` reads back.
    #[test]
    fn the_tweets_the_model_was_shown_are_kept_as_a_rerunnable_fixture() {
        let store = tempfile::tempdir().unwrap();
        let shown = vec![super::super::Tweet {
            id: "7".to_string(),
            screen_name: "alice".to_string(),
            text: "The bridge reopened".to_string(),
            created_at: "Mon Nov 09 01:00:00 +0000 2026".to_string(),
            favorite_count: 3,
            retweet_count: 0,
            reply_to: None,
        }];
        let path = keep(store.path(), "2026-11-09_0100", "m", "a", "r", &shown).unwrap();
        let input = fs::read_to_string(path.with_extension("json")).expect("the input is kept");
        let back: Vec<super::super::Tweet> = serde_json::from_str(&input).unwrap();
        assert_eq!(back, shown);
    }
}
