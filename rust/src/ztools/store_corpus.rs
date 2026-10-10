//! The corpus each weekend run judged, kept beside the plans it produced.
//!
//! A plan says "Candidates: 0/88 mention a date this weekend" and nothing
//! else about the 88: on 2026-10-10 that line was all there was, and the only
//! way to learn the 88 were `YouTube Kids`, a Mexican condo and Grand Theft Auto
//! was to rebuild the corpus by hand from the live engines -- which by then
//! answered differently. The run now writes the corpus it judged -- every
//! cleaned search result and followed listing line, in order, under a header
//! naming the window, the count and the queries -- so the next "why is this
//! plan thin?" starts from the evidence.
//!
//! Lives in `corpus/` UNDER the plan store (`store::weekend_output_dir`, so
//! `WEEKEND_OUTPUT_DIR` redirects it with everything else and tests stay
//! sandboxed), as `.txt`: the plan readers list the store directory itself and
//! take only `*.md`, so neither the dashboard's `--fetch-latest` nor the
//! status page can mistake a corpus for a plan (`store_corpus_tests` pins
//! both). Bounded: the newest [`CORPUS_KEEP`] runs are kept.

use anyhow::Result;
use std::path::{Path, PathBuf};

/// Subdirectory of the plan store the corpora live in.
pub const CORPUS_DIR: &str = "corpus";

/// How many runs' corpora are kept. A scheduled planner runs a few times a
/// weekend; ten covers the last several weekends without growing forever.
pub const CORPUS_KEEP: usize = 10;

const SUFFIX: &str = "_corpus.txt";

/// Where `store`'s corpora live.
#[must_use]
pub fn corpus_dir(store: &Path) -> PathBuf {
    store.join(CORPUS_DIR)
}

/// `2026-10-10_143005_corpus.txt`: the run's local time, so names sort in run
/// order and retention can go by name.
#[must_use]
pub fn corpus_filename(run: chrono::NaiveDateTime) -> String {
    format!("{}{SUFFIX}", run.format("%Y-%m-%d_%H%M%S"))
}

/// Write `header` then `corpus` to `<store>/corpus/<run>_corpus.txt`, creating
/// the directory, then prune to the newest [`CORPUS_KEEP`]. Returns the path.
///
/// # Errors
///
/// When the directory cannot be created or the file cannot be written. A
/// failed PRUNE is not an error: the record was written, and an old file
/// left behind is a disk-space matter, not a lost run.
pub fn save_corpus(
    store: &Path,
    run: chrono::NaiveDateTime,
    header: &str,
    corpus: &str,
) -> Result<PathBuf> {
    let dir = corpus_dir(store);
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(corpus_filename(run));
    std::fs::write(&path, format!("{header}\n{corpus}\n"))?;
    prune(&dir, CORPUS_KEEP);
    Ok(path)
}

/// Delete all but the newest `keep` corpus files in `dir`. Only files this
/// module names (`*_corpus.txt`) are candidates; anything else is left alone.
fn prune(dir: &Path, keep: usize) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut names: Vec<String> = entries
        .filter_map(Result::ok)
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| n.ends_with(SUFFIX))
        .collect();
    names.sort();
    let excess = names.len().saturating_sub(keep);
    for name in &names[..excess] {
        if let Err(e) = std::fs::remove_file(dir.join(name)) {
            eprintln!("\u{26a0} could not prune old corpus {name}: {e}");
        }
    }
}

#[cfg(test)]
#[path = "store_corpus_tests.rs"]
mod tests;
