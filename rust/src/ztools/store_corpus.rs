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
//!
//! The corpus is half the evidence. The other half is what the model made of
//! it, and on 2026-10-10 a corpus naming four in-window family events became a
//! plan with one, with nothing on disk to say which phase lost them. Each run's
//! phase transcript (`weekend/transcript.rs`) is kept here too, as
//! `<run>_phases.txt` under the corpus's own stamp, with the same bound.

use anyhow::Result;
use std::path::{Path, PathBuf};

/// Subdirectory of the plan store the corpora live in.
pub const CORPUS_DIR: &str = "corpus";

/// How many runs' corpora are kept. A scheduled planner runs a few times a
/// weekend; ten covers the last several weekends without growing forever.
pub const CORPUS_KEEP: usize = 10;

const SUFFIX: &str = "_corpus.txt";

/// The phase transcript's suffix: `<run>_phases.txt`, beside the corpus of the
/// same run (`weekend/transcript.rs`), so the two pair by name.
const PHASES_SUFFIX: &str = "_phases.txt";

/// Where `store`'s corpora live.
#[must_use]
pub fn corpus_dir(store: &Path) -> PathBuf {
    store.join(CORPUS_DIR)
}

/// `2026-10-10_143005_corpus.txt`: the run's local time, so names sort in run
/// order and retention can go by name.
#[must_use]
pub fn corpus_filename(run: chrono::NaiveDateTime) -> String {
    stamped(run, SUFFIX)
}

/// `2026-10-10_143005_phases.txt`: the same stamp as that run's corpus.
#[must_use]
pub fn phases_filename(run: chrono::NaiveDateTime) -> String {
    stamped(run, PHASES_SUFFIX)
}

fn stamped(run: chrono::NaiveDateTime, suffix: &str) -> String {
    format!("{}{suffix}", run.format("%Y-%m-%d_%H%M%S"))
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
    save_record(store, run, SUFFIX, &format!("{header}\n{corpus}\n"))
}

/// Write a run's phase transcript to `<store>/corpus/<run>_phases.txt`.
///
/// Then prune the transcripts to the newest [`CORPUS_KEEP`] -- the same bound
/// and the same directory as the corpora, counted separately so neither kind
/// can push the other out.
///
/// # Errors
///
/// As [`save_corpus`].
pub fn save_phases(store: &Path, run: chrono::NaiveDateTime, text: &str) -> Result<PathBuf> {
    save_record(store, run, PHASES_SUFFIX, text)
}

fn save_record(
    store: &Path,
    run: chrono::NaiveDateTime,
    suffix: &str,
    text: &str,
) -> Result<PathBuf> {
    let dir = corpus_dir(store);
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(stamped(run, suffix));
    std::fs::write(&path, text)?;
    prune(&dir, CORPUS_KEEP, suffix);
    Ok(path)
}

/// Delete all but the newest `keep` files ending in `suffix` in `dir`. Only
/// files this module names are candidates; anything else is left alone.
fn prune(dir: &Path, keep: usize, suffix: &str) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut names: Vec<String> = entries
        .filter_map(Result::ok)
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| n.ends_with(suffix))
        .collect();
    names.sort();
    let excess = names.len().saturating_sub(keep);
    for name in &names[..excess] {
        if let Err(e) = std::fs::remove_file(dir.join(name)) {
            eprintln!("\u{26a0} could not prune old run record {name}: {e}");
        }
    }
}

/// Keep the corpus this run judged beside the plans, so a thin plan can be
/// explained from what the engines actually returned. A failure to write it is
/// reported and never fails the run.
pub fn record_corpus(
    run: chrono::NaiveDateTime,
    corpus: &str,
    (d1, d2): (chrono::NaiveDate, chrono::NaiveDate),
    queries: &[String],
    (in_window, total): (usize, usize),
) {
    let mut header = format!("# window {d1}..{d2}\n# in-window {in_window}/{total}");
    for q in queries {
        header.push_str("\n# query ");
        header.push_str(q);
    }
    let store = crate::ztools::store::weekend_output_dir();
    match save_corpus(&store, run, &header, corpus) {
        Ok(path) => println!("\u{2192} Corpus kept at {}", path.display()),
        Err(e) => eprintln!("\u{26a0} corpus not kept under {}: {e}", store.display()),
    }
}

/// Keep what every model call of this run was asked and answered, under the
/// corpus's run stamp. A run that made no call keeps nothing; a failure to
/// write is reported and never fails the run.
pub fn record_phases(
    run: chrono::NaiveDateTime,
    log: &crate::ztools::weekend::transcript::PhaseLog,
    (d1, d2): (chrono::NaiveDate, chrono::NaiveDate),
    model: &str,
) {
    let calls = log.entries().len();
    if calls == 0 {
        return;
    }
    let text = format!(
        "# window {d1}..{d2}\n# model {model}\n# calls {calls}\n{}",
        log.render()
    );
    let store = crate::ztools::store::weekend_output_dir();
    match save_phases(&store, run, &text) {
        Ok(path) => println!("\u{2192} Model answers kept at {}", path.display()),
        Err(e) => eprintln!(
            "\u{26a0} model answers not kept under {}: {e}",
            store.display()
        ),
    }
}

#[cfg(test)]
#[path = "store_corpus_tests.rs"]
mod tests;
