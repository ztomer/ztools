//! Grounding: was this description written from the file, or from its name?
//!
//! The file-summary task instructs the model to describe what each file DOES and
//! never to infer functionality from the name, and until 2026-10-05 the scorer
//! could not tell whether it had obeyed: it counted sentences containing a
//! content verb, which a name-based guess produces as readily as a grounded one.
//! This module is the instrument that can.
//!
//! THE TRUTH IS PINNED, NOT RE-READ. [`FILE_SUMMARY_GROUND_TRUTH`] carries two
//! phrases per listed file, and `every_pinned_truth_still_appears_in_its_file`
//! asserts each one still occurs in the file it describes. Re-reading the working
//! tree here instead would make a leaderboard score depend on the tree at
//! SCORING time, so two runs of one commit could disagree.
//!
//! A PATH WITH NO PINNED TRUTH IS NOT GROUNDED, it is merely unjudgeable, and the
//! caller falls back to the verb heuristic. That is what keeps every existing
//! caller of [`super::validate_file_summary`] — the smoke fixtures, the JSON
//! snapshot loader, the ordering controls, all of which name other repos' files —
//! scoring exactly as they did.

use crate::ztools::eval::prompts::file_summary::FILE_SUMMARY_GROUND_TRUTH;
use crate::ztools::eval::validators::text_match::phrase_overlap;

/// Fraction of a phrase's identifying tokens that must appear for it to count
/// as covered.
///
/// The same bar `validate_factual_coverage` uses, for the same reason: half the
/// tokens of a phrase chosen to be unmistakable is a real hit, and demanding all
/// of them scores a paraphrase as a miss.
pub const GROUNDED_RATIO: f64 = 0.5;

/// The repo-relative rows that have pinned truth, in prompt order.
#[must_use]
pub fn pinned_rows() -> Vec<&'static str> {
    FILE_SUMMARY_GROUND_TRUTH
        .iter()
        .map(|(rel, _)| *rel)
        .collect()
}

/// Lowercase, colon-stripped, `./`-stripped. Models echo the listed path, a
/// repo-relative one, or a bare basename, and `## path: summary` headers carry a
/// trailing colon.
fn normalise(path: &str) -> String {
    path.trim()
        .trim_end_matches(':')
        .trim()
        .trim_start_matches("./")
        .to_lowercase()
}

fn matches_row(rel: &str, answer: &str) -> bool {
    if answer == rel {
        return true;
    }
    if answer.ends_with(&format!("/{rel}")) {
        return true;
    }
    // A bare basename resolves only because every pinned basename is unique;
    // `every_basename_is_unique` pins that, so a second `mod.rs` breaks loudly
    // here instead of silently resolving to whichever row came first.
    std::path::Path::new(rel)
        .file_name()
        .is_some_and(|base| base == answer)
}

/// The pinned truth for `path`, by suffix or basename. `None` when the path is
/// not one of the listed rows.
#[must_use]
pub fn facts_for(path: &str) -> Option<&'static [&'static str]> {
    let answer = normalise(path);
    if answer.is_empty() {
        return None;
    }
    FILE_SUMMARY_GROUND_TRUTH
        .iter()
        .find(|(rel, _)| matches_row(&rel.to_lowercase(), &answer))
        .map(|(_, facts)| *facts)
}

/// Does `desc_lower` read as having been written from the file's content?
///
/// Half the row's phrases must be covered: two facts means one, so a single
/// unmistakable hit proves the model saw something only the file says, while a
/// row that lists more truths demands proportionally more of them.
#[must_use]
pub fn grounded(facts: &[&str], desc_lower: &str) -> bool {
    if facts.is_empty() {
        return false;
    }
    let covered = facts
        .iter()
        .filter(|fact| phrase_overlap(fact, desc_lower) >= GROUNDED_RATIO)
        .count();
    covered * 2 >= facts.len()
}

#[cfg(test)]
#[path = "validate_grounding_tests.rs"]
mod tests;
