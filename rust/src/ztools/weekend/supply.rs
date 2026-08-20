//! Making the in-window candidates visible to the model, without starving it.
//!
//! Ported from `weekend/supply.py`. THE PROBLEM: event searches are
//! month-scoped and the aggregator pages list a whole month, so the draft
//! phase chooses events that mostly cannot happen on the planned weekend, and
//! the deterministic window filter downstream removes nearly all of them.
//!
//! THE FIX THAT DID NOT WORK: filtering candidates to the window BEFORE the
//! draft was tried and reverted. It did not make the model return fewer
//! events; it made the model INVENT them. A constraint a component cannot
//! satisfy honestly will be satisfied dishonestly.
//!
//! So this REMOVES NOTHING. It marks candidates that mention a date inside the
//! plan window and floats them to the top, leaving every other candidate
//! present and selectable below. Supply is unchanged, so the model can never be
//! starved into inventing; what changes is only what it sees first.

use chrono::{Datelike, NaiveDate};

use super::find_dates_in;

/// What a marked line is prefixed with. The model is told what it means in the
/// prompt; here it just has to be unmistakable and stable.
pub const IN_WINDOW_MARK: &str = "[THIS WEEKEND]";

/// Does this candidate mention any date inside the plan window?
///
/// Uses the same scanner the report checkers use, so a candidate this floats
/// cannot be one the checker would later call out-of-window on the same
/// evidence.
pub fn mentions_window(text: &str, start: NaiveDate, end: NaiveDate) -> bool {
    find_dates_in(text, start.year())
        .iter()
        .any(|d| *d >= start && *d <= end)
}

/// Float in-window candidates to the top of the corpus and mark them.
///
/// Order is preserved within each group, so this is a stable partition rather
/// than a re-ranking -- two runs over the same corpus produce the same text.
///
/// Returns the corpus unchanged when nothing matches: a corpus with no dated
/// candidates is a real situation (evergreen venue listings), and inventing a
/// marker for it would tell the model something untrue.
pub fn prioritise_in_window(corpus: &str, start: NaiveDate, end: NaiveDate) -> String {
    let mut marked = Vec::new();
    let mut rest = Vec::new();
    for line in corpus.lines() {
        if !line.trim().is_empty() && mentions_window(line, start, end) {
            marked.push(format!("{IN_WINDOW_MARK} {line}"));
        } else {
            rest.push(line.to_string());
        }
    }
    if marked.is_empty() {
        return corpus.to_string();
    }
    marked.extend(rest);
    marked.join("\n")
}

/// How many candidates actually land in the window.
///
/// Reported to the operator, because it is the number that explains a thin
/// plan: 20 candidates of which 0 are in-window is a SUPPLY problem, and it
/// looks identical to a model problem unless someone counts.
pub fn in_window_count(corpus: &str, start: NaiveDate, end: NaiveDate) -> usize {
    corpus
        .lines()
        .filter(|l| !l.trim().is_empty() && mentions_window(l, start, end))
        .count()
}