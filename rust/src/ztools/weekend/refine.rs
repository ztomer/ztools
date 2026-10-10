//! Phase 3, refine: the model merges and prunes the draft, and must SAY what
//! it removed.
//!
//! WHY IT IS ACCOUNTABLE. Its prompt used to read "merge any near-duplicates,
//! keep the best 8, remove low-quality or irrelevant ones, and sort by overall
//! appeal", and its answer REPLACED the draft. That is lossy by construction:
//! a cap of eight on a sixteen-event long weekend, a sort the scorer redoes
//! anyway, and a removal nobody recorded. On 2026-10-10 the Sugar Beach Fall
//! Harvest Market -- in the window, sourced, free, for families -- reached
//! refine and was not in its answer, and nothing said why; on the replay of the
//! same corpus refine left out a row silently again, and merged none of the
//! duplicates it was asked to merge (`dedup.rs` now does that).
//!
//! So the judgement stays with the model -- whether a row is a specific event
//! at all is semantic -- but the plumbing makes it accountable: the prompt has
//! no cap and no ranking, every removal must be a `DROPPED | NAME | reason`
//! line, and [`merge_refined`] lays the answer back over the draft. A draft row
//! the answer neither keeps nor drops with a reason is RESTORED and named on
//! the operator's channel. The plan can lose an event at refine only with a
//! recorded reason.

use std::collections::HashSet;

use super::PhaseLog;
use super::enforce::significant_tokens;
use super::phases::{call_llm_text, extracted_rows};
use super::prompts::{PHASE_REFINE, render};

/// The marker a refine answer opens a removal line with.
pub const DROPPED: &str = "DROPPED";

fn fields(row: &str) -> Vec<&str> {
    row.trim()
        .trim_start_matches(['-', '*', ' '])
        .trim_matches('|')
        .split('|')
        .map(str::trim)
        .collect()
}

fn name_of(row: &str) -> String {
    fields(row)
        .first()
        .map_or_else(String::new, |n| (*n).to_string())
}

/// Do two names name the same entry? The same significant words, or the
/// shorter's (of at least two words) all inside the longer's -- a refine that
/// writes "Fall Harvest Market at Sugar Beach" for "Fall Harvest Market" has
/// kept it.
fn same_entry(a: &str, b: &str) -> bool {
    let (x, y): (HashSet<String>, HashSet<String>) = (significant_tokens(a), significant_tokens(b));
    if x.is_empty() || y.is_empty() {
        return false;
    }
    let (short, long) = if x.len() <= y.len() {
        (&x, &y)
    } else {
        (&y, &x)
    };
    short == long || (short.len() >= 2 && short.is_subset(long))
}

/// A refine answer's removals: `(name, reason)` per `DROPPED | NAME | reason`.
fn removals(answer: &str) -> Vec<(String, String)> {
    answer
        .lines()
        .filter_map(|line| {
            let f = fields(line);
            (f.len() >= 2 && f[0].eq_ignore_ascii_case(DROPPED))
                .then(|| (f[1].to_string(), f[2..].join(" | ").trim().to_string()))
        })
        .collect()
}

/// Lay a refine `answer` over the `draft` it was given.
///
/// Returns the rows to structure -- the answer's own rows, then every draft
/// row the answer did not account for -- and one note per removal and per
/// restoration. A removal counts only with a reason; with none, the row is
/// restored. No answer at all, or a draft with no rows, leaves the draft as it
/// was.
#[must_use]
pub fn merge_refined(draft: &str, answer: Option<&str>) -> (String, Vec<String>) {
    let Some(answer) = answer else {
        return (draft.to_string(), Vec::new());
    };
    let draft_rows = extracted_rows(draft);
    if draft_rows.is_empty() {
        return (answer.to_string(), Vec::new());
    }
    let kept: Vec<&str> = extracted_rows(answer)
        .into_iter()
        .filter(|row| !name_of(row).eq_ignore_ascii_case(DROPPED))
        .collect();
    let mut notes = Vec::new();
    let mut reasoned = Vec::new();
    for (name, reason) in removals(answer) {
        // A "removal" of a row the answer also kept removed nothing: on the
        // 2026-10-10 replay raptor echoed all 80 rows AND wrote "DROPPED | X |
        // Duplicate of X" for 37 of them. Noting those would report losses that
        // did not happen.
        if reason.is_empty()
            || kept
                .iter()
                .any(|k| significant_tokens(&name_of(k)) == significant_tokens(&name))
        {
            continue;
        }
        notes.push(format!("refine dropped '{name}': {reason}"));
        reasoned.push(name);
    }
    let mut rows: Vec<&str> = kept.clone();
    for row in draft_rows {
        let name = name_of(row);
        let accounted = kept.iter().any(|k| same_entry(&name_of(k), &name))
            || reasoned.iter().any(|r| same_entry(r, &name));
        if !accounted {
            notes.push(format!(
                "refine left out '{name}' without a reason; kept from the draft"
            ));
            rows.push(row);
        }
    }
    (rows.join("\n"), notes)
}

/// Phase 3: the model merges same-event entries and removes non-events, with
/// a reason for each removal; the answer is laid back over the draft
/// ([`merge_refined`]) so nothing is lost without one.
#[must_use]
pub fn refine_draft(
    draft_text: &str,
    config: &crate::config::ZtoolsConfig,
    log: &PhaseLog,
) -> String {
    let prompt = render(PHASE_REFINE, &[("draft_text", draft_text)]);
    let answer = call_llm_text(&prompt, config);
    log.record("refine", &prompt, answer.as_deref());
    let (rows, notes) = merge_refined(draft_text, answer.as_deref());
    for note in notes {
        println!("\u{2192} {note}");
    }
    rows
}

#[cfg(test)]
#[path = "refine_tests.rs"]
mod tests;
