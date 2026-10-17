//! T3: an answer is repaired to what can be verified before it is judged.
//!
//! On 2026-10-10 every summary attempt on one 43-tweet timeline was thrown
//! away whole, and none of the reasons was what it seemed:
//! - "32 of 32 citations name no tweet the model was given" -- each one did:
//!   the model dropped the year (`Sat Oct 10 09:46:22 +0000` for `... +0000
//!   2026`), and every shortened stamp still named exactly one given tweet.
//! - "36 of 43 bullets repeat an earlier bullet" -- seven real, cited bullets,
//!   then a loop reciting two of them under every later heading.
//! - "1 of 8 citations" -- seven good bullets lost to one bad one.
//!
//! On 2026-10-16 `twitter-evening` lost both models the same way: raptor wrote
//! `(TomerPersico | Fri Oct 16 14:21:56 +0000 2026)` with no `@` ("23 of 25
//! bullets lack the attribution"), and foundation kept only the time of day,
//! `(@Uvalajnr | 04:11:19)` ("42 of 42 citations name no tweet"), joining a
//! bullet's sources in one bracket with `;` on other days. The class is a
//! faithful citation in a form this grammar did not parse; every form seen in
//! the store's `rejected/` is parsed below, and each still has to name exactly
//! one given tweet before it is rewritten.
//!
//! What is verifiably wrong is removed here; what is verifiably right stays;
//! the gate then judges the remainder (`handle_model_output`), and the saved
//! summary says what was removed. An answer that mostly cites nothing it was
//! given is inventing, and is rejected rather than trimmed.

use std::collections::HashSet;
use std::sync::LazyLock;

use regex::Regex;

use super::quality::{drop_bullets, drop_recited};

/// A citation group in any bracketing the models write: `(@h | ts)`,
/// `[@h | ts]`, either mixed (`[@h | ts)`), optionally bold, the `@` optional,
/// and several sources joined by `;` inside one bracket. The repair rewrites
/// each source to the one form the gate reads, `(@h | ts)`.
static ANY_CITATION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:\*\*)?[\[(](@?[A-Za-z0-9_]+\s*\|[^)\]\n]+?)\s*[)\]](?:\*\*)?")
        .expect("valid regex")
});

/// One source inside a citation group: `@h | ts`, the `@` optional.
static SOURCE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*(@?)([A-Za-z0-9_]+)\s*\|\s*(.+?)\s*$").expect("valid regex"));

/// A time of day as the source line writes it, to the minute at least:
/// `04:11:19` or `04:11`.
static TIME_OF_DAY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\d{2}:\d{2}(?::\d{2})?$").expect("valid regex"));

/// The shortest timestamp prefix that still names a tweet: down to the minute
/// (`Sat Oct 10 09:46`).
const MIN_STAMP_PREFIX: usize = 16;

/// Most tweets one bullet may cite.
///
/// More is unrelated claims run together: across 300+ accepted summaries (2026-07 to 2026-10) no bullet cited more
/// than three, and the fallback's rejected blob cited 19 in one.
pub const MAX_CITES_PER_BULLET: usize = 3;

/// What survives a repair, and what the repair did.
#[derive(Debug)]
pub struct Repaired {
    pub body: String,
    pub notes: Vec<String>,
}

/// The full timestamp of the given tweet a citation names, if exactly one.
///
/// `stamp` is that tweet's own, a prefix (to the minute at least) of exactly
/// one of `handle`'s given tweets, or the time of day (to the minute at least)
/// of exactly one of them. `None` when it names none, or more than one.
#[must_use]
pub fn resolve<'a>(handle: &str, stamp: &str, sources: &'a [(String, String)]) -> Option<&'a str> {
    let stamp = stamp.trim();
    let theirs = sources
        .iter()
        .filter(|(h, _)| h == handle)
        .map(|(_, ts)| ts.trim());
    if let Some(exact) = theirs.clone().find(|ts| *ts == stamp) {
        return Some(exact);
    }
    if TIME_OF_DAY.is_match(stamp) {
        return only(theirs.filter(|ts| {
            ts.split_whitespace().nth(3).is_some_and(|time| {
                time == stamp || time.strip_prefix(stamp).is_some_and(|s| s.starts_with(':'))
            })
        }));
    }
    if stamp.chars().count() < MIN_STAMP_PREFIX {
        return None;
    }
    only(theirs.filter(|ts| ts.starts_with(stamp)))
}

/// The one item, or `None` when there are none or several.
fn only<'a>(mut matching: impl Iterator<Item = &'a str>) -> Option<&'a str> {
    match (matching.next(), matching.next()) {
        (Some(one), None) => Some(one),
        _ => None,
    }
}

/// The sources one citation group names, as `(has_at, handle, stamp)`.
///
/// Several only when every `;`-separated part is itself a source; otherwise
/// the group is one source whose stamp is everything after the first bar, as
/// it always was.
fn group_sources(group: &str) -> Vec<(bool, &str, &str)> {
    let parts: Option<Vec<_>> = group.split(';').map(parse_source).collect();
    match parts {
        Some(parts) if parts.len() > 1 => parts,
        _ => parse_source(group).into_iter().collect(),
    }
}

/// One `@h | ts` source, as `(has_at, handle, stamp)`.
fn parse_source(part: &str) -> Option<(bool, &str, &str)> {
    let c = SOURCE.captures(part)?;
    Some((
        !c.get(1)?.as_str().is_empty(),
        c.get(2)?.as_str(),
        c.get(3)?.as_str(),
    ))
}

/// Repair `summary` against the tweets it was given.
///
/// # Errors
///
/// When more than a third of its citations name no given tweet: the model is
/// inventing sources, and what is left would be chosen by chance.
pub fn repair(summary: &str, sources: &[(String, String)]) -> Result<Repaired, String> {
    let mut total = 0usize;
    let mut unresolved = 0usize;
    let given: HashSet<&str> = sources.iter().map(|(h, _)| h.as_str()).collect();
    let body = ANY_CITATION
        .replace_all(summary, |cap: &regex::Captures<'_>| {
            let parts = group_sources(&cap[1]);
            // A bracket with a bar is a citation when it says so (`@`) or names
            // someone the model was given; `(wins | losses)` is prose.
            if !parts.iter().any(|(at, h, _)| *at || given.contains(h)) {
                return cap[0].to_string();
            }
            parts
                .iter()
                .map(|(_, handle, stamp)| {
                    total += 1;
                    resolve(handle, stamp, sources).map_or_else(
                        || {
                            unresolved += 1;
                            format!("(@{handle} | {})", stamp.trim())
                        },
                        |full| format!("(@{handle} | {full})"),
                    )
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .into_owned();
    if unresolved * 3 > total {
        return Err(format!(
            "{unresolved} of {total} citations name no tweet the model was given \
             (a garbled or invented handle or timestamp)"
        ));
    }
    let known: HashSet<(String, String)> = sources
        .iter()
        .map(|(h, ts)| (h.clone(), ts.trim().to_string()))
        .collect();
    let mut notes = Vec::new();
    let (body, invented) = drop_bullets(&body, |cites| cites.iter().all(|c| known.contains(c)));
    if invented > 0 {
        notes.push(format!(
            "dropped {invented} bullet(s) citing no tweet the model was given"
        ));
    }
    let (body, run_on) = drop_bullets(&body, |cites| cites.len() <= MAX_CITES_PER_BULLET);
    if run_on > 0 {
        notes.push(format!(
            "dropped {run_on} bullet(s) citing more than {MAX_CITES_PER_BULLET} tweets \
             (unrelated claims run together)"
        ));
    }
    let (body, recited) = drop_recited(&body);
    if recited > 0 {
        notes.push(format!(
            "dropped {recited} bullet(s) that repeat a tweet already cited (a repetition loop)"
        ));
    }
    Ok(Repaired { body, notes })
}

/// A model's answer as it will be saved: the repaired body, how many tweets it
/// was given, and what the repair changed (said in the saved document).
#[derive(Debug)]
pub struct Answer {
    pub body: String,
    pub processed: usize,
    pub repairs: Vec<String>,
}

/// Fewest verified bullets an answer may keep: this many, or the fewest that
/// could cite every tweet it was given ([`MAX_CITES_PER_BULLET`] each).
///
/// A quiet timeline makes few bullets, and the second bound allows for that.
/// No accepted summary since 2026-08-20 has fewer than seven; seven verified
/// bullets is what the 2026-10-10 loops kept after repair. One clean bullet
/// for 43 tweets (the fallback's run-on blob) is not a summary of them.
pub const MIN_BULLETS: usize = 5;

/// One model attempt's post-call branch: split thinking, repair, gate, merge.
///
/// The repair (`repair.rs`) removes only what is verifiably wrong -- a bullet
/// whose citation names no given tweet, one citing more than
/// [`MAX_CITES_PER_BULLET`], one re-citing tweets already cited (a loop's
/// repeats) -- and rewrites shortened citations to their full
/// form. The gate then judges the REPAIRED body; an appended `## Analysis`
/// never faces it. `cut` says the server stopped the answer at the token limit
/// and the caller kept only its whole lines. Port of `_summarize_with_model`'s
/// post-call branch in `references/twitter/summarize.py`, plus T3.
///
/// # Errors
///
/// The repair's or the gate's reasons, joined, when the answer must not be
/// saved.
pub fn handle_model_output(
    content: &str,
    sources: &[(String, String)],
    cut: bool,
) -> Result<Answer, String> {
    let processed = sources.len();
    let (thinking, cleaned) = super::extract_thinking(content);
    let body = if thinking.is_empty() {
        crate::ztools::eval::clean::remove_thinking_blocks(&cleaned)
    } else {
        cleaned
    };
    let repaired = repair(&body, sources)?;
    let mut repairs = repaired.notes;
    if cut {
        repairs.insert(
            0,
            "the answer was cut at the length limit; its unfinished last line was dropped"
                .to_string(),
        );
    }
    let quality = super::check_summary_quality(&repaired.body, processed);
    if quality.rejected() {
        return Err(quality.rejections.join("; "));
    }
    let kept = super::quality::bullets(&repaired.body).len();
    if kept < MIN_BULLETS.min(processed.div_ceil(MAX_CITES_PER_BULLET)) {
        let after = if repairs.is_empty() {
            String::new()
        } else {
            format!(" after repair ({})", repairs.join("; "))
        };
        return Err(format!(
            "{kept} verified bullet(s){after} for {processed} tweets: too few to stand for the timeline"
        ));
    }
    Ok(Answer {
        body: super::merge_thinking_with_summary(&thinking, &repaired.body),
        processed,
        repairs,
    })
}

/// One summary call. A cut answer's whole lines come back flagged `true`: they
/// are bullets like any other, judged by the repair and the gate, and the
/// saved summary says the answer was cut.
///
/// # Errors
///
/// The call's own failure, or a cut answer with no whole line.
pub fn ask(
    base_url: &str,
    model: &str,
    prompt: &str,
    timeout_secs: u64,
    config: &crate::config::ZtoolsConfig,
) -> anyhow::Result<(String, bool)> {
    match super::call_osaurus(base_url, model, prompt, timeout_secs, config) {
        Ok(raw) => Ok((raw, false)),
        Err(e) => e
            .downcast_ref::<crate::ztools::llm::CutAnswer>()
            .and_then(crate::ztools::llm::CutAnswer::whole_lines)
            .map_or_else(|| Err(e), |whole| Ok((whole, true))),
    }
}

#[cfg(test)]
#[path = "repair_tests.rs"]
mod tests;
