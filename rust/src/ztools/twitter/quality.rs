//! The gate a model's answer must pass before it is saved as a summary.
//!
//! THE INVARIANT. A saved summary is a DISTILLATION of the tweets it was given:
//! each bullet states one fact that no other bullet states, cites the tweet(s)
//! it came from in the `(@handle | timestamp)` form the prompt requires, and
//! there are never more bullets than tweets, because a bullet that cites no
//! tweet of its own is either a repeat or an invention.
//!
//! WHY THE OLD GATE WAS NOT ENOUGH. It rejected an answer only when it had no
//! `##` header AND no bullet, so anything with one heading passed. In October
//! 2026 the ~1B-active-parameter summarizer fell into repetition loops at
//! temperature 0 — one run turned 43 tweets into 58 bullets, 35 of them
//! duplicates — and those loops were saved as PRIMARY output with no degraded
//! banner, because nothing between the model and the file could see a loop.
//! Others wrote the attribution as `[@handle | ts]`, which the dashboard and the
//! eval's attribution checks do not read as a citation. Each shape now rejects
//! the answer, and a rejected answer falls to the next model in the chain
//! (`chain::run_chain`); it is never saved.

use std::collections::HashSet;
use std::sync::LazyLock;

use regex::Regex;

/// More exact-or-near duplicate bullets than this rejects the answer. Not zero:
/// two stories can legitimately close on the same sentence, and a single
/// restated fact is a weak summary, not a loop.
pub const MAX_DUPLICATE_BULLETS: usize = 2;

/// Two bullets whose word sets overlap at least this many percent (Jaccard:
/// shared words over all words) state the
/// same thing. A loop repeats a bullet with at most a word or two changed; two
/// different facts from one account share the handle and little else.
const NEAR_DUPLICATE_PERCENT: usize = 85;

/// Bullets shorter than this many words are compared exactly, never by
/// overlap: on three words one changed word is a different fact.
const NEAR_DUPLICATE_MIN_WORDS: usize = 5;

/// The required citation: `(@handle | timestamp)`. Position is not policed (a
/// merged bullet cites several sources in a row), the delimiters are.
static ATTRIBUTION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\(@[A-Za-z0-9_]+\s*\|[^)\n]+\)").expect("valid regex"));

/// One citation, with the handle and timestamp captured for checking against
/// the tweets the model was given.
static CITATION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\(@([A-Za-z0-9_]+)\s*\|\s*([^)\n]+?)\s*\)").expect("valid regex")
});

/// Citations that name no tweet the model was given.
///
/// A garbled timestamp, an invented one, a handle that was never in the
/// timeline. `sources` are the `(handle, created_at)` pairs the prompt showed.
/// `None` when at most a tenth are unmatched (a stray reformat is a weak
/// answer, not a fabricated one), else the reason. Ground truth, where
/// `is_attributed` only sees a shape.
#[must_use]
pub fn unmatched_citations(summary: &str, sources: &[(String, String)]) -> Option<String> {
    let known: HashSet<(&str, &str)> = sources
        .iter()
        .map(|(h, ts)| (h.as_str(), ts.trim()))
        .collect();
    let mut total = 0usize;
    let mut unmatched = 0usize;
    for cap in CITATION.captures_iter(summary) {
        total += 1;
        if !known.contains(&(&cap[1], cap[2].trim())) {
            unmatched += 1;
        }
    }
    (unmatched * 10 > total).then(|| {
        format!(
            "{unmatched} of {total} citations name no tweet the model was given \
             (a garbled or invented handle or timestamp)"
        )
    })
}

/// The answer with every bullet dropped whose citations ALL name tweets an
/// earlier bullet already cited, and how many were dropped.
///
/// Such a bullet adds no source the reader lacks -- it is the same tweet told
/// twice, which the prompt forbids and the loop rule tolerates up to
/// [`MAX_DUPLICATE_BULLETS`]. Whether two tweets say the same thing is a
/// judgement and stays the model's; whether two bullets cite the same tweet
/// is not. A topic left with no bullets loses its header. Unchanged, byte for
/// byte, when nothing is dropped.
#[must_use]
pub fn drop_recited(summary: &str) -> (String, usize) {
    // Blocks: a bullet with its indented continuation lines, or one other line.
    let mut blocks: Vec<(String, bool)> = Vec::new();
    for line in summary.split_inclusive('\n') {
        let trimmed = line.trim();
        let continues = blocks.last().is_some_and(|(_, bullet)| *bullet)
            && !trimmed.is_empty()
            && line.starts_with(char::is_whitespace);
        if continues {
            if let Some((text, _)) = blocks.last_mut() {
                text.push_str(line);
            }
        } else {
            let bullet = trimmed.starts_with("- ") || trimmed.starts_with("* ");
            blocks.push((line.to_string(), bullet));
        }
    }
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut dropped = 0;
    let mut kept: Vec<&str> = Vec::new();
    // Per section: where its header sits in `kept`, and whether it lost a bullet.
    let mut section: Option<(usize, bool)> = None;
    let close = |kept: &mut Vec<&str>, section: Option<(usize, bool)>| {
        if let Some((at, lost)) = section
            && lost
            && kept[at + 1..].iter().all(|l| l.trim().is_empty())
        {
            kept.truncate(at);
        }
    };
    for (text, bullet) in &blocks {
        if text.trim_start().starts_with("##") {
            close(&mut kept, section);
            section = Some((kept.len(), false));
        } else if *bullet {
            let cites: Vec<(String, String)> = CITATION
                .captures_iter(text)
                .map(|c| (c[1].to_string(), c[2].trim().to_string()))
                .collect();
            if !cites.is_empty() && cites.iter().all(|c| seen.contains(c)) {
                dropped += 1;
                if let Some((_, lost)) = section.as_mut() {
                    *lost = true;
                }
                continue;
            }
            seen.extend(cites);
        }
        kept.push(text);
    }
    close(&mut kept, section);
    if dropped == 0 {
        return (summary.to_string(), 0);
    }
    let mut out = kept.concat().trim_end().to_string();
    out.push('\n');
    (out, dropped)
}

/// Non-bullet prose under a topic header: the model talking to itself
/// ("Actually wait - I realize I've been overthinking this"). Prose is the
/// Executive Summary's alone.
fn prose_in_topics(summary: &str) -> usize {
    let mut in_topic = false;
    let mut prose = 0;
    for line in summary.lines() {
        let trimmed = line.trim();
        if let Some(title) = trimmed.strip_prefix("##") {
            in_topic = !title.to_lowercase().contains("summary");
            continue;
        }
        let bullet = trimmed.starts_with("- ") || trimmed.starts_with("* ");
        let continuation = line.starts_with(char::is_whitespace);
        if in_topic && !trimmed.is_empty() && !bullet && !continuation {
            prose += 1;
        }
    }
    prose
}

/// What the gate found: `warnings` describe a weak answer that is still saved,
/// `rejections` an answer that must not be.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Quality {
    pub warnings: Vec<String>,
    pub rejections: Vec<String>,
}

impl Quality {
    /// Whether the answer must not be saved.
    #[must_use]
    pub const fn rejected(&self) -> bool {
        !self.rejections.is_empty()
    }
}

/// The bullets of a markdown answer, each with its indented continuation lines
/// joined on, so a citation wrapped onto the next line still belongs to its
/// bullet.
#[must_use]
pub fn bullets(summary: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut open = false;
    for line in summary.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed
            .strip_prefix("- ")
            .or_else(|| trimmed.strip_prefix("* "))
        {
            out.push(rest.trim().to_string());
            open = true;
        } else if open
            && !trimmed.is_empty()
            && !trimmed.starts_with('#')
            && line.starts_with(char::is_whitespace)
        {
            if let Some(last) = out.last_mut() {
                last.push(' ');
                last.push_str(trimmed);
            }
        } else {
            open = false;
        }
    }
    out
}

/// Whether a bullet carries at least one `(@handle | timestamp)` citation.
#[must_use]
pub fn is_attributed(bullet: &str) -> bool {
    ATTRIBUTION.is_match(bullet)
}

/// Lowercased alphanumeric words: case, punctuation and spacing never make two
/// bullets different.
fn words(bullet: &str) -> Vec<String> {
    bullet
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn same_fact(a: &[String], b: &[String]) -> bool {
    if a == b {
        return true;
    }
    if a.len() < NEAR_DUPLICATE_MIN_WORDS || b.len() < NEAR_DUPLICATE_MIN_WORDS {
        return false;
    }
    let a: HashSet<&String> = a.iter().collect();
    let b: HashSet<&String> = b.iter().collect();
    let shared = a.intersection(&b).count();
    let union = a.union(&b).count();
    union > 0 && shared * 100 >= union * NEAR_DUPLICATE_PERCENT
}

/// How many bullets repeat an EARLIER bullet exactly or nearly.
#[must_use]
pub fn duplicate_bullets(bullets: &[String]) -> usize {
    let normalised: Vec<Vec<String>> = bullets.iter().map(|b| words(b)).collect();
    normalised
        .iter()
        .enumerate()
        .filter(|(i, b)| normalised[..*i].iter().any(|earlier| same_fact(earlier, b)))
        .count()
}

/// Validate a model answer for `input_tweets` tweets.
///
/// Rejects (see the module's invariant): an empty answer; one with neither a
/// `##` header nor a bullet; more than [`MAX_DUPLICATE_BULLETS`] repeated
/// bullets; more bullets than input tweets; an answer cut off mid-citation;
/// prose inside a topic section; and bullets that mostly lack the
/// `(@handle | timestamp)` citation. Whether those citations name real input
/// tweets is [`unmatched_citations`], which needs the tweets themselves.
#[must_use]
pub fn check_summary_quality(summary: &str, input_tweets: usize) -> Quality {
    let mut quality = Quality::default();
    if summary.trim().is_empty() {
        quality.rejections.push("Summary is empty".to_string());
        return quality;
    }
    let header_count = summary
        .lines()
        .filter(|l| l.trim().starts_with("##"))
        .count();
    let char_count: usize = summary.lines().map(|l| l.trim().len()).sum();
    let bullets = bullets(summary);

    if header_count == 0 {
        quality.warnings.push("No ## headers".to_string());
    }
    if bullets.len() < 3 {
        quality
            .warnings
            .push(format!("Only {} bullet points", bullets.len()));
    }
    if char_count < 100 {
        quality
            .warnings
            .push(format!("Very short ({char_count} chars)"));
    }

    if header_count == 0 && bullets.is_empty() {
        quality
            .rejections
            .push("no ## header and no bullet: the answer has no structure".to_string());
    }
    let duplicates = duplicate_bullets(&bullets);
    if duplicates > MAX_DUPLICATE_BULLETS {
        quality.rejections.push(format!(
            "{duplicates} of {} bullets repeat an earlier bullet (a repetition loop)",
            bullets.len()
        ));
    }
    if bullets.len() > input_tweets {
        quality.rejections.push(format!(
            "{} bullets for {input_tweets} input tweets: more bullets than tweets",
            bullets.len()
        ));
    }
    // The model stopped mid-sentence: the last bullet opens a citation it
    // never closes. Seen live 2026-10-10 -- 8 bullets for 45 tweets ending
    // `(@AION2Official | Sat Oct 10 336:51 +...` -- and every other rule
    // passed it. `llm::complete` refuses a token-limit stop; this catches a cut
    // the server did not report.
    if bullets.last().is_some_and(|last| {
        last.rfind("(@")
            .is_some_and(|open| !last[open..].contains(')'))
    }) {
        quality.rejections.push(
            "the last bullet is cut off mid-citation: the answer stops before it ends".to_string(),
        );
    }
    let prose = prose_in_topics(summary);
    if prose > 0 {
        quality.rejections.push(format!(
            "{prose} line(s) of prose inside topic sections: the model is talking to itself"
        ));
    }
    let unattributed = bullets.iter().filter(|b| !is_attributed(b)).count();
    if unattributed * 2 > bullets.len() {
        quality.rejections.push(format!(
            "{unattributed} of {} bullets lack the `(@handle | timestamp)` attribution",
            bullets.len()
        ));
    }
    quality
}

#[cfg(test)]
#[path = "quality_tests.rs"]
mod tests;
