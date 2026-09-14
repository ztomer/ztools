//! Follow the most promising aggregator pages and turn their text into corpus
//! candidates. Port of `references/weekend/followup.py`.
//!
//! A directory page is not an activity, but search for "events" returns
//! almost nothing BUT directory pages, so discarding them discards the whole
//! harvest. The events are INSIDE those pages: follow a bounded number of the
//! promising ones and add their readable text to the corpus instead.
//!
//! Bounded on purpose — a scheduled unattended run must not hang on a slow
//! site: at most `FOLLOW_LIMIT` pages, `FETCH_TIMEOUT` seconds each,
//! `MAX_PAGE_CHARS` kept per page.

use std::time::Duration;

use super::enforce::normalize_for_match;
use super::SearchResult;
use crate::ztools::weekend_cache::{has_region_evidence, RegionLists};

/// Phrases that mark a page as a DIRECTORY of activities rather than an
/// activity. Single source of truth for the weekend pipeline (the report
/// checkers keep their own copy in `eval/` because they judge curated rows,
/// and Rust `format.rs` reads the same list from there).
const AGGREGATOR_MARKERS: &[&str] = &[
    "things to do",
    "what's on",
    "whats on",
    "events guide",
    "activities guide",
    "events & activities",
    "events and activities",
    "guides for",
    "calendar of events",
    "event calendar",
    "event listings",
    "directory",
    "round-up",
    "roundup",
    "archives",
    "best places",
    "top 10",
    "your guide",
    "festivals",
    "fairs",
    "events in",
    "what to do",
];

const FOLLOW_LIMIT: usize = 3;
const FETCH_TIMEOUT_SECS: u64 = 8;
const MAX_PAGE_CHARS: usize = 4000;
const MIN_CANDIDATE_CHARS: usize = 8;
const MAX_LINES_PER_PAGE: usize = 60;

const FOLLOW_UA: &str =
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Safari/537.36";

/// Is this result a page that LISTS events?
///
/// Used here to decide what is worth FOLLOWING — the opposite of how the same
/// signal is used in the report checker, where it flags a row that should
/// never have been listed as an activity.
#[must_use]
pub fn looks_like_aggregator(title: &str) -> bool {
    let normalized = normalize_for_match(title);
    AGGREGATOR_MARKERS
        .iter()
        .any(|marker| normalized.contains(marker))
}

/// Readable text from `url`, or "" on any failure.
///
/// Returns "" rather than an error: a scheduled run must degrade to "fewer
/// events" and say so, never abort because one site was slow. Tag stripping is
/// raw-HTML (no bs4 in Rust): the block elements that carry no content are
/// removed wholesale, then one source line per listing is kept — collapsing
/// everything into one line made each followed page a single blob the
/// extractor could not read.
#[must_use]
pub fn fetch_page_text(url: &str) -> String {
    let client = match reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(FETCH_TIMEOUT_SECS))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("\u{26a0} Could not read {}: {e}", take60(url));
            return String::new();
        }
    };
    match client
        .get(url)
        .header("User-Agent", FOLLOW_UA)
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .and_then(reqwest::blocking::Response::text)
    {
        Ok(html) => extract_page_text(&html, MAX_PAGE_CHARS),
        Err(e) => {
            eprintln!("\u{26a0} Could not read {}: {e}", take60(url));
            String::new()
        }
    }
}

/// Strip markup to source-line-structured text, bounded to `max_chars`.
#[must_use]
pub(crate) fn extract_page_text(html: &str, max_chars: usize) -> String {
    let mut out = String::new();
    let mut i = 0;
    // Name of a noise block currently being skipped to its matching close.
    let mut discard: Option<&'static str> = None;
    while i < html.len() {
        let Some(lt) = html[i..].find('<') else {
            if discard.is_none() {
                out.push_str(&html[i..]);
            }
            break;
        };
        if let Some(tag) = discard {
            let candidate = &html[i + lt..];
            if candidate.starts_with("</") {
                let (name, consumed) = close_tag_name(candidate);
                if name.eq_ignore_ascii_case(tag) {
                    i += consumed;
                    discard = None;
                    continue;
                }
            }
            i += lt + 1;
            continue;
        }
        out.push_str(&html[i..i + lt]);
        let after = &html[i + lt + 1..];
        let Some(gt) = after.find('>') else {
            out.push_str(after);
            break;
        };
        let inner = &after[..gt];
        let is_close = inner.starts_with('/');
        let raw_name = if is_close { &inner[1..] } else { inner };
        let name = raw_name.split_whitespace().next().unwrap_or("");
        let lname = name.to_ascii_lowercase();
        if !is_close && is_noise_tag(&lname) {
            discard = noise_tag_name(&lname);
        } else if !is_close && is_block_tag(&lname) && !lname.is_empty() {
            out.push('\n');
        }
        i += lt + 1 + gt + 1;
    }

    // Collapse per-line whitespace like bs4's get_text(separator="\n", strip).
    out.lines()
        .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
        .chars()
        .take(max_chars)
        .collect()
}

/// Parse a `</name ...>` close tag, returning the name and the byte length to
/// advance (past the closing `>`).
fn close_tag_name(candidate: &str) -> (&str, usize) {
    let inner = &candidate[2..];
    let end = inner.find('>').unwrap_or(inner.len());
    let name = inner[..end].split_whitespace().next().unwrap_or("");
    (name, 2 + end + 1)
}

fn is_noise_tag(name: &str) -> bool {
    matches!(
        name,
        "script" | "style" | "nav" | "header" | "footer" | "form" | "noscript"
    )
}

fn noise_tag_name(name: &str) -> Option<&'static str> {
    match name {
        "script" => Some("script"),
        "style" => Some("style"),
        "nav" => Some("nav"),
        "header" => Some("header"),
        "footer" => Some("footer"),
        "form" => Some("form"),
        "noscript" => Some("noscript"),
        _ => None,
    }
}

fn is_block_tag(name: &str) -> bool {
    matches!(
        name,
        "p" | "div"
            | "li"
            | "tr"
            | "td"
            | "h1"
            | "h2"
            | "h3"
            | "br"
            | "section"
            | "article"
            | "ul"
            | "ol"
    )
}

/// Truncate a URL to 60 chars for a warning line.
fn take60(url: &str) -> String {
    url.chars().take(60).collect()
}

/// Render fetched page text as one `- ` candidate per line.
///
/// The source title is carried on every line so a row can still be traced back
/// to the page it came from after the corpus is flattened. Deliberately
/// permissive: nav chrome is 1-2 words, a real entry can be short. Starving
/// the pipeline is the failure this whole module exists to fix.
#[must_use]
pub fn as_candidate_lines(text: &str, title: &str) -> String {
    let mut lines = Vec::new();
    for raw in text.lines() {
        let line = raw.split_whitespace().collect::<Vec<_>>().join(" ");
        if line.chars().count() < MIN_CANDIDATE_CHARS {
            continue;
        }
        lines.push(format!("- [{title}] {line}"));
        if lines.len() >= MAX_LINES_PER_PAGE {
            break;
        }
    }
    lines.join("\n")
}

/// Fetch the most promising aggregator pages and return their text.
///
/// Only results that look like a directory AND carry in-region evidence are
/// followed, so the budget is spent on pages likely to list local events.
#[must_use]
pub fn follow_aggregators(results: &[SearchResult], region: &RegionLists) -> String {
    let mut followed = Vec::new();
    for r in results {
        if followed.len() >= FOLLOW_LIMIT {
            break;
        }
        let title = r.title.trim();
        let url = r.href.trim();
        if url.is_empty() || !looks_like_aggregator(title) {
            continue;
        }
        if !has_region_evidence(&format!("{title} {}", r.body), region) {
            continue;
        }
        let text = fetch_page_text(url);
        if !text.is_empty() {
            let block = as_candidate_lines(&text, title);
            if !block.is_empty() {
                followed.push(block);
            }
        }
    }
    if !followed.is_empty() {
        println!("\u{2192} Followed {} event listing page(s)", followed.len());
    }
    followed.join("\n")
}
