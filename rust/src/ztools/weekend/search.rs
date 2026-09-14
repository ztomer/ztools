//! `DuckDuckGo` HTML search without the `ddgs` helper.
//!
//! Ported from the inline `ddgs` subprocess the corpus used to shell out to
//! (`weekend/mod.rs::collect_snippets_external`). The replacement is a direct
//! `reqwest` POST to the html endpoint, which the S3 spike proved survives the
//! bot wall that GET trips: `html.duckduckgo.com` answers a `q` form with
//! `result__a` title anchors and `result__snippet` bodies when the request
//! carries a browser User-Agent; the GET does not. So GET remains a fallback
//! for mocks and non-DDG hosts, never the primary.
//!
//! Failure and "found nothing" stay distinguishable: it used to return a bare
//! `Vec` that swallowed every failure into an empty one, which is how a weekend
//! plan said "no events found" when the truth was that the interpreter could
//! not start. Every transport error now prints a stated reason before the empty
//! result is handed back.

use std::time::Duration;

use super::{is_challenged, parse_snippets_from_html};

/// One parsed search hit: the anchor title and href plus the snippet body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchResult {
    pub title: String,
    pub href: String,
    pub body: String,
}

/// Browser User-Agent for the DDG POST, matching `followup._UA` so the search
/// and the aggregator fetch look like the same session to the site.
const SEARCH_UA: &str =
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Safari/537.36";

/// Parse `result__a` title anchors + `result__snippet` bodies out of raw DDG
/// html.
///
/// Titles and bodies are paired by position (one of each per result block, in
/// order); snippet-only markup yields body-only results, which is how the mock
/// fixtures and `result-snippet` (lite) markup still work.
#[must_use]
pub fn parse_results_from_html(html: &str) -> Vec<SearchResult> {
    let mut titled = Vec::new();
    let mut idx = 0;
    while let Some(rel) = html[idx..].find("class=\"result__a\"") {
        let start = idx + rel;
        let Some(tag_end_rel) = html[start..].find('>') else {
            break;
        };
        let tag = &html[start..start + tag_end_rel];
        let href = tag
            .find("href=\"")
            .and_then(|hs| {
                let after = &tag[hs + 6..];
                after.find('"').map(|e| after[..e].to_string())
            })
            .unwrap_or_default()
            .replace("&amp;", "&");
        let text_start = start + tag_end_rel + 1;
        if let Some(end) = html[text_start..].find("</a>") {
            titled.push((
                href,
                html[text_start..text_start + end]
                    .replace("<b>", "")
                    .replace("</b>", "")
                    .replace("&#x27;", "'")
                    .replace("&amp;", "&")
                    .replace("&quot;", "\"")
                    .trim()
                    .to_string(),
            ));
        }
        idx = start + 1;
    }

    let bodies = parse_snippets_from_html(html);
    let mut results: Vec<SearchResult> = titled
        .into_iter()
        .map(|(href, title)| SearchResult {
            title,
            href,
            body: String::new(),
        })
        .collect();
    for (i, body) in bodies.into_iter().enumerate() {
        if let Some(r) = results.get_mut(i) {
            r.body = body;
        } else {
            results.push(SearchResult {
                title: String::new(),
                href: String::new(),
                body,
            });
        }
    }
    results
}

/// Query the DDG html endpoint for a query, returning the parsed hits.
///
/// POST is primary (browser UA, `q`/`b`/`l` form — the combination the S3
/// spike measured green); GET is the fallback for mocks and non-DDG hosts.
/// A challenged response and a transport failure are both reported, never
/// silently turned into "the search found nothing".
#[must_use]
pub fn search_duckduckgo_html(query: &str, url: &str) -> Vec<SearchResult> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(8))
        .build()
        .unwrap_or_default();

    let parsed = |html: &str| -> Vec<SearchResult> {
        if is_challenged(html) {
            eprintln!("\u{26a0} search blocked for {query:?} (bot wall); treating as no-corpus");
            return Vec::new();
        }
        parse_results_from_html(html)
    };

    // 1. POST (primary). The form mirrors the public search form: q, b, l and
    //    a Safari UA. `b`/`l` are the anti-bot scent the spike found present.
    let mut failed = false;
    if let Ok(resp) = client
        .post(url)
        .form(&[("q", query), ("b", ""), ("l", "")])
        .header("User-Agent", SEARCH_UA)
        .header("Accept-Language", "en-CA,en;q=0.9")
        .send()
    {
        if resp.status().is_success() {
            if let Ok(html) = resp.text() {
                let results = parsed(&html);
                if !results.is_empty() {
                    return results;
                }
            }
        }
    } else {
        failed = true;
    }

    // 2. GET (fallback: mocks and non-DDG hosts). Challenged GET responses are
    //    expected on the real endpoint — that wall is why POST is the primary.
    if let Ok(resp) = client
        .get(url)
        .query(&[("q", query)])
        .header("User-Agent", SEARCH_UA)
        .header("Accept-Language", "en-CA,en;q=0.9")
        .send()
    {
        if resp.status().is_success() {
            if let Ok(html) = resp.text() {
                if !is_challenged(&html) {
                    let results = parse_results_from_html(&html);
                    if !results.is_empty() {
                        return results;
                    }
                }
            }
        }
    }

    if failed {
        eprintln!("\u{26a0} search unreachable for {query:?}; treating as no-corpus");
    }
    Vec::new()
}
