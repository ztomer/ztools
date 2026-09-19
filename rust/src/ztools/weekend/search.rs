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

/// Parse Bing's web results: one `<li class="b_algo">` per hit, with the
/// title anchor inside its `<h2>` and the snippet in the first `<p>`.
///
/// Bing is the SECOND engine, not a replacement: `DuckDuckGo` was chosen for
/// its snippets and stays primary. But on 2026-09-19 every one of a run's
/// twelve DDG queries came back as an `anomaly-modal` challenge (both the
/// `html` and `lite` endpoints, POST and GET), and Bing answered the same
/// queries with ten results each, so a planner with one engine had no corpus
/// at all — for weeks, every plan said "no events" and meant "no search".
#[must_use]
pub fn parse_bing_results(html: &str) -> Vec<SearchResult> {
    let mut results = Vec::new();
    let mut idx = 0;
    while let Some(rel) = html[idx..].find("class=\"b_algo\"") {
        let start = idx + rel;
        let end = html[start..]
            .find("</li>")
            .map_or(html.len(), |e| start + e);
        let block = &html[start..end];
        // `<h2 class="">` on the live page, `<h2>` in older captures.
        let anchor = block
            .find("<h2")
            .and_then(|h| block[h..].find("<a").map(|a| h + a));
        if let Some(a) = anchor {
            let tag_end = block[a..].find('>').map_or(block.len(), |e| a + e);
            let tag = &block[a..tag_end];
            let href = tag
                .find("href=\"")
                .and_then(|hs| {
                    let after = &tag[hs + 6..];
                    after.find('"').map(|e| after[..e].to_string())
                })
                .unwrap_or_default()
                .replace("&amp;", "&");
            let href = unwrap_bing_redirect(&href);
            let title_end = block[tag_end..]
                .find("</a>")
                .map_or(block.len(), |e| tag_end + e);
            let title = strip_tags(&block[tag_end + 1..title_end]);
            let body = block
                .find("<p")
                .and_then(|p| {
                    let text_start = block[p..].find('>').map(|e| p + e + 1)?;
                    let text_end = block[text_start..].find("</p>")? + text_start;
                    Some(strip_tags(&without_date_badge(
                        &block[text_start..text_end],
                    )))
                })
                .unwrap_or_default();
            if !title.is_empty() {
                results.push(SearchResult { title, href, body });
            }
        }
        idx = end;
    }
    results
}

/// The live page links every result through `bing.com/ck/a?…&u=a1<base64>`,
/// where the payload is the destination URL, base64url-encoded behind an
/// `a1` prefix. The aggregator follow-up fetches these hrefs, and a redirect
/// page is not a listing page — so the real URL is recovered here, and a
/// link that does not decode is passed through as it came.
fn unwrap_bing_redirect(href: &str) -> String {
    use base64::Engine as _;
    if !href.contains("bing.com/ck/a") {
        return href.to_string();
    }
    let Some(payload) = href.split('&').find_map(|part| part.strip_prefix("u=a1")) else {
        return href.to_string();
    };
    base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload.trim_end_matches('='))
        .ok()
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .filter(|url| url.starts_with("http"))
        .unwrap_or_else(|| href.to_string())
}

/// Bing prefixes some snippets with a dated badge (`<span
/// class="algoSlug_icon">Sep 1</span>`) that is not part of the sentence;
/// stripped tags would glue it onto the first word.
fn without_date_badge(fragment: &str) -> String {
    let Some(start) = fragment.find("<span class=\"algoSlug_icon\"") else {
        return fragment.to_string();
    };
    let Some(end_rel) = fragment[start..].find("</span>") else {
        return fragment.to_string();
    };
    format!(
        "{}{}",
        &fragment[..start],
        &fragment[start + end_rel + "</span>".len()..]
    )
}

/// Drop tags and unescape the handful of entities search markup carries.
fn strip_tags(fragment: &str) -> String {
    let mut out = String::with_capacity(fragment.len());
    let mut in_tag = false;
    for c in fragment.chars() {
        match c {
            '<' => in_tag = true,
            '>' if in_tag => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out.replace("&#x27;", "'")
        .replace("&#39;", "'")
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&nbsp;", " ")
        .trim()
        .to_string()
}

/// What one engine said to one query.
///
/// `Blocked` is a bot wall, `Unreachable` a transport failure, `Empty` a real
/// answer with nothing in it. The three used to collapse into an empty `Vec`,
/// which is how "no events this weekend" was reported for a month of "no
/// search this weekend".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineVerdict {
    Answered(usize),
    Blocked,
    Unreachable,
    Empty,
    /// Not consulted, because an earlier engine already answered.
    Skipped,
}

/// The engines, in the order they are tried.
pub const ENGINES: [&str; 2] = ["DuckDuckGo", "Bing"];

/// One query's results and what each engine said about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryOutcome {
    pub query: String,
    pub results: Vec<SearchResult>,
    /// Indexed like [`ENGINES`].
    pub verdicts: [EngineVerdict; 2],
}

impl QueryOutcome {
    /// True when no engine produced a result AND at least one was walled —
    /// the case where an empty corpus means "blocked", not "quiet".
    #[must_use]
    pub fn starved_by_bot_wall(&self) -> bool {
        self.results.is_empty() && self.verdicts.contains(&EngineVerdict::Blocked)
    }
}

/// Query the engines in order until one answers, recording what each said.
///
/// `DuckDuckGo` first: POST is primary (browser UA, `q`/`b`/`l` form — the
/// combination the S3 spike measured green); GET is the fallback for mocks and
/// non-DDG hosts. Then Bing, GET only. A challenged response and a transport
/// failure are both recorded and printed, never silently turned into "the
/// search found nothing".
#[must_use]
pub fn search_engines(query: &str, ddg_url: &str, bing_url: &str) -> QueryOutcome {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(8))
        .build()
        .unwrap_or_default();

    let mut verdicts = [EngineVerdict::Skipped; 2];
    let (results, ddg) = search_duckduckgo(&client, query, ddg_url);
    verdicts[0] = ddg;
    if !results.is_empty() {
        return QueryOutcome {
            query: query.to_string(),
            results,
            verdicts,
        };
    }
    let (results, bing) = search_bing(&client, query, bing_url);
    verdicts[1] = bing;
    for (engine, verdict) in ENGINES.iter().zip(verdicts) {
        match verdict {
            EngineVerdict::Blocked => {
                eprintln!("\u{26a0} {engine} blocked {query:?} (bot wall)");
            }
            EngineVerdict::Unreachable => {
                eprintln!("\u{26a0} {engine} unreachable for {query:?}");
            }
            _ => {}
        }
    }
    QueryOutcome {
        query: query.to_string(),
        results,
        verdicts,
    }
}

/// The DDG leg: POST, then GET. Challenged GET responses are expected on the
/// real endpoint — that wall is why POST is the primary.
fn search_duckduckgo(
    client: &reqwest::blocking::Client,
    query: &str,
    url: &str,
) -> (Vec<SearchResult>, EngineVerdict) {
    let mut verdict = EngineVerdict::Empty;
    let mut unreachable = false;
    // The form mirrors the public search form: q, b, l and a Safari UA.
    // `b`/`l` are the anti-bot scent the spike found present.
    match client
        .post(url)
        .form(&[("q", query), ("b", ""), ("l", "")])
        .header("User-Agent", SEARCH_UA)
        .header("Accept-Language", "en-CA,en;q=0.9")
        .send()
    {
        Ok(resp) if resp.status().is_success() => {
            if let Ok(html) = resp.text() {
                match classify(&html, parse_results_from_html(&html)) {
                    (results, EngineVerdict::Answered(_)) => return answered(results),
                    (_, v) => verdict = v,
                }
            }
        }
        Ok(_) => {}
        Err(_) => unreachable = true,
    }
    if let Ok(resp) = client
        .get(url)
        .query(&[("q", query)])
        .header("User-Agent", SEARCH_UA)
        .header("Accept-Language", "en-CA,en;q=0.9")
        .send()
    {
        if resp.status().is_success() {
            if let Ok(html) = resp.text() {
                match classify(&html, parse_results_from_html(&html)) {
                    (results, EngineVerdict::Answered(_)) => return answered(results),
                    (_, v) => verdict = v,
                }
            }
        }
    } else if unreachable {
        verdict = EngineVerdict::Unreachable;
    }
    (Vec::new(), verdict)
}

/// The Bing leg: one GET with the same browser scent.
fn search_bing(
    client: &reqwest::blocking::Client,
    query: &str,
    url: &str,
) -> (Vec<SearchResult>, EngineVerdict) {
    match client
        .get(url)
        .query(&[("q", query)])
        .header("User-Agent", SEARCH_UA)
        .header("Accept-Language", "en-CA,en;q=0.9")
        .send()
    {
        Ok(resp) if resp.status().is_success() => {
            let Ok(html) = resp.text() else {
                return (Vec::new(), EngineVerdict::Unreachable);
            };
            let results = parse_bing_results(&html);
            classify(&html, results)
        }
        Ok(_) => (Vec::new(), EngineVerdict::Empty),
        Err(_) => (Vec::new(), EngineVerdict::Unreachable),
    }
}

/// Results FIRST, wall SECOND. A page that parsed into results is an answer
/// whatever else it carries: Bing's result page lists
/// `challenges.cloudflare.com` inside a script's domain allowlist, and the
/// marker scan alone called nine answered queries "blocked" on the first
/// live run. A wall page has no results, so the parse is the stronger
/// instrument and the markers only explain an empty one.
fn classify(html: &str, results: Vec<SearchResult>) -> (Vec<SearchResult>, EngineVerdict) {
    if !results.is_empty() {
        return answered(results);
    }
    if is_challenged(html) {
        (Vec::new(), EngineVerdict::Blocked)
    } else {
        (Vec::new(), EngineVerdict::Empty)
    }
}

/// A non-empty result set, with its count stamped into the verdict.
const fn answered(results: Vec<SearchResult>) -> (Vec<SearchResult>, EngineVerdict) {
    let n = results.len();
    (results, EngineVerdict::Answered(n))
}
