//! The Bing and Brave result-page parsers, split from `search.rs` for the cap.
//!
//! Both read live markup captured 2026-09-19 and are pinned on it in
//! `weekend_search_tests.rs`; the `DuckDuckGo` parser stays in `search.rs`
//! beside the snippet parser it shares with the aggregator follow-up.

use super::search::SearchResult;

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

/// Parse Brave's web results (2026-09-19 markup).
///
/// One `data-type="web"` block per hit; the first `<a href>` is the
/// destination, the title is the `search-snippet-title` div's `title`
/// attribute, and the body is the first `content` or `description` div after
/// it. Class tokens are matched by their stable stems — the page is
/// Svelte-compiled and every class carries a hash suffix that changes per
/// build.
#[must_use]
pub fn parse_brave_results(html: &str) -> Vec<SearchResult> {
    let mut results = Vec::new();
    let mut idx = 0;
    while let Some(rel) = html[idx..].find("data-type=\"web\"") {
        let start = idx + rel;
        let end = html[start + 1..]
            .find("data-type=\"web\"")
            .map_or(html.len(), |e| start + 1 + e);
        let block = &html[start..end];
        let href = block
            .find("href=\"")
            .and_then(|h| {
                let after = &block[h + 6..];
                after.find('"').map(|e| after[..e].to_string())
            })
            .unwrap_or_default()
            .replace("&amp;", "&");
        let title = block.find("search-snippet-title").and_then(|t| {
            let tag = &block[t..];
            let a = tag.find("title=\"")?;
            let after = &tag[a + 7..];
            let e = after.find('"')?;
            Some(strip_tags(&after[..e]))
        });
        let body = block
            .find("class=\"content ")
            .or_else(|| block.find("class=\"description "))
            .and_then(|c| {
                let text_start = block[c..].find('>')? + c + 1;
                let text_end = block[text_start..].find("</div>")? + text_start;
                Some(strip_tags(&block[text_start..text_end]))
            })
            .unwrap_or_default();
        if let Some(title) = title.filter(|t| !t.is_empty()) {
            if href.starts_with("http") {
                results.push(SearchResult { title, href, body });
            }
        }
        idx = end;
    }
    results
}
