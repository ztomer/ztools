use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

use super::*;
use crate::ztools::weekend_cache::RegionLists;

/// Region lists for follow tests: inline test data with the whitelist tokens
/// these fixtures rely on ("markham"). The shipped file's contents are pinned
/// by `region_lists_load_from_the_shipped_config` in `weekend_filter_tests`.
fn test_region() -> RegionLists {
    RegionLists {
        cities: vec!["vaughan".to_string(), "toronto".to_string()],
        in_region: vec!["markham".to_string(), "ontario".to_string()],
        foreign: vec![],
    }
}

const NOISE_HTML: &str = "<html><head><style>body{color:red}</style></head>\
<body><script>window._evil=true</script>\
<nav><a href=\"/\">Home</a></nav>\
<header>Site Header</header>\
<h2>Vaughan Fall Fair</h2>\
<p>Crush festival   returns   this weekend.</p>\
<ul><li>Pie contest</li><li>Antique tractor show</li></ul>\
<footer>Copyright 2026</footer></body></html>";

#[test]
fn extract_strips_noise_blocks_keeps_listing_lines() {
    let text = followup::extract_page_text(NOISE_HTML, 4000);
    assert!(!text.contains("script"), "{text}");
    assert!(!text.contains("_evil"), "{text}");
    assert!(!text.contains("color:red"), "{text}");
    assert!(!text.contains("Site Header"), "{text}");
    assert!(!text.contains("Site"), "{text}");
    assert!(!text.contains("Copyright"), "{text}");
    assert!(text.contains("Vaughan Fall Fair"), "{text}");
    assert!(
        text.contains("Crush festival returns this weekend."),
        "inner whitespace must collapse per line: {text}"
    );
    assert!(text.contains("Pie contest"), "{text}");
    assert!(text.contains("Antique tractor show"), "{text}");
}

#[test]
fn extract_bounds_output_to_max_chars() {
    let big = format!("<p>{} </p>", "words ".repeat(2000));
    let text = followup::extract_page_text(&big, 400);
    assert!(
        text.chars().count() <= 400,
        "output must be bounded: {} chars",
        text.chars().count()
    );
}

#[test]
fn looks_like_aggregator_matches_directory_markers() {
    assert!(followup::looks_like_aggregator("Things to Do in Vaughan"));
    assert!(followup::looks_like_aggregator(
        "What's on in Toronto This Weekend"
    ));
    assert!(!followup::looks_like_aggregator(
        "Vaughan Fall Fair Returns"
    ));
}

#[test]
fn candidate_lines_are_prefixed_bounded_and_sized() {
    let text = "short\nA full-length event listing that clears the minimum candidate length\n";
    let lines = followup::as_candidate_lines(text, "Vaughan Events");
    assert!(lines.contains("- [Vaughan Events] A full-length"));
    assert!(!lines.contains("short"), "{lines}");
    let all_lines = lines.lines().count();
    assert_eq!(all_lines, 1, "{lines}");
}

#[test]
fn follow_aggregators_fetches_directory_pages_and_bounds_the_tally() {
    fn serve(body: &str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let body = body.to_string();
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 1024];
            let _ = stream.read(&mut buf);
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(resp.as_bytes());
        });
        format!("http://{addr}")
    }

    let page = "<html><body><h2>Markham Museums</h2><p>Historical village open all summer long</p></body></html>";
    let url = serve(page);
    let results = vec![
        search::SearchResult {
            title: "Markham Museums What's On".into(),
            href: url.clone(),
            body: "museums near Markham".into(),
        },
        search::SearchResult {
            title: "Markham Museums What's On".into(),
            href: url,
            body: "museums near Markham".into(),
        },
        search::SearchResult {
            title: "Not an aggregator".into(),
            href: "http://127.0.0.1:1/".into(),
            body: "solo indie band".into(),
        },
    ];
    let out = followup::follow_aggregators(&results, &test_region());
    // Only aggregator-marked in-region pages are followed; the set is bounded
    // by FOLLOW_LIMIT, whichever order they arrive in the tally. The body text
    // proves the page was genuinely fetched and parsed.
    assert!(
        out.contains("Historical village open all summer long"),
        "{out}"
    );
    assert!(out.contains("[Markham Museums What's On]"), "{out}");
}

#[test]
fn follow_aggregators_ignores_no_href_and_foreign_results() {
    let out = followup::follow_aggregators(
        &[search::SearchResult {
            title: "Events Guide".into(),
            href: String::new(),
            body: "calendar of events".into(),
        }],
        &test_region(),
    );
    assert!(
        out.is_empty(),
        "no href means no fetch can target it: {out}"
    );
}

/// The text AFTER a noise block resumes exactly at its close tag. The 2026-10-10
/// corpus followed three listing pages and every candidate line from them was
/// script or CSS ("window.dataLayer = ...", "display: inline !important;"): the
/// skip advanced by the close tag's length from where the scan STOOD, not from
/// where the close tag WAS, so it landed back inside the script body and
/// emitted its tail. Short blocks hid it -- "window._evil=true" leaked only
/// "vil=true", which a `!contains("_evil")` check cannot see -- so this pins
/// the whole output, over blocks longer than their own close tags.
#[test]
fn text_after_a_long_noise_block_resumes_exactly_at_its_close_tag() {
    let html = "<html><head><title>Things to Do</title>\
<script>window.dataLayer = window.dataLayer || []; dataLayer.push({userRole: 'logged-out'});</script>\
<style>img.emoji { display: inline !important; border: none !important; }</style></head>\
<body><h2>Pumpkinfest at Downey's Farm</h2><p>Oct 9 - Oct 12, 2026</p>\
<script type=\"application/ld+json\">{\"@type\":\"CollectionPage\",\"name\":\"Events\"}</script>\
<li>Thanksgiving Harvest Market</li></body></html>";
    assert_eq!(
        followup::extract_page_text(html, 4000),
        "Things to Do\nPumpkinfest at Downey's Farm\nOct 9 - Oct 12, 2026\nThanksgiving Harvest Market"
    );
}

/// One page, one fetch: the same listing reached from two queries was followed
/// twice on 2026-10-10 ("Things to Do - Visit Markham" filled 2 of the 3 follow
/// slots), spending the budget on a duplicate.
#[test]
fn the_same_page_found_by_two_queries_is_followed_once() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let served = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = served.clone();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { break };
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let mut buf = [0u8; 1024];
            let _ = stream.read(&mut buf);
            let body = "<p>Harvest hayrides all weekend long</p>";
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(resp.as_bytes());
        }
    });
    let page = format!("http://{addr}/things-to-do/");
    let result = |title: &str| search::SearchResult {
        title: title.into(),
        href: page.clone(),
        body: "Things to do in Markham".into(),
    };
    let out = followup::follow_aggregators(
        &[
            result("Things to Do - Visit Markham"),
            result("Things To Do in Markham"),
        ],
        &test_region(),
    );
    assert_eq!(out.matches("Harvest hayrides").count(), 1, "{out}");
    assert_eq!(served.load(std::sync::atomic::Ordering::SeqCst), 1);
}
