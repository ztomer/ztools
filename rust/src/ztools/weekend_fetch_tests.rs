//! Search-corpus and fallback-pipeline tests (weekend/mod.rs + fetch.rs).
//!
//! Both external dependencies are replaced by loopback mock servers, so these
//! prove the corpus-building and monolithic-fallback paths without ever
//! touching DuckDuckGo or a real model endpoint. The snippet parser is pure
//! string work and is fed synthetic HTML directly.

use super::*;

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

fn ctx() -> PlanContext {
    PlanContext {
        location: "Vaughan".into(),
        ages: "6-12".into(),
        date_range: "Aug 7 to Aug 9".into(),
        year: 2026,
        exclusions: "none".into(),
    }
}

fn window() -> (chrono::NaiveDate, chrono::NaiveDate) {
    (
        chrono::NaiveDate::parse_from_str("2026-08-07", "%Y-%m-%d").unwrap(),
        chrono::NaiveDate::parse_from_str("2026-08-09", "%Y-%m-%d").unwrap(),
    )
}

/// Read one HTTP request completely: keep reading until the declared
/// Content-Length body has arrived, so prompt-content checks cannot flake on
/// TCP segmentation.
fn read_request(stream: &mut TcpStream) -> String {
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(5)))
        .ok();
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
        if let Some(header_end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&buf[..header_end]).to_lowercase();
            let len: usize = head
                .lines()
                .find_map(|l| {
                    l.strip_prefix("content-length:")
                        .and_then(|v| v.trim().parse().ok())
                })
                .unwrap_or(0);
            if buf.len() >= header_end + 4 + len {
                break;
            }
        }
    }
    String::from_utf8_lossy(&buf).into_owned()
}

/// Loopback server answering every request with `body` as HTML.
fn serve_html(body: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        while let Ok((mut stream, _)) = listener.accept() {
            let _request = read_request(&mut stream);
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(resp.as_bytes());
        }
    });
    format!("http://{addr}")
}

/// Loopback chat-completions mock. The monolithic extraction prompt is the
/// only one carrying "STRICTLY this weekend"; every other phase gets an empty
/// content so `call_llm_text` filters it out and the draft phase fails.
fn serve_chat(monolithic_content: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        while let Ok((mut stream, _)) = listener.accept() {
            let request = read_request(&mut stream);
            let content = if request.contains("STRICTLY this weekend") {
                monolithic_content
            } else {
                ""
            };
            let body = format!(
                r#"{{"choices":[{{"message":{{"content":"{}"}}}}]}}"#,
                content.replace('\\', "\\\\").replace('"', "\\\"")
            );
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(resp.as_bytes());
        }
    });
    format!("http://{addr}")
}

const SNIPPETS_HTML: &str = "<html><body>\
<div><a class=\"result__snippet\">Vaughan Fall Fair returns this weekend</a></div>\
<div><a class=\"result__snippet\">Vaughan Fall Fair returns this weekend</a></div>\
<div><a class=\"result__snippet\">Aspen ski school opens for the season</a></div>\
</body></html>";

#[test]
fn corpus_building_dedupes_keeps_only_region_backed_snippets_and_counts_them() {
    let ddg = serve_html(SNIPPETS_HTML);
    let config = crate::config::ZtoolsConfig {
        duckduckgo_url: ddg,
        osaurus_url: "http://127.0.0.1:1".into(),
        llm_timeout_secs: 1,
        ..crate::config::ZtoolsConfig::default()
    };

    // Every one of the fan-out queries hits the same mock, so the duplicate
    // snippet arrives many times over; dedup must collapse it to one line,
    // the Aspen result must be dropped for lacking region evidence, and the
    // candidate counter must see exactly one line (the operator println path).
    let (events, corpus) =
        fetch_duckduckgo_events("Vaughan", window().0, window().1, "sunny", &ctx(), &config);

    assert_eq!(
        corpus.matches("- Event:").count(),
        1,
        "duplicate snippets must collapse to one candidate: {corpus}"
    );
    assert!(corpus.contains("Vaughan Fall Fair"), "{corpus}");
    assert!(
        !corpus.contains("Aspen"),
        "a snippet without region evidence must not enter the corpus: {corpus}"
    );
    assert!(
        events.is_empty(),
        "with the model dead the pipeline must yield no invented events"
    );
}

#[test]
fn a_dead_draft_phase_falls_back_to_the_monolithic_prompt_and_parses_real_events() {
    let monolithic_json = r#"{"transient_events":[{"name":"Maple Syrup Festival","location":"Vaughan","target_ages":"All ages","price":"By donation","start_date":"2026-08-08","end_date":"2026-08-08","day":"Saturday","weather":"Clear","duration":"All day","description":"Sap to syrup demos"}]}"#;
    let osaurus = serve_chat(monolithic_json);
    let config = crate::config::ZtoolsConfig {
        duckduckgo_url: "http://127.0.0.1:1/".into(),
        osaurus_url: osaurus,
        llm_timeout_secs: 1,
        ..crate::config::ZtoolsConfig::default()
    };

    let (events, corpus) =
        fetch_duckduckgo_events("Vaughan", window().0, window().1, "sunny", &ctx(), &config);

    assert!(corpus.is_empty(), "search is dead, so no corpus: {corpus}");
    assert_eq!(events.len(), 1, "{events:?}");
    assert_eq!(events[0].name, "Maple Syrup Festival");
    assert_eq!(events[0].price, "By donation");
    assert_eq!(events[0].day, "Saturday");
    assert!(events[0].is_transient);
}

// --- parse_snippets_from_html: pure parsing of synthetic DDG markup ---

#[test]
fn snippet_parser_extracts_standard_and_lite_markup_and_cleans_entities() {
    let standard = r#"<td class="result__snippet"><b>Kleinburg</b> Maple &#x27;syrup&#x27; &amp; &quot;pancakes&quot;</a>"#;
    assert_eq!(
        parse_snippets_from_html(standard),
        vec!["Kleinburg Maple 'syrup' & \"pancakes\""]
    );

    let lite = r#"<tr><td class="result-snippet">Lite &amp; plain</td></tr>"#;
    assert_eq!(parse_snippets_from_html(lite), vec!["Lite & plain"]);
}

#[test]
fn when_both_snippet_patterns_exist_the_earlier_match_wins_each_round() {
    let lite_first = r#"<div><td class="result-snippet">lite one</td></div><div><a class="result__snippet">standard two</a></div>"#;
    assert_eq!(
        parse_snippets_from_html(lite_first),
        vec!["lite one", "standard two"]
    );

    let standard_first = r#"<div><a class="result__snippet">standard one</a></div><div><td class="result-snippet">lite two</td></div>"#;
    assert_eq!(
        parse_snippets_from_html(standard_first),
        vec!["standard one", "lite two"]
    );
}

#[test]
fn a_snippet_terminated_by_either_tag_ends_at_the_first_closer() {
    let both = r#"<td class="result__snippet">A</a>MIDDLE</td>"#;
    assert_eq!(parse_snippets_from_html(both), vec!["A"]);
}

#[test]
fn malformed_snippets_are_skipped_and_scanning_still_advances() {
    // Class attribute never closed by '>': advance past the pattern.
    let unclosed_attr = r#"<p class="result__snippet trailing"#;
    assert!(parse_snippets_from_html(unclosed_attr).is_empty());

    // Text with no </a>/</td> terminator: not extractable, scan continues.
    let unterminated = r#"<td class="result__snippet">dangling text with no closer at all"#;
    assert!(parse_snippets_from_html(unterminated).is_empty());

    // Whitespace-only snippet is dropped but a later good one still lands.
    let mixed = r#"<td class="result__snippet">   </td><td class="result__snippet">Real</td>"#;
    assert_eq!(parse_snippets_from_html(mixed), vec!["Real"]);

    // No pattern anywhere at all.
    assert!(parse_snippets_from_html("<html><body>nothing here</body></html>").is_empty());
}

#[test]
fn test_build_search_queries_derives_month_from_target_friday() {
    let sep_friday = chrono::NaiveDate::from_ymd_opt(2026, 9, 4).unwrap();
    let queries = build_search_queries(sep_friday);
    assert!(!queries.is_empty());
    assert!(queries
        .iter()
        .any(|q| q.contains("September") && q.contains("2026")));
    assert!(queries
        .iter()
        .any(|q| q.contains("harvest festival farm pumpkin")));
    assert!(!queries.iter().any(|q| q.contains("August")));

    let jan_friday = chrono::NaiveDate::from_ymd_opt(2027, 1, 1).unwrap();
    let jan_queries = build_search_queries(jan_friday);
    assert!(jan_queries
        .iter()
        .any(|q| q.contains("January") && q.contains("2027")));
    assert!(jan_queries
        .iter()
        .any(|q| q.contains("winter festival holiday lights")));
}

#[test]
fn test_is_challenged_detects_captcha_and_waf_markers() {
    let ddg_anomaly = r#"<html><body><form id="challenge-form" action="/anomaly.js"><div class="anomaly-modal__modal"></div></form></body></html>"#;
    assert!(is_challenged(ddg_anomaly));

    let cloudflare_turnstile = r#"<html><body><iframe src="https://challenges.cloudflare.com/turnstile/v0/api.js"></iframe></body></html>"#;
    assert!(is_challenged(cloudflare_turnstile));

    let human_verify =
        r#"<html><title>Please Verify You Are Human</title><body>Just a moment...</body></html>"#;
    assert!(is_challenged(human_verify));

    let normal_results =
        r#"<html><body><a class="result__snippet">Normal search snippet</a></body></html>"#;
    assert!(!is_challenged(normal_results));
}

#[test]
fn test_degradation_banner_rendered_when_transient_empty() {
    let fixed = vec![WeekendEvent {
        name: "Test Park".into(),
        location: "Vaughan".into(),
        price: "Free".into(),
        target_ages: "All Ages".into(),
        day: "Sat-Sun".into(),
        dates: "Year-Round".into(),
        description: "Outdoor park".into(),
        is_transient: false,
        score: 4.0,
        start_date: "".into(),
        end_date: "".into(),
        weather: "outdoor".into(),
        duration: "".into(),
    }];
    let empty_transient: Vec<WeekendEvent> = Vec::new();

    let gorgeous =
        render_weekend_plan_gorgeous("Sep 04 to Sep 06", "Sunny", &fixed, &empty_transient);
    assert!(gorgeous.contains("⚠ Transient Events: None found"));

    let markdown = format_weekend_plan(
        &empty_transient,
        &fixed,
        "Vaughan",
        "6-12",
        "Sep 04 to Sep 06",
        "Sunny",
    );
    assert!(markdown.contains("> [!WARNING]"));
    assert!(markdown.contains("Plan Degraded"));
}
