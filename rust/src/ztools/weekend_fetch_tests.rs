//! Search-corpus and fallback-pipeline tests (weekend/mod.rs + fetch.rs).
//!
//! Both external dependencies are replaced by loopback mock servers, so these
//! prove the corpus-building and monolithic-fallback paths without ever
//! touching `DuckDuckGo` or a real model endpoint. The snippet parser is pure
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

/// A throwaway record file per test process: the fetch appends this run's
/// engine walls to it, and a unit test must not write the operator's.
fn record_path() -> String {
    std::env::temp_dir()
        .join(format!("ztools-search-record-{}.json", std::process::id()))
        .to_string_lossy()
        .into_owned()
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
<div><a class=\"result__a\" href=\"https://example.com/fall-fair\">Vaughan Fall Fair</a><span class=\"result__snippet\">Vaughan Fall Fair returns this weekend</span></div>\
<div><a class=\"result__a\" href=\"https://example.com/fall-fair-2\">Vaughan Fall Fair</a><span class=\"result__snippet\">Vaughan Fall Fair weekend guide and family fun</span></div>\
<div><a class=\"result__a\" href=\"https://example.com/aspen\">Aspen Ski School</a><span class=\"result__snippet\">Aspen ski school opens for the season</span></div>\
</body></html>";

#[test]
fn corpus_building_dedupes_keeps_only_region_backed_snippets_and_counts_them() {
    let ddg = serve_html(SNIPPETS_HTML);
    let config = crate::config::ZtoolsConfig {
        duckduckgo_url: ddg,
        bing_url: "http://127.0.0.1:1/".into(),
        brave_url: "http://127.0.0.1:1/".into(),
        search_record_path: record_path(),
        osaurus_url: "http://127.0.0.1:1".into(),
        llm_timeout_secs: 1,
        llm_warmup_timeout_secs: 1,
        ..crate::config::ZtoolsConfig::default()
    };

    // Every one of the fan-out queries hits the same mock, so the duplicates
    // arrive many times over; dedup is on the TITLE alone (the Python
    // `_clean_search_results` key), so the two different-bodied Fall Fair
    // results must still collapse to one line, the Aspen result must be
    // dropped for lacking region evidence, and the candidate counter must see
    // exactly one line (the operator println path).
    let (events, corpus, _health) =
        fetch_duckduckgo_events("Vaughan", window().0, window().1, "sunny", &ctx(), &config);

    assert_eq!(
        corpus.matches("- Vaughan Fall Fair:").count(),
        1,
        "duplicate-titled snippets must collapse to one candidate: {corpus}"
    );
    assert!(
        !corpus.contains("- Event:"),
        "an empty title is dropped outright, never labeled Event (Python parity): {corpus}"
    );
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
        bing_url: "http://127.0.0.1:1/".into(),
        brave_url: "http://127.0.0.1:1/".into(),
        search_record_path: record_path(),
        osaurus_url: osaurus,
        llm_timeout_secs: 1,
        ..crate::config::ZtoolsConfig::default()
    };

    let (events, corpus, _health) =
        fetch_duckduckgo_events("Vaughan", window().0, window().1, "sunny", &ctx(), &config);

    assert!(corpus.is_empty(), "search is dead, so no corpus: {corpus}");
    assert_eq!(events.len(), 1, "{events:?}");
    assert_eq!(events[0].name, "Maple Syrup Festival");
    assert_eq!(events[0].price, "By donation");
    assert_eq!(events[0].day, "Saturday");
    assert!(events[0].is_transient);
}

/// The fan-out runs and yields nothing when neither the search nor the model
/// can be reached. Both endpoints point at a closed port so the outcome does
/// not depend on `DuckDuckGo` being up -- this test used to hit the live site
/// thirteen times, and whether it answered moved the coverage number.
#[test]
fn test_fetch_duckduckgo_events() {
    let config = crate::config::ZtoolsConfig {
        duckduckgo_url: "http://127.0.0.1:1/".into(),
        bing_url: "http://127.0.0.1:1/".into(),
        brave_url: "http://127.0.0.1:1/".into(),
        search_record_path: record_path(),
        osaurus_url: "http://127.0.0.1:1".into(),
        llm_timeout_secs: 1,
        llm_warmup_timeout_secs: 1,
        ..crate::config::ZtoolsConfig::default()
    };
    let (events, corpus, health) =
        fetch_duckduckgo_events("Vaughan", window().0, window().1, "sunny", &ctx(), &config);
    assert!(
        events.is_empty(),
        "unreachable search and model must yield nothing, not invented events: {events:?}"
    );
    assert!(corpus.is_empty());
    assert!(
        !health.model.is_ready(),
        "a model that never answered the warm-up must be recorded as unavailable: {health:?}"
    );
    let n = health.search.queries;
    assert!(n > 0);
    assert_eq!(
        health.search.unreachable,
        [n, n, n],
        "every query must record every engine unreachable: {:?}",
        health.search
    );
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
        r"<html><title>Please Verify You Are Human</title><body>Just a moment...</body></html>";
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
        start_date: String::new(),
        end_date: String::new(),
        weather: "outdoor".into(),
        duration: String::new(),
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
        &PlanHealth::nominal(),
    );
    assert!(markdown.contains("> [!WARNING]"));
    assert!(markdown.contains("Plan Degraded"));
}

/// A plan WITH events still says when the search behind it was partly
/// walled — and the status page reads the same phrase off either shape.
#[test]
fn a_partly_walled_search_is_noted_under_a_populated_table() {
    let mut search = SearchHealth::default();
    search.record(&crate::ztools::weekend::QueryOutcome {
        query: "q".into(),
        results: Vec::new(),
        verdicts: [
            crate::ztools::weekend::EngineVerdict::Blocked,
            crate::ztools::weekend::EngineVerdict::Blocked,
            crate::ztools::weekend::EngineVerdict::Blocked,
        ],
    });
    let health = PlanHealth {
        search,
        model: ModelHealth::Ready {
            model: "m".into(),
            secs: 1,
        },
        provenance: Provenance::default(),
    };
    let event = WeekendEvent {
        name: "Fall Fair".into(),
        location: "Vaughan".into(),
        price: "$12".into(),
        target_ages: "all".into(),
        day: "Saturday".into(),
        dates: "2026-08-08".into(),
        description: "rides".into(),
        is_transient: true,
        score: 3.0,
        start_date: "2026-08-08".into(),
        end_date: "2026-08-08".into(),
        weather: "outdoor".into(),
        duration: String::new(),
    };
    let markdown = format_weekend_plan(
        &[event],
        &[],
        "Vaughan",
        "6-12",
        "Aug 07-09",
        "Sunny",
        &health,
    );
    assert!(markdown.contains("**Fall Fair**"), "{markdown}");
    assert!(
        markdown
            .contains("> 1 of 1 searches were blocked by a bot wall; this list may be incomplete."),
        "{markdown}"
    );
    assert!(
        !markdown.contains("Plan Degraded"),
        "events were found: {markdown}"
    );
}
