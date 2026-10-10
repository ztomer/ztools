//! A failed forecast must be VISIBLE in the saved plan, and no temperature may
//! appear that nobody measured.
//!
//! THE DEFECT, in the form it reached a reader: `fetch_weather_from` answers a
//! failed fetch with a fixed string (the fetcher's fallback, whose only line
//! starts with "Daily Forecast"); `format_weather_display` skipped every line
//! starting with "Daily Forecast"; finding nothing left, it rendered a SECOND
//! hardcoded forecast of its own. So the value the code documented as the
//! failure never reached a human, and what the planner saved on a dead endpoint
//! was "Fri 28.2°C (clear), Sat 32.0°C (precipitation), Sun 29.7°C (clear)" —
//! a reading that was invented in two places, neither of which had been
//! measured, and neither of which said so.
//!
//! WHAT EACH CASE PINS, and why it is not the same case twice:
//!
//!   * the document, end to end, from a real refused connection — the artefact
//!     a person opens, not the formatter's return value;
//!   * the anti-drift guard: the fetcher's LIVE output must equal the one
//!     constant the formatter recognises. `weekend/fetch.rs` holds a literal
//!     this test cannot see, so the constant in `format.rs` is only true by
//!     accident until something asserts the two against each other;
//!   * a real forecast, still rendered as a real forecast — the differential
//!     control. A "return the failure sentence unconditionally" fix passes
//!     every other test in this file and fails this one, which is the only thing
//!     that stops the fix being over-applied;
//!   * identity rather than shape — a genuine dry spell at the fallback's exact
//!     temperatures is still a forecast;
//!   * the scoring path, because the weather string is not only printed: it is
//!     what `apply_scores` reads for its sunny/cloudy bonus, so a failure string
//!     containing "clear" pays an outdoor row +2.0 for weather nobody has.
//!
//! Hermetic: every URL is `127.0.0.1` (a stub bound to port 0, or a port that is
//! bound-then-released so the connection is refused locally), no browser is
//! launched, no process is signalled, and the GPU lock is never reached.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

use ztools::config::ZtoolsConfig;
use ztools::weekend::{
    FORECAST_FETCH_FAILED, PlanHealth, WEATHER_FAILURE_CAUSE, WEATHER_UNAVAILABLE, WeekendEvent,
    format_weather_display, format_weekend_plan, is_forecast_failure,
};

/// A loopback port nothing is listening on: bound, then released.
///
/// `bind` then `drop` rather than a constant, because a constant is either
/// permanently wrong or eventually wrong — some other process takes the port and
/// this test quietly starts measuring whatever now owns it. Losing the race
/// makes the assertion go red naming the URL that answered, which is the failure
/// worth having.
fn closed_loopback_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// A loopback stub answering ONE `GET` with `body`, forever.
///
/// `Connection: close` so the client does not hold the socket open for a second
/// request this loop never reads.
fn stub_once(body: &'static str) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let mut stream = stream;
            let mut buf = [0u8; 8192];
            let _ = stream.read(&mut buf);
            let http = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(http.as_bytes());
            let _ = stream.flush();
        }
    });
    // Wait for the condition a client needs — a listener accepting — rather than
    // sleeping and hoping. `TcpStream::connect` is the probe: a bound listener
    // completes handshakes from its own backlog.
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(250))
        .is_err()
    {
        assert!(
            std::time::Instant::now() < deadline,
            "the stub on 127.0.0.1:{port} never accepted a connection"
        );
        thread::sleep(std::time::Duration::from_millis(5));
    }
    port
}

/// One transient row, so the document has something to degrade around.
fn event(name: &str, description: &str) -> WeekendEvent {
    WeekendEvent {
        name: name.to_string(),
        location: "Vaughan".to_string(),
        price: "$20".to_string(),
        target_ages: "6-12".to_string(),
        day: "Saturday 10am".to_string(),
        dates: "August 8".to_string(),
        description: description.to_string(),
        is_transient: true,
        score: 3.0,
        start_date: "2026-08-08".to_string(),
        end_date: "2026-08-08".to_string(),
        weather: "outdoor".to_string(),
        duration: String::new(),
    }
}

/// The saved document for a weekend whose forecast could not be fetched.
///
/// The whole path, not a stage of it: the real fetch, the real formatter, the
/// real writer. A chain proved one link at a time still ships a broken chain —
/// the fetcher can agree with the formatter and the writer can disagree with
/// both.
fn document_for(raw: &str) -> String {
    let display = format_weather_display(raw);
    format_weekend_plan(
        &[event("Union Summer", "Bounce houses in the sun.")],
        &[],
        "Vaughan",
        "6-12",
        "Aug 07 to Aug 09, 2026",
        &display,
        &PlanHealth::nominal(),
    )
}

/// Every temperature that must never appear in a document whose forecast was
/// never measured: the fetcher's fallback numbers AND the formatter's own.
///
/// Both, deliberately. The two hardcoded forecasts are the defect; pinning one
/// leaves the other as a live way to print weather nobody measured.
const FABRICATED_TEMPERATURES: &[&str] = &["24.5", "26.0", "23.0", "28.2", "32.0", "29.7", "°C"];

/// A refused connection produces a document that says the forecast is missing.
///
/// The headline, in the artefact a person reads: `**Weather:**` must name the
/// failure and the cause, and no degree reading of any kind may appear anywhere
/// in the file.
#[test]
fn a_failed_forecast_fetch_renders_a_visible_failure_in_the_saved_document() {
    let port = closed_loopback_port();
    let raw = ztools::weekend::fetch_weather_from(&format!("http://127.0.0.1:{port}/v1/forecast"));
    let doc = document_for(&raw);

    // Printed so the bytes a human gets are in the failure message of any
    // regression, not only in a passing run's captured output.
    eprintln!("---- saved document on a refused forecast fetch ----\n{doc}");

    let weather_line = doc
        .lines()
        .find(|l| l.starts_with("**Weather:**"))
        .unwrap_or_else(|| panic!("the document has no weather line:\n{doc}"));
    assert_eq!(
        weather_line,
        format!("**Weather:** {WEATHER_UNAVAILABLE}: {WEATHER_FAILURE_CAUSE}"),
        "a failed fetch must render the ONE failure value, word for word:\n{doc}"
    );
    for temperature in FABRICATED_TEMPERATURES {
        assert!(
            !doc.contains(temperature),
            "a fabricated {temperature:?} reached a document whose forecast was never \
             measured:\n{doc}"
        );
    }
}

/// The fetcher's failure value IS the constant the formatter recognises.
///
/// The anti-drift guard. `weekend/fetch.rs` returns a literal this test cannot
/// reach by import, and `format.rs` holds the copy the formatter matches on;
/// nothing else in the tree compares them, so a reworded fallback on either side
/// would leave the formatter scoring the fallback as a forecast — the original
/// defect, with the same fabricated temperatures and one more word of drift.
#[test]
fn the_failure_value_the_fetcher_returns_is_the_one_the_formatter_recognises() {
    let port = closed_loopback_port();
    for url in [
        format!("http://127.0.0.1:{port}/v1/forecast"),
        // A body that is not JSON at all, and a JSON body with no daily block,
        // are the fetcher's other two failure paths and must land on the SAME
        // value: one failure value, whichever way the fetch failed.
        format!(
            "http://127.0.0.1:{}/v1/forecast",
            stub_once("<html>502</html>")
        ),
        format!(
            "http://127.0.0.1:{}/v1/forecast",
            stub_once(r#"{"latitude":43.8}"#)
        ),
    ] {
        let raw = ztools::weekend::fetch_weather_from(&url);
        assert_eq!(
            raw, FORECAST_FETCH_FAILED,
            "{url} failed, so it must produce the one failure value"
        );
        assert!(
            is_forecast_failure(&raw),
            "{url}: the formatter must recognise its own fetcher's failure value"
        );
        assert_eq!(
            format_weather_display(&raw),
            format!("{WEATHER_UNAVAILABLE}: {WEATHER_FAILURE_CAUSE}"),
            "{url}: the failure must render as the failure, not as weather"
        );
    }
}

/// A real forecast is still rendered as a real forecast.
///
/// The differential control, and the reason the fix above is not a fix that
/// reports every weekend's weather as unavailable. A stub that answers the real
/// Open-Meteo shape must produce real temperatures and no failure marker; an
/// implementation that simply returned the failure sentence would pass every
/// other test in this file and fail here.
#[test]
fn a_real_forecast_is_still_rendered_as_a_real_forecast() {
    let body = r#"{"daily":{"time":["2026-08-07","2026-08-08","2026-08-09"],
        "temperature_2m_max":[28.2,32.0,29.7],"precipitation_sum":[0.0,1.2,0.0]}}"#;
    let port = stub_once(body);
    let raw = ztools::weekend::fetch_weather_from(&format!("http://127.0.0.1:{port}/v1/forecast"));
    assert_ne!(
        raw, FORECAST_FETCH_FAILED,
        "a served forecast must not be mistaken for the failure value: {raw:?}"
    );
    assert!(!is_forecast_failure(&raw), "{raw:?}");

    let doc = document_for(&raw);
    let weather_line = doc
        .lines()
        .find(|l| l.starts_with("**Weather:**"))
        .expect("the weather line");
    assert_eq!(
        weather_line,
        "**Weather:** Fri 28.2°C (clear), Sat 32.0°C (precipitation), Sun 29.7°C (clear)",
        "a served forecast must reach the document unchanged:\n{doc}"
    );
    assert!(
        !doc.contains(WEATHER_UNAVAILABLE),
        "a served forecast was reported as a failure:\n{doc}"
    );
}

/// Identity, not shape: the same temperatures in the parseable form are a
/// forecast.
///
/// The fallback happens to be plausible weather, so a formatter that decided
/// "unavailable" by recognising those temperatures would throw away real dry
/// spells — and would be one refactor away from doing exactly that to every
/// forecast. The sentinel is a VALUE, matched as one.
#[test]
fn a_forecast_with_the_fallbacks_own_temperatures_is_still_a_forecast() {
    let raw = "Daily Forecast:\n2026-08-07: 24.5°C, Clear (0.0mm)\n\
               2026-08-08: 26.0°C, Clear (0.0mm)\n2026-08-09: 23.0°C, Clear (0.0mm)";
    assert!(!is_forecast_failure(raw), "{raw:?}");
    let display = format_weather_display(raw);
    assert_eq!(
        display, "Fri 24.5°C (clear), Sat 26.0°C (clear), Sun 23.0°C (clear)",
        "{display}"
    );
    assert!(!display.contains(WEATHER_UNAVAILABLE), "{display}");
}

/// An empty forecast body is a failure, not a forecast.
///
/// This is the case `weekend_filter_tests.rs::an_empty_forecast_falls_back_to_a_stated_default`
/// pinned the other way round: it asserted that an empty forecast renders "Fri"
/// and "Sun", which is the fabricated-forecast defect stated as a requirement.
/// It now asserts the failure sentence. The change is in that file's assertion,
/// not in what the code does for a served forecast.
#[test]
fn an_empty_forecast_body_renders_the_failure_not_an_invented_weekend() {
    for raw in ["Daily Forecast:\n\n", "", "   \n\n"] {
        let doc = document_for(raw);
        assert!(
            doc.contains(&format!("**Weather:** {WEATHER_UNAVAILABLE}:")),
            "an empty forecast must render the failure value:\n{doc}"
        );
        for temperature in FABRICATED_TEMPERATURES {
            assert!(
                !doc.contains(temperature),
                "an empty forecast invented {temperature:?}:\n{doc}"
            );
        }
    }
}

/// The failure string cannot buy a row a weather bonus.
///
/// The weather string is not only printed: `apply_scores` reads it for the
/// sunny/cloudy bonus, and the fabricated "clear" it used to carry paid an
/// outdoor row +2.0 — so the failure was also inflating every fit score in the
/// plan. The assertion is behavioural (a score, not a substring) because a
/// keyword list in a test is a copy of the production list, which is the thing
/// that drifts.
#[test]
fn a_failed_forecast_scores_no_row_better_than_a_clear_one_does_not() {
    let sunny = "An outdoor water park on a sunny day.";
    // Through `format_weather_display`, not through the constant: the defect
    // was never the constant, it was a code path that produced a DIFFERENT
    // string than the constant, and a test that fed the constant in would have
    // stayed green while the product went back to fabricating. Calibrated: with
    // the old hardcoded fallback restored, this goes red.
    let failure_display = format_weather_display(FORECAST_FETCH_FAILED);
    let mut under_failure = vec![event("Union Summer", sunny)];
    let mut under_clear = vec![event("Union Summer", sunny)];
    let mut under_nothing = vec![event("Union Summer", sunny)];

    ztools::weekend::apply_scores(&mut under_failure, &failure_display, &[10]);
    ztools::weekend::apply_scores(&mut under_clear, "Fri 30.0°C (clear)", &[10]);
    ztools::weekend::apply_scores(&mut under_nothing, "", &[10]);

    assert!(
        under_failure[0].score < under_clear[0].score,
        "a missing forecast must not score like clear weather ({} vs {}): {failure_display}",
        under_failure[0].score,
        under_clear[0].score
    );
    ztools::assert_exact!(
        under_failure[0].score,
        under_nothing[0].score,
        "a missing forecast must read as no forecast at all, not as weather of some kind"
    );
}

/// The two failure values cannot drift apart in either direction.
///
/// Both halves of the contract in one place: the sentinel the fetcher emits is
/// recognised as a failure, and what the formatter renders for it is a value an
/// operator cannot mistake for a reading. `is_forecast_failure` is a total
/// predicate over the two public constants, so a future edit to either that
/// breaks the other half fails here rather than in a document.
#[test]
fn the_failure_contract_is_one_value_each_way() {
    assert!(is_forecast_failure(FORECAST_FETCH_FAILED));
    assert!(is_forecast_failure(&format!("  {FORECAST_FETCH_FAILED}\n")));
    assert!(!is_forecast_failure(""));
    assert!(!is_forecast_failure("Daily Forecast:"));
    assert_ne!(
        WEATHER_UNAVAILABLE, FORECAST_FETCH_FAILED,
        "the rendered failure must not BE the fabricated forecast"
    );
    assert!(
        WEATHER_FAILURE_CAUSE.contains("no forecast"),
        "the cause must say what is missing: {WEATHER_FAILURE_CAUSE:?}"
    );
}

/// The config-level fetch (`fetch_weather`) lands on the same value.
///
/// `fetch_weather` is what `weekend-plan` actually calls: it builds the URL from
/// `config.weather_url` and hands it to `fetch_weather_from`. If the seam
/// between them ever returned something else, every assertion above would hold
/// and the product would still fabricate a forecast.
#[test]
#[serial_test::serial]
fn the_config_level_fetch_against_a_dead_endpoint_is_the_same_failure() {
    // `ZtoolsConfig::default()` resolves `~/…` paths through the process-global
    // `dirs::home_dir()`, so building one outside the sandbox is a hazard the
    // house audit gate (`test_env::audit`) fails on by name — it caught exactly
    // this shape in `tests/tls_probe.rs` when the weather endpoint became
    // configurable. `#[serial]`: TestEnv holds a process-wide lock.
    let _env = ztools::test_env::TestEnv::new();
    let port = closed_loopback_port();
    let config = ZtoolsConfig {
        weather_url: format!("http://127.0.0.1:{port}"),
        ..ZtoolsConfig::default()
    };
    let raw = ztools::weekend::fetch_weather("2026-08-07", "2026-08-09", &config);
    assert_eq!(raw, FORECAST_FETCH_FAILED);
    assert_eq!(
        format_weather_display(&raw),
        format!("{WEATHER_UNAVAILABLE}: {WEATHER_FAILURE_CAUSE}")
    );
}
