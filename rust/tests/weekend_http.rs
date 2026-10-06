//! Integration tests for the weekend module's HTTP-dependent functions.
//!
//! Both endpoints are loopback stubs bound to `127.0.0.1:0`, so the real
//! transport runs — client build, request line, body decode, and the failure
//! path — and nothing here can reach the live Open-Meteo endpoint. The stub
//! RECORDS what it was asked for, so a passing test proves the bytes crossed a
//! socket rather than that a hand-fed value parsed.
//!
//! The LLM half of the pipeline is covered where its mock is built for what it
//! returns: `weekend_fetch_tests.rs` drives `fetch_duckduckgo_events` against a
//! loopback chat endpoint and asserts the parsed events field by field.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::thread;

/// A loopback stub answering up to `requests` connections with `body` as JSON,
/// recording each request line. Returns the URL and the recorder.
fn stub_server(body: &'static str, requests: usize) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let recorder = Arc::clone(&seen);
    thread::spawn(move || {
        for _ in 0..requests {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let mut buf = [0u8; 4096];
            let read = stream.read(&mut buf).unwrap_or(0);
            let head = String::from_utf8_lossy(&buf[..read]).into_owned();
            if let Some(line) = head.lines().next() {
                recorder
                    .lock()
                    .expect("recorder lock")
                    .push(line.to_string());
            }
            let http = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(http.as_bytes());
            let _ = stream.flush();
        }
    });
    (format!("http://{addr}/v1/forecast"), seen)
}

/// The exact forecast line for a day: `{date}: {temp}°C, {cond} ({precip}mm)`.
/// Rain is `precip > 0.5`, so 1.2mm is Precipitation and 0.0mm is Clear.
const METEO_BODY: &str = r#"{"daily":{"time":["2026-08-07","2026-08-08"],"temperature_2m_max":[28.2,32.0],"precipitation_sum":[0.0,1.2]}}"#;

#[test]
fn fetch_weather_over_http_returns_the_forecast_the_stub_served() {
    let (url, seen) = stub_server(METEO_BODY, 1);

    let forecast = ztools::weekend::fetch_weather_from(&url);

    // Both days, both conditions, byte for byte — the parser AND the transport
    // that delivered the bytes it parsed.
    assert_eq!(
        forecast,
        "Daily Forecast:\n\
         2026-08-07: 28.2°C, Clear (0.0mm)\n\
         2026-08-08: 32.0°C, Precipitation (1.2mm)"
    );
    let requests = seen.lock().expect("recorder lock").clone();
    assert_eq!(
        requests,
        vec!["GET /v1/forecast HTTP/1.1".to_string()],
        "the forecast came off a socket, so the endpoint must actually have been asked"
    );
}

/// The exact fallback both failure paths must produce. Pinned verbatim: the
/// plan renders this string, so "the fetch failed" is only honest if the words
/// on the page are these ones.
const FALLBACK_FORECAST: &str =
    "Daily Forecast: Friday: 24.5°C Clear, Saturday: 26.0°C Clear, Sunday: 23.0°C Clear";

#[test]
fn fetch_weather_returns_the_fixed_forecast_when_the_endpoint_is_dead() {
    // Port 1 on loopback refuses: the transport fails and the wrapper must
    // answer with the documented fallback rather than panic or return "".
    let forecast = ztools::weekend::fetch_weather_from("http://127.0.0.1:1/v1/forecast");

    assert_eq!(forecast, FALLBACK_FORECAST);
}

#[test]
fn a_reachable_endpoint_answering_zero_days_still_yields_the_fixed_forecast() {
    // A valid envelope and a 200 with an EMPTY daily block: the other failure
    // mode, reached after a successful round trip rather than instead of one.
    // The plan must not be handed an empty "Daily Forecast:" header instead.
    let (url, seen) = stub_server(
        r#"{"daily":{"time":[],"temperature_2m_max":[],"precipitation_sum":[]}}"#,
        1,
    );

    let forecast = ztools::weekend::fetch_weather_from(&url);

    assert_eq!(forecast, FALLBACK_FORECAST);
    assert_eq!(
        seen.lock().expect("recorder lock").len(),
        1,
        "the endpoint answered, so this must be the parse-fail path and not a dead one"
    );
}
