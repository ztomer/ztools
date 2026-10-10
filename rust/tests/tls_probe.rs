//! The only tests in this crate that put a certificate on the wire.
//!
//! WHY: every other HTTP test binds `127.0.0.1:0` and speaks PLAINTEXT, so the
//! suite exercises reqwest's request/response path and never its certificate
//! path. That is exactly why the `webpki-roots` feature removal was
//! unprovable: reqwest 0.13.5 deleted that feature and takes its root store
//! from `rustls-platform-verifier` (on this machine, the macOS keychain), and a
//! fully green suite said nothing about whether TLS still worked at all.
//!
//! Two halves, and the second is what makes the first trustworthy:
//!
//! - `a_failed_tls_handshake_is_not_mistaken_for_a_forecast` runs by default,
//!   offline, against a loopback stub that answers PLAINTEXT behind an
//!   `https://` URL. It proves two things the live probe cannot show by
//!   passing: that the assertions below really can go red, and that a
//!   connection which looks successful to the caller here had genuinely
//!   started a TLS handshake (the recorded bytes are a `ClientHello`) before it
//!   failed at one.
//! - `live_forecast_over_https_returns_the_weekend_the_planner_asked_for` is
//!   OPT-IN, because the gate must stay runnable with no network. Set
//!   `ZTOOLS_TLS_PROBE=1` to run it; set `ZTOOLS_TLS_PROBE_URL` as well to aim
//!   the probe at an endpoint that is SUPPOSED to fail (calibration), which
//!   then makes one claim only: it must not yield a forecast.
//!
//! SKIPS ARE LOUD. The skip line is written to the stderr HANDLE, not through
//! `eprintln!`, because libtest captures the macro and a default run would
//! otherwise print `ok` for a test that asserted nothing — which is exactly
//! how a blind spot stays invisible.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use chrono::Local;

/// Opt in: `ZTOOLS_TLS_PROBE=1`. Anything else leaves the machine alone.
const PROBE_ENV: &str = "ZTOOLS_TLS_PROBE";
/// Calibration override: an endpoint chosen because it must fail.
const URL_ENV: &str = "ZTOOLS_TLS_PROBE_URL";

/// The exact string `fetch_weather_from` answers when the round trip fails or
/// the body does not parse — pinned verbatim, and shared with `weekend_http.rs`.
///
/// It is the whole reason the probe cannot settle for "did not error": a dead
/// endpoint, a refused certificate, a captive portal and an HTML error page
/// ALL produce this string, so an error-only assertion would pass on every one
/// of them.
const FALLBACK_FORECAST: &str =
    "Daily Forecast: Friday: 24.5°C Clear, Saturday: 26.0°C Clear, Sunday: 23.0°C Clear";

/// Announce a skip no reader can mistake for a pass.
///
/// `eprintln!` alone is not enough — libtest captures it, so the default run
/// prints `ok` for a test that asserted nothing. Writing the stderr HANDLE
/// bypasses the capture, so the line is visible in a plain `cargo test`.
fn skip(reason: &str) {
    let _ = writeln!(std::io::stderr(), "SKIP: {reason}");
}

/// Serve PLAINTEXT to whatever connects, recording the first two bytes of each
/// connection until one of them opens a TLS handshake.
///
/// A TLS `ClientHello` begins with record type `0x16` (handshake) and version
/// `0x03`, so those two bytes are what separates "the client tried to
/// negotiate TLS and the negotiation failed" from "the client never got that
/// far" — a distinction the fallback string cannot express.
///
/// The accept loop is non-blocking with a deadline, not a bare blocking
/// `accept`: a thread parked in `accept` would leave the join below hanging
/// forever whenever the client declined to connect at all. The deadline sits
/// past the production client's own 5s timeout, so any connection it is ever
/// going to make has been made by then.
///
/// The ACCEPTED socket is put back to blocking explicitly. On the BSDs (macOS
/// included) an accepted socket inherits the listener's `O_NONBLOCK`, unlike
/// Linux — so without this the first `read` returns `WouldBlock` instantly,
/// records zero bytes, and the handshake assertion below fails for a reason
/// that has nothing to do with TLS.
/// The first bytes of every connection the stub accepted, one entry each.
type FirstBytes = Arc<Mutex<Vec<Vec<u8>>>>;

fn plaintext_stub() -> (String, FirstBytes, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    listener.set_nonblocking(true).unwrap();
    let seen: Arc<Mutex<Vec<Vec<u8>>>> = Arc::new(Mutex::new(Vec::new()));
    let recorder = Arc::clone(&seen);
    let handle = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(7);
        while Instant::now() < deadline {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    let _ = stream.set_nonblocking(false);
                    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
                    let mut buf = [0u8; 2];
                    let mut got = 0;
                    while got < 2 {
                        match stream.read(&mut buf[got..]) {
                            Ok(0) | Err(_) => break,
                            Ok(n) => got += n,
                        }
                    }
                    recorder
                        .lock()
                        .expect("recorder lock")
                        .push(buf[..got].to_vec());
                    // Plaintext where a ServerHello must be: the handshake
                    // cannot continue past this.
                    let _ =
                        stream.write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n");
                    let _ = stream.flush();
                    // One handshake is the whole claim; waiting for a second
                    // connection would only burn the deadline.
                    if recorder.lock().expect("recorder lock")[0].len() == 2 {
                        break;
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(10));
                }
                Err(_) => return,
            }
        }
    });
    (format!("https://{addr}/v1/forecast"), seen, handle)
}

/// The calibration, run on every default `cargo test`: an `https://` URL whose
/// server speaks plaintext cannot complete a handshake, so the production
/// client must render the fixed forecast and say so — and the bytes it sent
/// must prove the failure happened AT the handshake.
#[test]
fn a_failed_tls_handshake_is_not_mistaken_for_a_forecast() {
    let (url, seen, stub) = plaintext_stub();

    let forecast = ztools::weekend::fetch_weather_from(&url);
    stub.join().expect("stub thread");

    assert_eq!(
        forecast, FALLBACK_FORECAST,
        "a plaintext answer behind an https:// URL must render the fixed forecast, \
         never anything that could be read as data"
    );
    let hellos = seen.lock().expect("recorder lock").clone();
    assert!(
        hellos.iter().any(|b| b.as_slice() == [0x16, 0x03]),
        "no connection began with a TLS ClientHello (record type 0x16, version \
         0x03), so the fallback above was produced by something else — a refused \
         connection or a DNS miss — and this test has stopped proving anything \
         about TLS. Recorded: {hellos:?}"
    );
}

/// ONE real HTTPS request through the production client, against the endpoint
/// the weekend planner depends on, asserted on the parsed forecast rather than
/// on the absence of an error.
#[test]
fn live_forecast_over_https_returns_the_weekend_the_planner_asked_for() {
    if !std::env::var(PROBE_ENV).is_ok_and(|v| v == "1") {
        skip(&format!(
            "{PROBE_ENV} is not 1, so nothing left this machine. \
             Re-run with {PROBE_ENV}=1 cargo test --test tls_probe -- --nocapture"
        ));
        return;
    }
    // `Some(url)` is calibration: an endpoint picked because it must fail, and
    // then the shape assertions below would be meaningless against it.
    let calibration = std::env::var(URL_ENV).ok().filter(|u| !u.trim().is_empty());

    // The window `weekend_plan` itself asks for, from the same helper it uses.
    // Hardcoding dates here would rot into a false failure the day they aged
    // out of the endpoint's forecast horizon, or quietly probe a range the
    // product never requests.
    let shipped = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crate lives in <repo>/rust")
        .join("conf/weekend.toml");
    let province =
        ztools::weekend::holidays::load_province(&[shipped.to_string_lossy().into_owned()])
            .expect("the shipped weekend config names a supported province");
    let (friday, sunday) =
        ztools::weekend::plan_window(Local::now().naive_local().date(), province);
    let dates = [friday, friday + chrono::Duration::days(1), sunday];
    let iso = dates.map(|d| d.format("%Y-%m-%d").to_string());

    let calibrated = calibration.is_some();
    let url_for_print = calibration
        .clone()
        .unwrap_or_else(|| format!("api.open-meteo.com {iso:?}"));
    // `ZtoolsConfig::default()` names `~/...` paths, so building one without the
    // sandbox is a hazard: the house audit gate in `test_env::audit.rs` fails on
    // exactly this shape, and it caught this line when the weather endpoint
    // became configurable. It matters more here than elsewhere — this is the one
    // test that deliberately leaves the machine.
    let _env = ztools::test_env::TestEnv::new();
    let config = ztools::config::ZtoolsConfig::default();
    let forecast = calibration.as_deref().map_or_else(
        || ztools::weekend::fetch_weather(&iso[0], &iso[2], &config),
        ztools::weekend::fetch_weather_from,
    );

    // Printed so a human running with `--nocapture` sees the bytes the planner
    // would have rendered, not merely that the assertions held.
    println!("→ {url_for_print} returned:\n{forecast}");

    assert_ne!(
        forecast, FALLBACK_FORECAST,
        "the production client fell back to the fixed forecast, so the request \
         produced NO forecast: a refused certificate, a captive portal and an \
         HTML error page all land here. The error class is on stderr, named by \
         `fetch_weather_from`."
    );
    if calibrated {
        // Calibration mode reaches the one claim it makes. An endpoint that
        // WAS supposed to fail just did not, so stop here rather than grading
        // its body against the live endpoint's shape.
        return;
    }

    let mut lines = forecast.lines();
    assert_eq!(
        lines.next(),
        Some("Daily Forecast:"),
        "expected the parsed-forecast header, got {forecast:?}"
    );
    let days: Vec<&str> = lines.collect();
    assert_eq!(
        days.len(),
        iso.len(),
        "a Friday-to-Sunday window is three days of daily data; anything else \
         means the body was not the forecast this request asked for: {forecast:?}"
    );

    for (line, date) in days.iter().zip(&iso) {
        // The date is the strongest thing here: it is the window this request
        // put on the wire, so it cannot be satisfied by a cached page, a
        // different endpoint's answer, or an error document.
        let rest = line.strip_prefix(&format!("{date}: ")).unwrap_or_else(|| {
            panic!("forecast line does not carry the requested date {date}: {line:?}")
        });
        let (temp, rest) = rest
            .split_once("\u{b0}C, ")
            .unwrap_or_else(|| panic!("no temperature in {line:?}"));
        let (cond, precip) = rest
            .split_once(" (")
            .unwrap_or_else(|| panic!("no precipitation in {line:?}"));
        let precip = precip
            .strip_suffix("mm)")
            .unwrap_or_else(|| panic!("no precipitation unit in {line:?}"));
        let temp: f64 = temp
            .parse()
            .unwrap_or_else(|_| panic!("temperature {temp:?} is not a number in {line:?}"));
        let precip: f64 = precip
            .parse()
            .unwrap_or_else(|_| panic!("precipitation {precip:?} is not a number in {line:?}"));
        assert!(
            (-90.0..=60.0).contains(&temp),
            "not a Celsius reading: {line:?}"
        );
        assert!(precip >= 0.0, "negative precipitation: {line:?}");
        assert!(
            matches!(cond, "Clear" | "Precipitation"),
            "unknown condition, so this line was not built by the parser: {line:?}"
        );
    }
}
