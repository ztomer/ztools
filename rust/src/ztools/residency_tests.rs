//! The load, run, release seam against the recording fake server.

use std::time::Duration;

use super::fake::{Call, FakeOsaurus};
use super::*;
use crate::test_env::TestEnv;

const BODY: &str = r#"{"choices":[{"message":{"content":"ok"}}]}"#;

fn fast() -> Timing {
    Timing {
        lock_wait: Duration::from_secs(2),
        load_request: Duration::from_secs(5),
        settle: Duration::from_millis(300),
        release_grace: Duration::from_millis(300),
        release_cap: Duration::from_secs(2),
        poll_min: Duration::from_millis(5),
        poll_max: Duration::from_millis(20),
    }
}

#[test]
fn with_model_loads_first_and_ends_with_nothing_resident() {
    let _env = TestEnv::new();
    let fake = FakeOsaurus::start(BODY);
    let ran = with_model_kept(&fake.url, "m1", fast(), |held| {
        assert_eq!(held.model(), "m1");
        assert!(held.load_seconds() >= 0.0);
        Ok(7)
    })
    .unwrap();
    assert_eq!(ran.out.unwrap(), 7);
    assert!(ran.released.fault().is_none(), "{:?}", ran.released);
    assert_eq!(fake.loads(), vec!["m1".to_string()]);
    assert_eq!(fake.calls().first(), Some(&Call::Load("m1".to_string())));
    assert_eq!(fake.last_health(), Some(vec![]), "{:?}", fake.calls());
}

#[test]
fn a_failing_body_is_still_released() {
    let _env = TestEnv::new();
    let fake = FakeOsaurus::start(BODY);
    let err = with_model_kept(&fake.url, "m1", fast(), |_| -> Result<()> {
        bail!("body broke")
    })
    .unwrap()
    .out
    .unwrap_err();
    assert!(err.to_string().contains("body broke"), "{err:#}");
    assert_eq!(fake.last_health(), Some(vec![]), "{:?}", fake.calls());
}

#[test]
fn a_panicking_body_is_still_released() {
    let _env = TestEnv::new();
    let fake = FakeOsaurus::start(BODY);
    let url = fake.url.clone();
    let joined = std::thread::spawn(move || {
        let _ = with_model_kept(&url, "m1", fast(), |_| -> Result<()> { panic!("boom") });
    })
    .join();
    assert!(joined.is_err(), "the body panicked");
    assert_eq!(fake.last_health(), Some(vec![]), "{:?}", fake.calls());
}

#[test]
fn a_server_that_keeps_the_model_loaded_is_a_fault_beside_the_kept_result() {
    let _env = TestEnv::new();
    let fake = FakeOsaurus::keeping_loaded(BODY);
    let ran = with_model_kept(&fake.url, "m1", fast(), |_| Ok(7)).unwrap();
    assert_eq!(ran.out.unwrap(), 7, "the finished result is kept");
    let text = ran.released.fault_text().expect("the model outstayed it");
    assert!(text.contains("no idle unload"), "{text}");
    assert_eq!(fake.last_health(), Some(vec!["m1".to_string()]));
}

#[test]
fn a_dead_server_fails_the_load_without_running_the_body() {
    let _env = TestEnv::new();
    let mut ran = false;
    let err = with_model_kept("http://127.0.0.1:9", "m1", fast(), |_| {
        ran = true;
        Ok(())
    })
    .err()
    .expect("a dead server fails the load");
    assert!(!ran);
    assert!(format!("{err:#}").contains("loading m1"), "{err:#}");
}

#[test]
fn health_that_is_not_osaurus_is_refused() {
    assert!(parse_health(BODY).is_err());
    assert!(parse_health("not json").is_err());
}

#[test]
fn health_reads_the_measured_shape() {
    let h = parse_health(
        r#"{"batch_diagnostics":{"accepting_requests":false},"resident_models":[
            {"name":"raptor-v0.5-8b-a1b-jang_6m","is_current":true,"idle_seconds_remaining":29},
            {"name":"kept","is_current":false,"idle_seconds_remaining":null}]}"#,
    )
    .unwrap();
    assert!(!h.accepting);
    let r = h.find("OsaurusAI/Raptor-v0.5-8B-A1B-JANG_6M").unwrap();
    assert_eq!(r.idle_seconds_remaining, Some(29));
    assert_eq!(h.find("kept").unwrap().idle_seconds_remaining, None);
    assert!(h.find("absent").is_none());
    assert_eq!(names(&h), "raptor-v0.5-8b-a1b-jang_6m, kept");
    assert_eq!(
        names(&parse_health(r#"{"resident_models":[]}"#).unwrap()),
        "none"
    );
}

#[test]
fn a_model_that_never_shows_resident_fails_the_load_and_names_what_is() {
    let _env = TestEnv::new();
    // A server that always reports nothing resident: the load "answered" but
    // the model is not there, which must not pass as loaded.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        use std::io::{Read, Write};
        for mut s in listener.incoming().flatten() {
            let mut b = [0u8; 4096];
            let _ = s.read(&mut b);
            let body = if String::from_utf8_lossy(&b).starts_with("GET /health") {
                r#"{"resident_models":[]}"#
            } else {
                BODY
            };
            let _ = write!(
                s,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
        }
    });
    let err = load(&format!("http://127.0.0.1:{port}"), "m1", fast())
        .err()
        .expect("must not load");
    assert!(format!("{err:#}").contains("resident: none"), "{err:#}");
}
