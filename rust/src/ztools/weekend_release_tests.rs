//! The weekend planner's release of its model (ZT.EMBEDSEAM): a model left
//! resident after the phases fails the run, against the Osaurus fake.

use super::weekend_fetch_tests::{DEAD, ctx, sandboxed_config, window};
use crate::test_env::TestEnv;
use crate::ztools::weekend::fetch_duckduckgo_events_at;
use serial_test::serial;

/// The seam's timings with the release wait cut to milliseconds, so a model
/// the server will not unload fails the run in the test's time, not in 30 s.
fn quick_release() -> crate::ztools::residency::Timing {
    use std::time::Duration;
    crate::ztools::residency::Timing {
        release_grace: Duration::from_millis(200),
        settle: Duration::from_secs(5),
        ..crate::ztools::residency::Timing::default()
    }
}

/// A model left resident after the plan's phases is the run's FAILURE, with
/// the server's reason: the release error used to be printed and the plan
/// returned as a success, so a model kept in memory while idle went unnoticed.
#[test]
#[serial]
fn a_model_left_resident_after_the_phases_fails_the_run_with_its_reason() {
    use crate::ztools::residency::fake::FakeOsaurus;
    let env = TestEnv::new();
    let fake = FakeOsaurus::keeping_loaded(r#"{"choices":[{"message":{"content":""}}]}"#);
    let config = sandboxed_config(&env, DEAD, &fake.url, DEAD);

    let outcome = fetch_duckduckgo_events_at(
        "Vaughan",
        window(),
        "sunny",
        &ctx(),
        &config,
        quick_release(),
    );

    let err = outcome.expect_err("a model left resident must fail the run");
    let reason = format!("{err:#}");
    assert!(reason.contains("still resident"), "{reason}");
    assert!(reason.contains("Keep Model Loaded"), "{reason}");
    drop(env);
}

/// The other half: a model the server unloads leaves the run a success, so the
/// failure above is the release and not the fake.
#[test]
#[serial]
fn a_model_the_server_unloads_leaves_the_run_a_success() {
    use crate::ztools::residency::fake::FakeOsaurus;
    let env = TestEnv::new();
    let fake = FakeOsaurus::start(r#"{"choices":[{"message":{"content":""}}]}"#);
    let config = sandboxed_config(&env, DEAD, &fake.url, DEAD);

    let outcome = fetch_duckduckgo_events_at(
        "Vaughan",
        window(),
        "sunny",
        &ctx(),
        &config,
        quick_release(),
    );

    assert!(outcome.is_ok(), "{:?}", outcome.err());
    assert_eq!(fake.last_health(), Some(vec![]), "{:?}", fake.calls());
    drop(env);
}

/// ZT.TWGPULOCK, the weekend planner's entry: the roster, the warm-up load,
/// every phase's completion and the release polls are all made while the GPU
/// lock is held.
#[test]
#[serial]
fn a_weekend_run_makes_no_request_outside_the_gpu_lock() {
    use crate::ztools::residency::fake::FakeOsaurus;
    let env = TestEnv::new();
    let fake = FakeOsaurus::start(r#"{"choices":[{"message":{"content":""}}]}"#);
    let config = sandboxed_config(&env, DEAD, &fake.url, DEAD);

    let outcome = fetch_duckduckgo_events_at(
        "Vaughan",
        window(),
        "sunny",
        &ctx(),
        &config,
        quick_release(),
    );

    assert!(outcome.is_ok(), "{:?}", outcome.err());
    assert!(
        fake.calls().len() > 3,
        "the run must have reached the server: {:?}",
        fake.calls()
    );
    assert_eq!(
        fake.unlocked_requests(),
        Vec::<String>::new(),
        "{:?}",
        fake.calls()
    );
    drop(env);
}
