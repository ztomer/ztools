//! T2: the summary's decoding is the shipped `SUMMARY_SAMPLING` unless one run
//! asks for another through `TWITTER_TEMPERATURE` -- the seam the loop-rate
//! measurement in `docs/MODEL_QUIRKS.md` runs through.

use serial_test::serial;

use super::super::{SUMMARY_SAMPLING, TEMPERATURE_ENV, summary_sampling};
use crate::test_env::TestEnv;

#[test]
#[serial]
fn one_run_can_ask_for_another_temperature() {
    let env = TestEnv::new();
    assert_eq!(summary_sampling(), SUMMARY_SAMPLING);
    env.set(TEMPERATURE_ENV, "0");
    assert!(summary_sampling().temperature.abs() < f64::EPSILON);
    assert_eq!(summary_sampling().frequency_penalty, None);
    env.set(TEMPERATURE_ENV, "0.7");
    assert!((summary_sampling().temperature - 0.7).abs() < f64::EPSILON);
    // Not a temperature, or not one a server takes: the shipped policy.
    for bad in ["warm", "-1", "3", ""] {
        env.set(TEMPERATURE_ENV, bad);
        assert_eq!(summary_sampling(), SUMMARY_SAMPLING, "{bad:?}");
    }
    drop(env);
}
