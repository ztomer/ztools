#![expect(clippy::float_cmp, reason = "exact; see eval::scoring_math")]

use super::{ratio, rounded};

#[test]
fn ratio_is_the_ordinary_quotient() {
    assert!((ratio(1, 2) - 0.5).abs() < f64::EPSILON);
    assert!((ratio(3, 4) - 0.75).abs() < f64::EPSILON);
    assert!((ratio(0, 5) - 0.0).abs() < f64::EPSILON);
}

#[test]
fn a_zero_denominator_stays_nan_rather_than_becoming_zero() {
    // Pinned as a DECISION, not an accident. Callers compare the ratio against
    // a threshold, and `NaN < x` is false where `0.0 < x` is true -- so
    // "tidying" this to 0.0 flips the verdict for every empty input.
    assert!(ratio(0, 0).is_nan(), "0/0 is NaN");
    assert_eq!(ratio(3, 0), f64::INFINITY, "n/0 is +inf, not NaN");
    // The property the callers actually depend on: neither shape compares
    // below a threshold, which 0.0 would.
    for empty in [ratio(0, 0), ratio(3, 0)] {
        assert!(
            !(empty < 0.5),
            "an empty ratio must not read as below threshold"
        );
    }
}

#[test]
fn rounded_rounds_half_away_from_zero() {
    assert_eq!(rounded(0.4), 0);
    assert_eq!(rounded(0.5), 1);
    assert_eq!(rounded(99.5), 100);
    assert_eq!(rounded(-0.5), -1);
}

#[test]
fn rounded_saturates_and_never_wraps() {
    // The property the bare cast relies on and never states: since Rust 1.45
    // `f64 as i64` saturates, and NaN becomes 0 rather than something arbitrary.
    assert_eq!(rounded(f64::INFINITY), i64::MAX);
    assert_eq!(rounded(f64::NEG_INFINITY), i64::MIN);
    assert_eq!(rounded(f64::NAN), 0);
    assert_eq!(rounded(1e30), i64::MAX);
}
