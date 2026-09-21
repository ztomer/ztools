use super::{pct_floor, pct_floor_mean, pct_round, ratio, rounded};

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
    assert_exact!(ratio(3, 0), f64::INFINITY, "n/0 is +inf, not NaN");
    // The property the callers actually depend on: neither shape compares
    // below a threshold, which 0.0 would.
    for empty in [ratio(0, 0), ratio(3, 0)] {
        // Not `a >= b`: for the non-finite values this test is about, `<`
        // and `>=` are BOTH false, and the callers use `<`. `partial_cmp`
        // says the same thing without a negated operator: never `Less`.
        assert_ne!(
            empty.partial_cmp(&0.5),
            Some(std::cmp::Ordering::Less),
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

#[test]
fn pct_round_ties_to_even_like_python() {
    // round(12.5) == 12 and round(37.5) == 38 in Python.
    assert_eq!(pct_round(1, 8), 12);
    assert_eq!(pct_round(3, 8), 38);
    assert_eq!(pct_round(2, 3), 67);
}

#[test]
fn pct_floor_truncates_like_python_int() {
    assert_eq!(pct_floor(2, 3), 66);
    assert_eq!(pct_floor(1, 1), 100);
    assert_eq!(pct_floor(0, 5), 0);
}

#[test]
fn pct_floor_mean_blends_then_truncates() {
    assert_eq!(pct_floor_mean(1.0, 0.5), 75);
    assert_eq!(pct_floor_mean(2.0 / 3.0, 1.0), 83);
}

#[test]
fn pct_multiplies_before_it_divides_like_python() {
    // `100.0 * (29.0 / 100.0)` is 28.999999999999996 in f64; Python's
    // `int(100 * 29 / 100)` multiplies first and gets 29. So do we.
    assert_eq!(pct_floor(29, 100), 29);
    assert_eq!(pct_round(29, 100), 29);
    for whole in 1..=400_usize {
        for part in 0..=whole {
            let exact = (100 * part) / whole;
            assert_eq!(
                pct_floor(part, whole),
                i64::try_from(exact).unwrap(),
                "{part}/{whole}"
            );
        }
    }
}
