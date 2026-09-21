use super::{millis, pid};
use std::time::Duration;

#[test]
fn pid_is_the_ordinary_answer_for_real_pids() {
    assert_eq!(pid(1), 1);
    let live = std::process::id();
    assert_eq!(u32::try_from(pid(live)).unwrap(), live);
}

#[test]
fn pid_never_produces_a_negative_number() {
    // The whole point. `kill(-1, 0)` asks about every process the caller may
    // signal and succeeds whenever anything is running, so a negative pid turns
    // "is the lock owner alive?" into "is anything alive?" -- permanently true.
    for raw in [u32::MAX, u32::MAX - 1, 1 << 31] {
        assert!(pid(raw) > 0, "{raw} produced a non-positive pid");
    }
    assert_eq!(pid(u32::MAX), i32::MAX);
}

#[test]
fn millis_is_the_ordinary_answer_and_saturates_at_the_top() {
    assert_eq!(millis(Duration::from_millis(0)), 0);
    assert_eq!(millis(Duration::from_secs(3)), 3_000);
    assert_eq!(millis(Duration::MAX), u64::MAX);
}

#[test]
fn count_is_exact_for_every_count_f64_can_hold() {
    use super::count;
    assert_exact!(count(0), 0.0f64);
    assert_exact!(count(7), 7.0f64);
    // Straddles the two halves: 2^32 + 5.
    assert_exact!(count(4_294_967_301), 4_294_967_301.0f64);
    // The last integer f64 holds exactly.
    assert_exact!(count(1 << 53), 9_007_199_254_740_992.0f64);
}

#[test]
fn signed_carries_the_sign_and_the_magnitude() {
    use super::signed;
    assert_exact!(signed(-3), (-3.0f64));
    assert_exact!(signed(12), 12.0f64);
    assert_exact!(signed(0), 0.0f64);
}

#[test]
fn whole_i64_has_the_boundaries_the_cast_had() {
    use super::whole_i64;
    assert_eq!(whole_i64(3.0), 3);
    assert_eq!(whole_i64(3.9), 3, "truncates toward zero");
    assert_eq!(whole_i64(-3.9), -3, "truncates toward zero");
    assert_eq!(whole_i64(-3.0), -3);
    assert_eq!(whole_i64(f64::INFINITY), i64::MAX);
    assert_eq!(whole_i64(f64::NEG_INFINITY), i64::MIN);
    assert_eq!(whole_i64(1e30), i64::MAX);
    assert_eq!(whole_i64(f64::NAN), 0);
}

#[test]
fn whole_unsigned_clamps_below_at_zero_and_above_at_max() {
    use super::{whole_u32, whole_u64};
    assert_eq!(whole_u64(3.0), 3);
    assert_eq!(whole_u64(3.9), 3, "truncates toward zero");
    assert_eq!(whole_u64(-0.5), 0);
    assert_eq!(whole_u64(-1.0), 0);
    assert_eq!(whole_u64(f64::NAN), 0);
    assert_eq!(whole_u64(1e30), u64::MAX);
    assert_eq!(whole_u32(70_000.0), 70_000);
    assert_eq!(whole_u32(-1.0), 0);
    assert_eq!(whole_u32(1e12), u32::MAX);
}
