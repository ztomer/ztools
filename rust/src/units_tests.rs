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
