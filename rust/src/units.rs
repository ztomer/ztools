//! Narrowing conversions that are decisions, not incidents.
//!
//! A sibling of `routines::units` (same author, same two conversions, same
//! reasons). Kept local rather than shared because a crate dependency between
//! two unrelated tools to hold twenty lines would be the larger mistake -- but
//! if a third repo needs these, that is the moment to extract them.

use std::time::Duration;

/// A process id as the signed `pid_t` the libc calls take.
///
/// `as` here is not a formality. `u32 as i32` turns anything above `i32::MAX`
/// NEGATIVE, and a negative pid is not rejected by `kill(2)` -- it names a
/// process GROUP. `kill(-1, 0)` in particular asks about every process the
/// caller may signal, so it succeeds whenever anything is running.
///
/// That matters because the pid this is called with is PARSED FROM A FILE. A
/// truncated or corrupted GPU-lock file holding `4294967295` would otherwise
/// report its owner as alive forever, and the lock would never be reclaimed.
/// Saturating to `i32::MAX` yields a pid that does not exist, so the answer is
/// "not alive" -- which is the safe direction for a lock.
#[must_use]
pub fn pid(id: u32) -> i32 {
    i32::try_from(id).unwrap_or(i32::MAX)
}

/// A duration as whole milliseconds.
///
/// `Duration::as_millis` is `u128`; every caller reports the number as `u64`.
/// `as u64` would WRAP on an absurd duration and report a small number, which
/// is the one reading nobody would question. `u64::MAX` ms is 584 million
/// years, so saturating makes the impossible case obvious instead.
#[must_use]
pub fn millis(d: Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
#[path = "units_tests.rs"]
mod tests;
