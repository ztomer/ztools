//! Narrowing conversions that are decisions, not incidents.
//!
//! A sibling of `routines::units` (same author, same two conversions, same
//! reasons). Kept local rather than shared because a crate dependency between
//! two unrelated tools to hold twenty lines would be the larger mistake -- but
//! if a third repo needs these, that is the moment to extract them.

use num_traits::ToPrimitive;
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

/// A count as `f64`, exact.
///
/// `usize as f64` is `cast_precision_loss` because a count above 2^53 would
/// round. Every count this crate converts is items in one answer or tasks in
/// one run, but the conversion should not have to argue that: it is done in
/// two exact halves and is exact for every value below 2^53, which is where
/// `f64` itself stops being able to hold an integer.
#[must_use]
pub fn count(n: usize) -> f64 {
    unsigned(u64::try_from(n).unwrap_or(u64::MAX))
}

/// A `u64` as `f64`, exact below 2^53 -- the two-halves conversion behind
/// [`count`], for the sizes and totals that are already 64-bit.
#[must_use]
pub fn unsigned(n: u64) -> f64 {
    let hi = u32::try_from(n >> 32).unwrap_or(u32::MAX);
    let lo = u32::try_from(n & 0xffff_ffff).unwrap_or(u32::MAX);
    f64::from(hi).mul_add(4_294_967_296.0, f64::from(lo))
}

/// A signed whole number as `f64`, exact below 2^53 in magnitude.
#[must_use]
pub fn signed(n: i64) -> f64 {
    let magnitude = unsigned(n.unsigned_abs());
    if n < 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// An `f64` as `i64`, truncating toward zero, with the cast's boundaries.
///
/// `f64 as i64` has saturated out of range and sent NaN to 0 since Rust
/// 1.45. Stated here so the narrowing is one named decision instead of an
/// `as` at every site.
#[must_use]
pub fn whole_i64(v: f64) -> i64 {
    v.to_i64().unwrap_or_else(|| {
        if v.is_nan() {
            0
        } else if v.is_sign_negative() {
            i64::MIN
        } else {
            i64::MAX
        }
    })
}

/// An `f64` as `u64`, truncating toward zero: negative and NaN are 0, too
/// large saturates -- the `f64 as u64` boundaries, made explicit.
#[must_use]
pub fn whole_u64(v: f64) -> u64 {
    v.to_u64().unwrap_or_else(|| {
        if v.is_nan() || v.is_sign_negative() {
            0
        } else {
            u64::MAX
        }
    })
}

/// An `f64` as `u32`, same boundaries as [`whole_u64`].
#[must_use]
pub fn whole_u32(v: f64) -> u32 {
    v.to_u32().unwrap_or_else(|| {
        if v.is_nan() || v.is_sign_negative() {
            0
        } else {
            u32::MAX
        }
    })
}

#[cfg(test)]
#[path = "units_tests.rs"]
mod tests;
