//! The two numeric conversions the scorers do everywhere, done once.
//!
//! Validators turn counts into ratios and ratios into whole-number scores, and
//! both steps were written as bare `as` casts at more than fifty sites. The
//! reason to centralise them is not the lint count: an `as` cast is a SILENT
//! conversion, so each site was its own unstated decision about what happens at
//! the boundary, and no two of them could be checked against each other.
//!
//! # On `clippy::float_cmp` in the test modules
//!
//! Several test modules carry `#![expect(clippy::float_cmp)]` pointing here.
//! Every comparison they make is against a value that is EXACT in binary
//! floating point: small integers, one-half steps, and correctly-rounded
//! quotients of small integers. IEEE-754 division is correctly rounded, so
//! `4.0/5.0` and the literal `0.8` are the same bits, and so are `4e6/1e9` and
//! `0.004`. Replacing those with an epsilon would not make the tests more
//! robust; it would WEAKEN them, because they assert that the arithmetic is
//! right rather than that it is close.
//!
//! Neither helper changes any existing behaviour. `ratio` deliberately does NOT
//! special-case a zero denominator -- see its own note.

/// `part / whole`, both counts.
///
/// A `usize` count is exact in `f64` up to 2^53; these are counts of items in
/// one task's answer, so the conversion cannot lose anything.
///
/// A ZERO DENOMINATOR IS LEFT NON-FINITE, ON PURPOSE: `0/0` is NaN and `n/0`
/// for `n > 0` is `+inf`. Returning 0.0 instead would look tidier and would
/// change results, because callers compare the ratio against a threshold and
/// `x < threshold` is FALSE for both NaN and `+inf`, where `0.0 < threshold` is
/// true. That flips the verdict for an empty input -- a decision each caller
/// has already made for itself by guarding, or not guarding, its own empty
/// case. Changing it here would move scores silently.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "counts of items within a single task's answer -- tens to \
              thousands, far below 2^53 -- so both conversions are exact"
)]
pub fn ratio(part: usize, whole: usize) -> f64 {
    part as f64 / whole as f64
}

/// A score rounded to a whole number.
///
/// `f64 as i64` has saturated rather than been undefined since Rust 1.45, so
/// an out-of-range value clamps instead of wrapping and NaN becomes 0. What
/// this adds over the bare cast is that the rounding rule and that guarantee
/// are stated once instead of at twenty call sites.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    reason = "`round()` has already made the value whole, and the `as i64` \
              saturates rather than wrapping, so the only values that move are \
              ones no score can reach"
)]
pub const fn rounded(value: f64) -> i64 {
    value.round() as i64
}

#[cfg(test)]
#[path = "scoring_math_tests.rs"]
mod tests;
