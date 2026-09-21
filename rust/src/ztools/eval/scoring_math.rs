//! The two numeric conversions the scorers do everywhere, done once.
//!
//! Validators turn counts into ratios and ratios into whole-number scores, and
//! both steps were written as bare `as` casts at more than fifty sites. The
//! reason to centralise them is not the lint count: an `as` cast is a SILENT
//! conversion, so each site was its own unstated decision about what happens at
//! the boundary, and no two of them could be checked against each other.
//!
//! # On exact float comparison in the test modules
//!
//! Several test modules compare floats with `assert_exact!` (see the crate's
//! `test_support`). Every comparison they make is against a value that is
//! EXACT in binary floating point: small integers, one-half steps, and
//! correctly-rounded quotients of small integers. IEEE-754 division is
//! correctly rounded, so `4.0/5.0` and the literal `0.8` are the same bits,
//! and so are `4e6/1e9` and `0.004`. Replacing those with an epsilon would not
//! make the tests more robust; it would WEAKEN them, because they assert that
//! the arithmetic is right rather than that it is close. The macro says so in
//! its name, where `assert_eq!` on floats only looks like an accident.
//!
//! Neither helper changes any existing behaviour. `ratio` deliberately does NOT
//! special-case a zero denominator -- see its own note.

use crate::units::{count, whole_i64};

/// `part / whole`, both counts.
///
/// A `usize` count is exact in `f64` up to 2^53 ([`count`] does the conversion
/// in two exact halves); these are counts of items in one task's answer, so
/// nothing is lost.
///
/// A ZERO DENOMINATOR IS LEFT NON-FINITE, ON PURPOSE: `0/0` is NaN and `n/0`
/// for `n > 0` is `+inf`. Returning 0.0 instead would look tidier and would
/// change results, because callers compare the ratio against a threshold and
/// `x < threshold` is FALSE for both NaN and `+inf`, where `0.0 < threshold` is
/// true. That flips the verdict for an empty input -- a decision each caller
/// has already made for itself by guarding, or not guarding, its own empty
/// case. Changing it here would move scores silently.
#[must_use]
pub fn ratio(part: usize, whole: usize) -> f64 {
    count(part) / count(whole)
}

/// A score rounded to a whole number, half away from zero.
///
/// The narrowing is [`whole_i64`]: out of range saturates and NaN becomes 0,
/// stated once instead of at twenty call sites.
#[must_use]
pub fn rounded(value: f64) -> i64 {
    whole_i64(value.round())
}

/// Python's `round(100 * part / whole)`: percent with ties to even, which is
/// what Python's `round` does and `f64::round` does not (`12.5` -> 12, not 13).
///
/// Integer arithmetic throughout, because the Python multiplies FIRST:
/// `100 * part` is exact and one division follows. Multiplying the ratio
/// instead (`100.0 * (part / whole)`) rounds twice, and `29/100` came out as
/// `28.999...` -- a 28 where Python says 29 (fixed 2026-09-21).
///
/// A zero denominator saturates like [`rounded`]: `0/0` is 0 (NaN) and
/// `n/0` is `i64::MAX`.
#[must_use]
pub fn pct_round(part: usize, whole: usize) -> i64 {
    let Some((quotient, remainder)) = pct_divmod(part, whole) else {
        return rounded(ratio(part, whole));
    };
    let twice = remainder.saturating_mul(2);
    let up = twice > whole || (twice == whole && quotient % 2 == 1);
    i64::try_from(quotient + usize::from(up)).unwrap_or(i64::MAX)
}

/// Python's `int(100 * part / whole)`: percent truncated toward zero.
///
/// Same integer arithmetic as [`pct_round`], for the same reason.
#[must_use]
pub fn pct_floor(part: usize, whole: usize) -> i64 {
    pct_divmod(part, whole).map_or_else(
        || rounded(ratio(part, whole)),
        |(quotient, _)| i64::try_from(quotient).unwrap_or(i64::MAX),
    )
}

/// `(100 * part) / whole` and its remainder, or `None` for a zero divisor.
const fn pct_divmod(part: usize, whole: usize) -> Option<(usize, usize)> {
    if whole == 0 {
        return None;
    }
    let scaled = part.saturating_mul(100);
    Some((scaled / whole, scaled % whole))
}

/// Python's `int(100 * (0.5 * a + 0.5 * b))`: the truncated percent of the
/// mean of two ratios.
#[must_use]
pub fn pct_floor_mean(a: f64, b: f64) -> i64 {
    whole_i64((100.0 * 0.5f64.mul_add(b, 0.5 * a)).trunc())
}

#[cfg(test)]
#[path = "scoring_math_tests.rs"]
mod tests;
