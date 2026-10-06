//! The shared assertions must be REACHABLE from an integration test, and must
//! behave once they get there.
//!
//! This file is a GATE, not a convenience: it compiles against
//! `ztools::assert_exact!` / `ztools::assert_empty!` / `ztools::assert_nonempty!`,
//! which is only possible while `lib.rs` compiles `mod test_support`
//! unconditionally and exports the macros by path. Re-gating the module behind
//! `#[cfg(test)]` turns this file into a compile error naming the missing path,
//! which is the regression this pins -- an integration test wanting an emptiness
//! assertion must never be pushed back into hand-rolling a spelling the house
//! emptiness checker rejects (`assert!(x.is_empty(), …)`).
//!
//! The behaviour half is not decoration. A macro that is reachable but prints
//! nothing is exactly as useless as one that is missing, and the value-on-failure
//! is the whole documented purpose: `assert!(x.is_empty())` fails with
//! "assertion failed" at the one moment the value matters. So each of the three is
//! pinned in both directions -- passing on the value it should accept, and
//! panicking WITH THE VALUE on the one it should not.
//!
//! Hermetic: no I/O at all, so there is nothing to reach the network for.

use ztools::{assert_empty, assert_exact, assert_nonempty};

/// Every sanctioned spelling is usable from here, on values each accepts.
#[test]
fn sanctioned_assertions_are_reachable_from_an_integration_test() {
    assert_exact!(0.1_f64 + 0.2_f64, 0.300_000_000_000_000_04_f64);
    assert_empty!(Vec::<u8>::new());
    assert_empty!("");
    assert_nonempty!(vec![1_u8]);
    assert_nonempty!("x");
}

/// The emptiness assertion names the value it found, which is its entire reason
/// to exist over `assert!(x.is_empty())`.
#[test]
#[should_panic(expected = "expected empty, got [7, 9]")]
fn assert_empty_names_the_value_that_broke_it() {
    assert_empty!(vec![7_u8, 9]);
}

/// …and the message arm keeps a caller's explanation alongside that value, the
/// defect the house checker exists to stop (clippy is silent when a helpful
/// message is supplied, so an author writes one and ships the finding anyway).
#[test]
#[should_panic(expected = "the roll-up must drop 429s")]
fn assert_empty_keeps_the_caller_s_message_and_the_value() {
    assert_empty!([429_u16], "the roll-up must drop 429s");
}

/// The negation fails on the empty value, not silently passing as a length test
/// would when written as `!= 0` with no message.
#[test]
#[should_panic(expected = "expected non-empty, got []")]
fn assert_nonempty_rejects_the_empty_collection() {
    assert_nonempty!(Vec::<u32>::new());
}

/// Exact float equality is exact: the tolerance a reviewer would reach for would
/// make this comparison a different assertion than the one the macro names.
#[test]
#[should_panic(expected = "floats differ")]
fn assert_exact_rejects_a_difference_a_tolerance_would_forgive() {
    assert_exact!(0.1_f64 + 0.2_f64, 0.3_f64);
}
