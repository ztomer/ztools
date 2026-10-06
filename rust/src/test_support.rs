//! Assertions the tests share.
//!
//! `#[macro_export]` on each one, so they are `ztools::assert_exact!`,
//! `ztools::assert_empty!` and `ztools::assert_nonempty!` from an integration
//! test, which sees only the public API. `lib.rs` keeps `#[macro_use]` on this
//! module for the other half -- the crate's own unit tests spell them bare. The
//! module itself is compiled unconditionally for the same reason `test_env` is:
//! a sanctioned assertion nobody outside the crate can spell is not sanctioned.
//!
//! Exporting a macro does not document it anywhere a reader will look, so each
//! doc comment here is the whole contract.

/// Exact float equality, named as such.
///
/// The tests that use this compare against values that are EXACT in binary
/// floating point (see `eval::scoring_math`); a tolerance would weaken them.
/// `==` on floats is `clippy::float_cmp` because it is usually an accident;
/// this macro is the statement that here it is not. The comparison is
/// `partial_cmp`, so it has `==`'s own semantics: `0.0` equals `-0.0`, NaN
/// equals nothing.
#[macro_export]
macro_rules! assert_exact {
    ($left:expr, $right:expr $(,)?) => {{
        let (left, right) = ($left, $right);
        assert!(
            ::std::cmp::PartialOrd::partial_cmp(&left, &right) == Some(::std::cmp::Ordering::Equal),
            "floats differ: {left:?} vs {right:?}"
        );
    }};
    ($left:expr, $right:expr, $($arg:tt)+) => {{
        let (left, right) = ($left, $right);
        assert!(
            ::std::cmp::PartialOrd::partial_cmp(&left, &right) == Some(::std::cmp::Ordering::Equal),
            $($arg)+
        );
    }};
}

/// Emptiness, named as such, with the offending value on failure.
///
/// `assert!(x.is_empty())` fails with nothing but "assertion failed" — the one
/// moment the value matters — and `clippy::assert_is_empty` (pedantic) asks for
/// `assert_eq!`, which prints it. Clippy's own suggestion would spell the element
/// type out (`[] as [std::string::String; 0]`); doing that by hand at every site
/// is noise, and hand-written `x.len() == 0` is the same defect wearing a hat.
/// This is the one spelling, so a failure names what was actually there.
///
/// The house checker is stricter than the lint, deliberately: clippy is SILENT
/// on `assert!(v.is_empty(), "with a message")`, so an author who writes a
/// helpful failure message gets a clean lint run and ships the finding anyway.
/// Hence the message arm, which keeps the explanation AND the value.
#[macro_export]
macro_rules! assert_empty {
    ($left:expr $(,)?) => {{
        let left = $left;
        assert_eq!(left.len(), 0, "expected empty, got {left:?}");
    }};
    ($left:expr, $($arg:tt)+) => {{
        let left = $left;
        assert_eq!(left.len(), 0, $($arg)+);
    }};
}

/// The negation, for the same reason and with the same value on failure.
#[macro_export]
macro_rules! assert_nonempty {
    ($left:expr $(,)?) => {{
        let left = $left;
        assert_ne!(left.len(), 0, "expected non-empty, got {left:?}");
    }};
    ($left:expr, $($arg:tt)+) => {{
        let left = $left;
        assert_ne!(left.len(), 0, $($arg)+);
    }};
}
