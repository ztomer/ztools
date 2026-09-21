//! Assertions the tests share.

/// Exact float equality, named as such.
///
/// The tests that use this compare against values that are EXACT in binary
/// floating point (see `eval::scoring_math`); a tolerance would weaken them.
/// `==` on floats is `clippy::float_cmp` because it is usually an accident;
/// this macro is the statement that here it is not. The comparison is
/// `partial_cmp`, so it has `==`'s own semantics: `0.0` equals `-0.0`, NaN
/// equals nothing.
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
