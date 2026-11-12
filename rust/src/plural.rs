//! A count meets its noun here, never as a hedge (`file(s)`).
//!
//! `1 task`, `0 tasks`, `3 tasks`: the number decides the form. A hedged plural in a
//! string says the author did not know the count, and the house `plural-hedge` scan
//! (GOH.PLURALHEDGE) turns every one into a finding. Every noun this crate counts
//! pluralises with a plain `s`, so there is one helper and no irregular table; a
//! noun that is not regular belongs in [`noun_as`].

/// The noun alone, for a sentence that places the number itself: `noun(2, "task")` is `tasks`.
#[must_use]
pub fn noun<N: Copy + PartialEq + From<u8>>(n: N, singular: &str) -> String {
    if n == N::from(1) {
        singular.to_string()
    } else {
        format!("{singular}s")
    }
}

/// As [`noun`], for a noun whose plural is not the singular plus `s`.
#[must_use]
pub fn noun_as<N: Copy + PartialEq + From<u8>>(n: N, singular: &str, plural: &str) -> String {
    if n == N::from(1) {
        singular.to_string()
    } else {
        plural.to_string()
    }
}

/// A count with its noun: `counted(1, "file")` is `1 file`, `counted(0, "file")` is `0 files`.
#[must_use]
pub fn counted<N: Copy + PartialEq + From<u8> + std::fmt::Display>(n: N, singular: &str) -> String {
    format!("{n} {}", noun(n, singular))
}

#[cfg(test)]
#[path = "plural_tests.rs"]
mod tests;
