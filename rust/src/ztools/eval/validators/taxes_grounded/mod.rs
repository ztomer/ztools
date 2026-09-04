//! Grounded arithmetic and citation validation for taxes tasks.
//!
//! Port of `lib/validators/taxes_grounded.py`.
//!
//! Shim: split into `amounts` (money parsing and grounding arithmetic),
//! `grounding` (fixture loading and output unwrapping) and `validate` (the three
//! public validators) to stay under the 500-line production cap. It was at
//! 498/500 -- passing, with room for two lines. Every public name is re-exported
//! here, so `validators::taxes_grounded::X` keeps resolving.

mod amounts;
mod grounding;
mod validate;

pub use amounts::{
    cents, known_set, prose_amounts, score_prose_amounts, traceable_sums, MAX_SCORE,
    MAX_SUBSET_VALUES,
};
pub use validate::{validate_taxes_qa, validate_taxes_slip_qa, validate_taxes_yoy_narrative};

#[cfg(test)]
#[path = "taxes_grounded_tests.rs"]
mod tests;
