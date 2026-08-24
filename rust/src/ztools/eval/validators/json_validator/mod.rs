//! JSON structure, details, validity, and signal/noise validation.
//!
//! Port of `lib/validators/json_validator.py`.
//!
//! Shim: split into `weights` (the scoring numbers), `items` (item shape),
//! `names` (fuzzy name matching), `source` (grounding ratio) and `score` (the
//! three public validators) to stay under the 500-line production cap. Every
//! public name is re-exported here, so `validators::json_validator::X` and
//! `super::json_validator::_names_match` keep resolving.

mod items;
mod names;
mod score;
mod source;
mod weights;

pub use items::{extract_list_from_dict, has_item_details, is_valid_list_item};
pub use names::{_name_tokens, _names_match, _norm_name};
pub use score::{validate_detailed_json, validate_json, validate_mixed_signal};
pub use source::check_source_extraction;
pub use weights::*;

#[cfg(test)]
#[path = "json_validator_tests/mod.rs"]
mod tests;
