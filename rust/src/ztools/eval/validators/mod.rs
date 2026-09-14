//! Deep evaluation validators and scorers for ranking models across tasks.
//!
//! Submodules:
//! - `text_match`: identifying token and semantic paraphrase overlap.
//! - `contract`: reading prompt instructions and bounds from prompt text.
//! - `defects`: generic location, constant column, and near duplicate detectors.
//! - `json_validator`: detailed JSON and signal/noise validation.
//! - `adversarial`: fabrication traps and prompt injection resistance.
//! - `attribution`: tweet summary author and timestamp citation faithfulness.
//! - `faithfulness`: instruction-leak, strict-schema, and contradiction gates.
//! - `filename`: filename-quality scorer with leak and relevance gates.
//! - `summary`: summary-quality scorer with misattribution and placeholder caps.
//! - `taxes_grounded`: arithmetic and citation grounding for financial tasks.

pub mod adversarial;
pub mod attribution;
pub mod contract;
pub mod defects;
pub mod faithfulness;
pub mod filename;
pub mod json_validator;
pub mod summary;
pub mod taxes_grounded;
pub mod taxes_rubric;
pub use taxes_rubric::{
    validate_taxes_anomalies, validate_taxes_audit_readiness, validate_taxes_synthesis,
};
pub mod text_match;

pub use adversarial::*;
pub use attribution::*;
pub use contract::*;
pub use defects::*;
pub use faithfulness::{validate_no_contradiction, validate_no_leak, validate_strict_schema};
pub use filename::{filename_relevance, validate_filename, FILENAME_IRRELEVANT_MAX_SCORE};
pub use json_validator::{
    check_source_extraction, extract_list_from_dict, has_item_details, is_valid_list_item,
    name_tokens, names_match, norm_name, validate_detailed_json, validate_json,
    validate_mixed_signal,
};
pub use summary::{validate_summary, MISATTRIBUTION_MAX_SCORE};
pub use taxes_grounded::{
    cents, known_set, prose_amounts, score_prose_amounts, traceable_sums, validate_taxes_qa,
    validate_taxes_slip_qa, validate_taxes_yoy_narrative,
};
pub use text_match::*;

#[cfg(test)]
#[path = "ordering_tests.rs"]
mod ordering_tests;

#[cfg(test)]
#[path = "summary_arm_tests.rs"]
mod summary_arm_tests;
