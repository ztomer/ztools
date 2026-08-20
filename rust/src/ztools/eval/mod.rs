//! Eval machinery: output cleaning (`clean.rs`) and task validators
//! (`validate.rs`). Ported from `lib/content_processing.py` and
//! `eval/validate.py` so the Rust eval path judges the same cleaned, parsed
//! text the Python eval does.

pub mod clean;
pub mod validate;

pub use clean::{clean_model_output, extract_content_from_code_blocks};
pub use validate::validate_file_summary;