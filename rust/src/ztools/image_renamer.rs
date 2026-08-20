//! Shim re-exporting the split `rename/` package (500-line-cap split).
//!
//! `rename/mod.rs` owns the decision flow and filesystem walk, `rename/helpers.rs`
//! ports the pure text cleaning from `rename/helpers.py`, and `rename/vlm.rs`
//! ports the LLM/VLM naming paths from `rename/llm.py`.
pub use crate::ztools::rename::*;
