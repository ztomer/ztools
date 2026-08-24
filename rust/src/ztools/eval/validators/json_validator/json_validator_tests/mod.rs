//! json_validator's tests, split by which validator/helper they target. Split
//! out of a single 556-line tests.rs for the 500-line cap (no test exemption;
//! see CLAUDE.md). `support` holds `detailed_items`, shared by several modules.

mod support;

mod basics;
mod detailed_score;
mod json_score;
mod mixed_signal;
mod shape;
