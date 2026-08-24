//! Tests for the post-parse constraint enforcement ported from
//! `references/weekend/enforce.py` (weakness classes C3/C5/C8 in
//! `docs/REPORT_WEAKNESS_CLASSES.md`).
//!
//! These are pure functions over the parsed event list -- no LLM, no network --
//! so they are deterministic and cheap to test. Each drops/corrects and reports
//! a note rather than failing silently.
//!
//! Split by domain (exclusions, constant-column defects, dates/provenance,
//! float-to-top) to stay under the 500-line cap; `support` holds shared helpers.

mod support;

mod dates;
mod defects;
mod exclusions;
mod float_top;
