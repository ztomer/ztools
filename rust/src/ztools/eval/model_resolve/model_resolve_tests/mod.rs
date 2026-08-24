//! `model_resolve`'s tests, split by the domain each targets. Split out of a
//! single 571-line `tests.rs` for the 500-line cap (no test exemption; see
//! CLAUDE.md). `support` holds `DiskGuard`, shared by `disk` and `fetch`.

mod support;

mod roster;
mod disk;
mod fetch;
