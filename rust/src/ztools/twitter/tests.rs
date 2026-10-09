//! Unit tests for the Rust Twitter summarizer module.
//!
//! Split by domain for the 500-line cap (no test exemption; see CLAUDE.md):
//! `text` for the pure string helpers, `summary` for the end-to-end
//! `run_summary` path, and `support` for the sandbox config plus the loopback
//! stub both share.
//!
//! WHAT THE SANDBOX IS FOR. `ZtoolsConfig::default()` names three paths on the
//! developer's disk: a cache file, the Playwright collector's checkout and
//! that checkout's `conf/twitter.toml`. Four of these tests passed the
//! default and were safe only because `run_summary` reads the cache
//! EXCLUSIVELY when handed no tweets — one reorder and they wrote the
//! operator's real cache. The third had already leaked in a quieter way: the
//! suite only passed because THIS CHECKOUT's `conf/twitter.toml` carries the
//! `[fallback]` table `load_fallback_policy` requires, so the tests silently
//! depended on the developer's checkout being present and current. Every path
//! now comes from `support::sandboxed_config`, which points all of them inside
//! the test's own temp dir.

// `#[path]` is needed because a `#[path = "tests.rs"]` module resolves its
// children from the PARENT directory, not from `tests/`.
#[path = "tests/summary.rs"]
mod summary;
#[path = "tests/support.rs"]
mod support;
#[path = "tests/text.rs"]
mod text;
