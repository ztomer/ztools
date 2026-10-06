//! The `ztools` library: the four tools as modules plus the CLI dispatch.
//!
//! A library plus a thin binary so `pub` module API does not trip the
//! dead-code lint the way it does in a pure binary crate (these items were
//! library-public in `routines` and are still public API here).

// Unconditional, NOT `#[cfg(test)]`, for the same reason as `test_env` below: a
// sanctioned assertion is only sanctioned if it is reachable from BOTH
// compilation universes. Gated, `assert_empty!` is invisible to the integration
// tests under `rust/tests/`, which see only the library's public API -- and the
// house emptiness checker is STRICTER than clippy (it rejects
// `assert!(x.is_empty(), …)` outright), so every integration test wanting an
// emptiness assertion would hand-roll the spelling the gate is about to reject.
//
// `#[macro_use]` stays because it is the OTHER half of that, not a leftover: the
// crate's own 161 call sites spell these macros bare, and `#[macro_export]` --
// which is what makes `ztools::assert_empty!` nameable from a downstream test
// binary -- exports to the crate root without putting them in textual scope.
#[macro_use]
pub mod test_support;

// Unconditional, NOT `#[cfg(test)]`: the sandbox guard has to be reachable from
// BOTH compilation universes — this crate's own unit tests and the integration
// tests under `rust/tests/`, which only see the library's public API. Gating it
// would leave `tests/model_resolve_http.rs` unable to stop writing to the
// developer's real `$HOME`.
pub mod test_env;

pub mod cli;
pub mod cli_ztools;
pub mod cli_ztools_twitter;
pub mod config;
pub mod manifest;
pub mod units;
pub mod ztools;

// The ported modules live under `ztools/` to keep their `#[path]` test wiring
// intact; re-export them at the crate root so callers (and the integration
// tests) get `ztools::weekend` instead of `ztools::ztools::weekend`.
pub use ztools::embeddings;
pub use ztools::eval;
pub use ztools::image_renamer;
pub use ztools::model_eval;
pub use ztools::model_health;
pub use ztools::twitter;
pub use ztools::weekend;
pub use ztools::weekend_cache;
