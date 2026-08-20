//! The `ztools` library: the four tools as modules plus the CLI dispatch.
//!
//! A library plus a thin binary so `pub` module API does not trip the
//! dead-code lint the way it does in a pure binary crate (these items were
//! library-public in `routines` and are still public API here).

pub mod cli;
pub mod cli_ztools;
pub mod config;
pub mod manifest;
pub mod ztools;

// The ported modules live under `ztools/` to keep their `#[path]` test wiring
// intact; re-export them at the crate root so callers (and the integration
// tests) get `ztools::weekend` instead of `ztools::ztools::weekend`.
pub use ztools::embeddings;
pub use ztools::image_renamer;
pub use ztools::model_eval;
pub use ztools::model_health;
pub use ztools::twitter;
pub use ztools::weekend;
pub use ztools::weekend_cache;
