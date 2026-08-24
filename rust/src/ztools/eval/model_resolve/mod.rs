//! Resolve a configured model name against the roster the server can serve.
//!
//! Ported from `lib/model_resolve.py`. A config names models by server tag,
//! and that tag is not a stable identity: models get deleted and renamed on
//! disk underneath a config that still names the old one, after which the
//! server answers every request with
//!
//! ```text
//! HTTP 404 {"error": {"message": "Model 'X' is not installed ..."}}
//! ```
//!
//! Substitution is a probe-and-degrade for exactly that case. Nothing here
//! rewrites any config; a substitution is a stopgap that says so out loud on
//! every use -- the fix is to re-derive best_models from an eval sweep.
//!
//! Shim: split into `roster` (entries, scoring, fallback chain), `disk`
//! (filesystem probing) and `fetch` (live roster + corroboration) to stay under
//! the 500-line cap. Every public name is re-exported here.

mod disk;
mod fetch;
mod roster;

pub use disk::{disk_corroborated, is_generative_model, model_config_path};
pub use fetch::fetch_roster;
pub use roster::{
    default_fallback_chain, is_missing_model_error, parameter_billions, substitute_model,
    RosterEntry, MISSING_MODEL_MARKERS,
};

#[cfg(test)]
#[path = "model_resolve_tests/mod.rs"]
mod tests;
