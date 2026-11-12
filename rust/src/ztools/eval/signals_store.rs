//! The signal store on disk: where the LIVE file is, the read-only seed it
//! starts from, and load/save. Split out of `signals.rs` for the 500-line cap,
//! along the seam the code already had (storage vs. estimation).

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde_json::Value;

/// Where the LIVE `eval_signals.json` lives: `$EVAL_SIGNALS_DIR`, else
/// `~/.local/state/ztools/eval/`. Env-overridable so tests can point it at tmp.
///
/// A run's learned rates are STATE, never a tracked config (ZT.EVALSIGNALS):
/// this file was `conf/eval_signals.json` in the checkout that built the binary,
/// so every `model-eval` -- and the scheduled models run -- left the tree dirty
/// and blocked the next ship. The checkout's copy is now only the read-only
/// [`seed_path`] a machine with no state starts from.
#[must_use]
pub fn signals_path() -> PathBuf {
    let dir = std::env::var("EVAL_SIGNALS_DIR").map_or_else(
        |_| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".local/state/ztools/eval")
        },
        PathBuf::from,
    );
    dir.join("eval_signals.json")
}

/// The shipped, read-only `conf/eval_signals.json` a fresh machine starts from.
///
/// `None` when `EVAL_SIGNALS_DIR` names the store explicitly: that directory is
/// then the WHOLE store -- a test or a sweep that pointed it somewhere asked for
/// nothing else to be mixed in.
///
/// Looked for in `ZTOOLS_CONF_DIR`, then the checkout that built the binary
/// (`CARGO_MANIFEST_DIR/../conf`), then the home-directory checkout's `conf/`. Nothing in
/// this module ever writes it.
#[must_use]
pub fn seed_path() -> Option<PathBuf> {
    if std::env::var_os("EVAL_SIGNALS_DIR").is_some() {
        return None;
    }
    let candidates = [
        std::env::var_os("ZTOOLS_CONF_DIR").map(PathBuf::from),
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .map(|p| p.join("conf")),
        dirs::home_dir().map(|h| h.join("Projects/ztools/conf")),
    ];
    // An explicit `ZTOOLS_CONF_DIR` is the only candidate: falling through to
    // the real checkout from a sandbox would seed a test from the operator's
    // data.
    let explicit = candidates[0].is_some();
    candidates
        .into_iter()
        .flatten()
        .take(if explicit { 1 } else { usize::MAX })
        .map(|d| d.join("eval_signals.json"))
        .find(|p| p.is_file())
}

/// The file a read should come from: the live store when it exists, else the
/// seed, else nothing.
#[must_use]
pub fn store_source() -> Option<PathBuf> {
    let live = signals_path();
    if live.is_file() {
        return Some(live);
    }
    seed_path()
}

pub type SignalStore = BTreeMap<String, Value>;

#[must_use]
pub fn load_signals() -> SignalStore {
    store_source()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .map_or_else(SignalStore::new, |text| {
            serde_json::from_str(&text).unwrap_or_default()
        })
}

/// Write the live store (never the seed), creating its directory.
pub fn save_signals(signals: &SignalStore) {
    let path = signals_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(text) = serde_json::to_string_pretty(signals) {
        let _ = std::fs::write(path, text);
    }
}
