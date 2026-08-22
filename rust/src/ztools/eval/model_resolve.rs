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

use std::path::{Path, PathBuf};
use std::time::Duration;

const ROSTER_TIMEOUT_SECS: u64 = 10;
const API_TAGS: &str = "/api/tags";

/// Substrings of the server's 404 body identifying a stale model tag
/// specifically, as opposed to a 404 from a wrong URL path.
pub const MISSING_MODEL_MARKERS: &[&str] = &["is not installed", "not registered with any provider"];

/// True when a 404 means "that model tag is gone", not "wrong endpoint".
pub fn is_missing_model_error(status_code: u16, body: &str) -> bool {
    if status_code != 404 {
        return false;
    }
    let lowered = body.to_lowercase();
    MISSING_MODEL_MARKERS.iter().any(|m| lowered.contains(m))
}

/// One `/api/tags` entry, narrowed to what selection reads.
#[derive(Debug, Clone, PartialEq)]
pub struct RosterEntry {
    pub model: String,
    pub parameter_size: String,
}

impl RosterEntry {
    pub fn from_json(v: &serde_json::Value) -> Option<RosterEntry> {
        let model = v.get("model")?.as_str()?.to_string();
        if model.is_empty() {
            return None;
        }
        let parameter_size = v
            .get("details")
            .and_then(|d| d.get("parameter_size"))
            .and_then(|p| p.as_str())
            .unwrap_or("")
            .to_string();
        Some(RosterEntry { model, parameter_size })
    }
}

/// Parse `details.parameter_size` ("27B", "4M", "") into billions, 0.0 if absent.
pub fn parameter_billions(entry: &RosterEntry) -> f64 {
    let raw = entry.parameter_size.trim().to_uppercase();
    if raw.is_empty() {
        return 0.0;
    }
    let scale = match raw.as_bytes()[raw.len() - 1] {
        b'B' => Some(1.0),
        b'M' => Some(0.001),
        b'K' => Some(0.000_001),
        _ => None,
    };
    match scale {
        Some(scale) => raw[..raw.len() - 1]
            .parse::<f64>()
            .map(|n| n * scale)
            .unwrap_or(0.0),
        None => 0.0,
    }
}

/// Sort key: biggest model first, then name, so the pick is deterministic.
///
/// Size is the tiebreak rather than the name because a name-sorted pick
/// silently tracks version-string formatting ("qwen3.10" sorts below
/// "qwen3.8"), which is a property of ASCII rather than of the model.
fn pick_best<'a>(entries: impl Iterator<Item = &'a RosterEntry>) -> Option<String> {
    entries
        .min_by(|a, b| {
            parameter_billions(b)
                .total_cmp(&parameter_billions(a))
                .then_with(|| a.model.cmp(&b.model))
        })
        .map(|e| e.model.clone())
}

/// Pick a servable stand-in for `configured`.
///
/// Returns `(model, reason)`. `reason` is `None` when nothing was substituted --
/// either because `configured` is installed, or because the roster is empty and
/// we have no grounds to override the caller. It is a human-readable sentence
/// otherwise, and every caller must surface it rather than swallow it.
pub fn substitute_model(
    configured: &str,
    roster: &[RosterEntry],
    fallback_chain: &[&str],
) -> (String, Option<String>) {
    if roster.is_empty() {
        return (configured.to_string(), None);
    }
    if roster.iter().any(|e| e.model == configured) {
        return (configured.to_string(), None);
    }

    let family = crate::ztools::eval::quirks::get_model_family(configured);
    if family != "default" {
        let same: Vec<&RosterEntry> = roster
            .iter()
            .filter(|e| crate::ztools::eval::quirks::get_model_family(&e.model) == family)
            .collect();
        if !same.is_empty() {
            // Non-empty by construction, so the pick cannot fail.
            let pick = pick_best(same.into_iter()).expect("non-empty family shortlist");
            return (
                pick.clone(),
                Some(format!(
                    "model '{configured}' is not installed; using '{pick}' \
                     (largest installed '{family}' model). Re-derive best_models."
                )),
            );
        }
    }

    for preferred in fallback_chain {
        let matches: Vec<&RosterEntry> = roster
            .iter()
            .filter(|e| e.model.to_lowercase().contains(preferred))
            .collect();
        if !matches.is_empty() {
            // Non-empty by construction, so the pick cannot fail.
            let pick = pick_best(matches.into_iter()).expect("non-empty chain matches");
            return (
                pick.clone(),
                Some(format!(
                    "model '{configured}' is not installed and no '{family}' model is \
                     either; falling back to '{pick}'. Re-derive best_models."
                )),
            );
        }
    }

    // Roster is known non-empty here.
    let pick = pick_best(roster.iter()).expect("non-empty roster");
    (
        pick.clone(),
        Some(format!(
            "model '{configured}' is not installed and nothing in the preference chain \
             is either; falling back to '{pick}'. Re-derive best_models."
        )),
    )
}

/// The known families, on-device first (`conf/config.toml [model_fallback_chain]`
/// overrides this in Python; the Rust eval carries no such table yet).
pub fn default_fallback_chain() -> Vec<&'static str> {
    let families = crate::ztools::eval::quirks::MODEL_FAMILIES;
    // DEFAULT_MODEL is "foundation" (lib/llm/constants.py).
    std::iter::once("foundation")
        .chain(families.iter().copied().filter(|f| *f != "foundation"))
        .collect()
}

/// `MODELS_DIR` / `HF_CACHE_DIR`, mirroring `lib/model_caps.py`.
fn models_dir() -> PathBuf {
    std::env::var("MLX_MODELS_DIR").map(PathBuf::from).unwrap_or_else(|_| {
        dirs::home_dir().unwrap_or_default().join("MLXModels")
    })
}

fn hf_cache_dir() -> PathBuf {
    let home = std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| dirs::home_dir().unwrap_or_default().join(".cache/huggingface"));
    home.join("hub")
}

/// The on-disk config.json for a served model id, if it can be found.
///
/// Served ids are lowercased while directories keep their original case, so
/// match case-insensitively. MLXModels: `<Org>/<Model>/config.json`; HF cache:
/// `models--<org>--<model>/snapshots/<sha>/config.json`.
pub fn model_config_path(model: &str) -> Option<PathBuf> {
    if model.is_empty() {
        return None;
    }
    let target = model.trim().to_lowercase();
    for root in [models_dir(), hf_cache_dir()] {
        if !root.is_dir() {
            continue;
        }
        if let Ok(entries) = walk_configs(&root) {
            for config in entries {
                let parent_is_target = config
                    .parent()
                    .and_then(|p| p.file_name())
                    .map(|n| n.to_string_lossy().to_lowercase() == target)
                    .unwrap_or(false);
                if parent_is_target {
                    return Some(config);
                }
                for part in config.components() {
                    let part_str = part.as_os_str().to_string_lossy();
                    if part_str.starts_with("models--")
                        && part_str
                            .rsplit("--")
                            .next()
                            .map(|last| last.eq_ignore_ascii_case(&target))
                            .unwrap_or(false)
                    {
                        return Some(config);
                    }
                }
            }
        }
    }
    None
}

/// config.json files under `root`, recursively.
///
/// Python's `model_config_path` rglobs both roots without a depth limit; the
/// first cut of this walk stopped at two levels and silently missed the REAL
/// HF snapshot layout `hub/models--org--model/snapshots/<sha>/config.json`
/// (three directory levels), so every HF-cache model was invisible to
/// corroboration and disk-byte sizing. Depth-capped rather than unbounded
/// purely to bound worst-case I/O on a huge cache.
fn walk_configs(root: &Path) -> std::io::Result<Vec<PathBuf>> {
    const MAX_DEPTH: usize = 6;
    fn walk(dir: &Path, depth: usize, found: &mut Vec<PathBuf>) -> std::io::Result<()> {
        if depth > MAX_DEPTH {
            return Ok(());
        }
        let direct = dir.join("config.json");
        if direct.is_file() {
            found.push(direct);
        }
        for entry in std::fs::read_dir(dir)?.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, depth + 1, found)?;
            }
        }
        Ok(())
    }
    let mut found = Vec::new();
    walk(root, 0, &mut found)?;
    Ok(found)
}

/// A window declared in `conf/models/<family>.toml` rather than read off disk.
///
/// Apple's on-device `foundation` has no config.json anywhere, so the number
/// has to be written down. Only the presence of a documented window matters
/// here, never its value.
fn documented_context_window(model: &str) -> Option<u64> {
    let lower = model.to_lowercase();
    let family = crate::ztools::eval::quirks::MODEL_FAMILIES
        .iter()
        .find(|f| lower.contains(*f))?;
    let content = std::fs::read_to_string(conf_models_root().join(format!("{family}.toml"))).ok()?;
    let val: toml::Value = toml::from_str(&content).ok()?;
    val.get("context_window")
        .and_then(|w| w.as_integer())
        .filter(|w| *w > 0)
        .map(|w| w as u64)
}

/// Where `conf/models/` lives: the repo checkout beside the compiled-in
/// candidates, same convention as `ZtoolsConfig`'s conf lookups.
fn conf_models_root() -> PathBuf {
    if let Ok(dir) = std::env::var("ZTOOLS_CONF_DIR") {
        return PathBuf::from(dir).join("models");
    }
    if let Some(home) = dirs::home_dir() {
        let p = home.join("Projects/ztools/conf/models");
        if p.is_dir() {
            return p;
        }
    }
    PathBuf::from("conf/models")
}

/// Whether a served model can generate text, judged by its config.
///
/// Ported from `lib/model_caps.py::is_generative_model` so the prefill probe
/// skips embedding models by WHAT THEY ARE instead of by name-matching a list
/// that silently misses the next one. Unknown on disk: assume generative
/// rather than silently skipping a model the user installed (foundation lands
/// here and is generative). An unreadable config: same, keep probing.
pub fn is_generative_model(model: &str) -> bool {
    const NON_GENERATIVE_TYPES: &[&str] = &["model2vec", "sentence-transformer", "static"];
    const NON_GENERATIVE_ARCHITECTURES: &[&str] = &["staticmodel", "sentencetransformer"];

    let Some(config) = model_config_path(model) else {
        return true;
    };
    let Ok(text) = std::fs::read_to_string(config) else {
        return true;
    };
    let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&text) else {
        return true;
    };
    if NON_GENERATIVE_TYPES.contains(
        &cfg.get("model_type")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_lowercase()
            .as_str(),
    ) {
        return false;
    }
    let arches = cfg
        .get("architectures")
        .and_then(|a| a.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_lowercase())
                .collect::<Vec<String>>()
        })
        .unwrap_or_default();
    !arches
        .iter()
        .any(|a| NON_GENERATIVE_ARCHITECTURES.contains(&a.as_str()))
}

/// Does anything on this machine back up the roster's claim to serve `model`?
///
/// `/api/tags` is a CLAIM, not proof: osaurus keeps its roster in memory, so a
/// model deleted from disk stays advertised until restart. Corroboration is
/// "a config.json on disk" OR "a documented context window". An unreadable
/// probe is NOT evidence of absence -- keep the entry, because wrongly dropping
/// a servable model is worse than keeping a stale one that still degrades
/// loudly at call time.
pub fn disk_corroborated(model: &str) -> bool {
    std::panic::catch_unwind(|| {
        model_config_path(model).is_some() || documented_context_window(model).is_some()
    })
    .unwrap_or(true)
}

/// Remove roster entries with nothing on disk behind them.
///
/// Done HERE, at the boundary where the claim enters the process, rather than
/// inside [`substitute_model`] -- keeping the selector pure. If NOTHING is
/// corroborated the original list is returned unchanged: that is far more
/// likely to mean the probe is broken than that every served model is a ghost.
fn drop_uncorroborated(entries: Vec<RosterEntry>) -> Vec<RosterEntry> {
    let kept: Vec<RosterEntry> = entries
        .iter()
        .filter(|e| disk_corroborated(&e.model))
        .cloned()
        .collect();
    if kept.is_empty() {
        entries
    } else {
        kept
    }
}

/// Return `/api/tags` entries, or `[]` if the server cannot be asked.
///
/// `[]` is deliberately indistinguishable from "server down": both mean we have
/// no evidence about what is installed, and callers must treat no-evidence as
/// "change nothing" rather than as "nothing is installed".
pub fn fetch_roster(host: &str, port: u16) -> Vec<RosterEntry> {
    let url = if host.contains("://") {
        format!("{}{API_TAGS}", host.trim_end_matches('/'))
    } else {
        format!("http://{host}:{port}{API_TAGS}")
    };
    let client = match reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(ROSTER_TIMEOUT_SECS))
        .build()
    {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    let resp = match client.get(&url).send() {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    if resp.status().as_u16() != 200 {
        return Vec::new();
    }
    let data: serde_json::Value = match resp.json() {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };
    let entries: Vec<RosterEntry> = data
        .get("models")
        .and_then(|m| m.as_array())
        .map(|arr| arr.iter().filter_map(RosterEntry::from_json).collect())
        .unwrap_or_default();
    drop_uncorroborated(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::thread;

    fn entry(model: &str, size: &str) -> RosterEntry {
        RosterEntry {
            model: model.to_string(),
            parameter_size: size.to_string(),
        }
    }

    #[test]
    fn missing_model_404_is_recognised_not_other_404s_or_statuses() {
        let body = r#"{"error": {"message": "Model 'X' is not installed or registered"}}"#;
        assert!(is_missing_model_error(404, body));
        assert!(!is_missing_model_error(404, "404 page not found"));
        assert!(!is_missing_model_error(503, "Server is at capacity"));
        assert!(!is_missing_model_error(404, ""));
    }

    #[test]
    fn parameter_billions_parses_b_m_and_garbage() {
        assert_eq!(parameter_billions(&entry("a", "27B")), 27.0);
        assert_eq!(parameter_billions(&entry("a", "4M")), 0.004);
        assert_eq!(parameter_billions(&entry("a", "")), 0.0);
        assert_eq!(parameter_billions(&entry("a", "junk")), 0.0);
    }

    #[test]
    fn empty_roster_and_installed_model_substitute_nothing() {
        let (model, reason) = substitute_model("gone-70b", &[], &default_fallback_chain());
        assert_eq!(model, "gone-70b");
        assert!(reason.is_none());

        let roster = vec![entry("live-model", "8B")];
        let (model, reason) = substitute_model("live-model", &roster, &[]);
        assert_eq!(model, "live-model");
        assert!(reason.is_none());
    }

    #[test]
    fn same_family_prefers_the_largest_installed_model() {
        let roster = vec![entry("gemma-4-e2b-it-4bit", "4B"), entry("gemma-4-12b-it-mxfp8", "12B")];
        let (model, reason) =
            substitute_model("gemma-4-99b-it-mxfp8", &roster, &default_fallback_chain());
        assert_eq!(model, "gemma-4-12b-it-mxfp8");
        let reason = reason.expect("substitution happened");
        assert!(reason.contains("largest installed 'gemma' model"), "{reason}");
    }

    #[test]
    fn no_family_match_falls_through_to_the_preference_chain_then_biggest() {
        let roster = vec![
            entry("qwen3.6-35b-a3b-mxfp8", "35B"),
            entry("laguna-70b", "70B"),
        ];
        // No laguna-family configured name exists; chain names foundation first,
        // then qwopus/qwen/gemma/nemotron/laguna -> qwen wins over laguna by order.
        let (model, _) = substitute_model("ghost-70b", &roster, &default_fallback_chain());
        assert_eq!(model, "qwen3.6-35b-a3b-mxfp8");

        // Chain exhausted: biggest model on the roster.
        let (model, reason) = substitute_model("ghost-70b", &roster, &["nothing-matches"]);
        assert_eq!(model, "laguna-70b");
        let reason = reason.expect("substitution happened");
        assert!(
            reason.contains("nothing in the preference chain"),
            "{reason}"
        );
    }

    #[test]
    #[serial_test::serial]
    fn hf_snapshot_layout_three_levels_deep_is_found() {
        // REGRESSION: the walk used to stop at two directory levels, so the
        // real HF layout hub/models--org--model/snapshots/<sha>/config.json
        // was invisible -- corroboration dropped servable HF-cache models.
        let dir = tempfile::tempdir().unwrap();
        let snap = dir.path().join("hub/models--org--model/snapshots/abc123");
        std::fs::create_dir_all(&snap).unwrap();
        std::fs::write(snap.join("config.json"), "{}").unwrap();
        let prev = std::env::var_os("MLX_MODELS_DIR");
        let prev_hf = std::env::var_os("HF_HOME");
        std::env::set_var("MLX_MODELS_DIR", dir.path().join("empty-mlx"));
        std::env::set_var("HF_HOME", dir.path());
        let found = model_config_path("model");
        match prev {
            Some(v) => std::env::set_var("MLX_MODELS_DIR", v),
            None => std::env::remove_var("MLX_MODELS_DIR"),
        }
        match prev_hf {
            Some(v) => std::env::set_var("HF_HOME", v),
            None => std::env::remove_var("HF_HOME"),
        }
        assert!(found.is_some(), "three-level snapshot must be found");
    }

    #[test]
    fn size_tiebreak_beats_name_sorting() {
        // "qwen3.10" sorts below "qwen3.8" alphabetically but is bigger.
        let roster = vec![entry("qwen3.8-8b", "8B"), entry("qwen3.10-27b", "27B")];
        let (model, _) = substitute_model("qwen-gone", &roster, &[]);
        assert_eq!(model, "qwen3.10-27b");
    }

    /// Isolates every disk/config seam from the operator's real machine
    /// (~/MLXModels and the checkout's conf/ both exist here) and restores
    /// whatever was set before.
    struct DiskGuard {
        saved: Vec<(&'static str, Option<std::ffi::OsString>)>,
        _dir: tempfile::TempDir,
    }

    impl DiskGuard {
        fn new() -> Self {
            let keys = ["MLX_MODELS_DIR", "HF_HOME", "ZTOOLS_CONF_DIR"];
            let saved = keys.iter().map(|k| (*k, std::env::var_os(k))).collect();
            let dir = tempfile::tempdir().unwrap();
            std::fs::create_dir_all(dir.path().join("mlx")).unwrap();
            std::fs::create_dir_all(dir.path().join("conf")).unwrap();
            std::env::set_var("MLX_MODELS_DIR", dir.path().join("mlx"));
            std::env::set_var("HF_HOME", dir.path().join("hf"));
            std::env::set_var("ZTOOLS_CONF_DIR", dir.path().join("conf"));
            Self { saved, _dir: dir }
        }

        fn conf_dir(&self) -> std::path::PathBuf {
            self._dir.path().join("conf")
        }

        fn write_family_toml(&self, family: &str, content: &str) {
            let path = self.conf_dir().join("models").join(format!("{family}.toml"));
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, content).unwrap();
        }
    }

    impl Drop for DiskGuard {
        fn drop(&mut self) {
            for (key, prev) in self.saved.drain(..) {
                match prev {
                    Some(v) => std::env::set_var(key, v),
                    None => std::env::remove_var(key),
                }
            }
        }
    }

    #[test]
    fn roster_entry_from_json_accepts_and_rejects_precisely() {
        use serde_json::json;
        let full = json!({"model": "m-7b", "details": {"parameter_size": "7B"}});
        assert_eq!(RosterEntry::from_json(&full), Some(entry("m-7b", "7B")));

        // details absent entirely: the entry still counts, size just unknown.
        let bare = json!({"model": "m"});
        assert_eq!(RosterEntry::from_json(&bare), Some(entry("m", "")));

        // A non-string parameter_size must not poison the entry.
        let numeric = json!({"model": "m", "details": {"parameter_size": 7}});
        assert_eq!(RosterEntry::from_json(&numeric), Some(entry("m", "")));

        // No usable model name: no entry at all.
        assert_eq!(RosterEntry::from_json(&json!({"details": {}})), None);
        assert_eq!(RosterEntry::from_json(&json!({"model": ""})), None);
        assert_eq!(RosterEntry::from_json(&json!({"model": 42})), None);
    }

    #[test]
    fn parameter_billions_k_suffix_unknown_suffixes_and_parse_failures() {
        assert!(
            (parameter_billions(&entry("a", "640K")) - 0.00064).abs() < 1e-12,
            "640K is 0.00064B"
        );
        assert_eq!(parameter_billions(&entry("a", "27X")), 0.0, "unknown suffix");
        assert_eq!(parameter_billions(&entry("a", "junk!")), 0.0, "non-numeric body");
        assert_eq!(parameter_billions(&entry("a", " 12B ")), 12.0, "surrounding whitespace");
        assert_eq!(parameter_billions(&entry("a", "12.5B")), 12.5, "fractional sizes");
    }

    #[test]
    fn substitute_model_family_miss_falls_through_to_the_chain() {
        // 'qwen-gone' resolves to the qwen family but no qwen model is served:
        // the family arm finds nothing, so the chain decides.
        let roster = vec![entry("laguna-70b", "70B")];
        let (model, reason) = substitute_model("qwen-gone-27b", &roster, &["laguna"]);
        assert_eq!(model, "laguna-70b");
        let reason = reason.expect("substitution happened");
        assert!(
            reason.contains("no 'qwen' model is either"),
            "{reason}"
        );
    }

    #[test]
    fn substitute_model_prefers_earlier_chain_entries_over_bigger_models() {
        let roster = vec![entry("gemma-2b", "2B"), entry("nemotron-90b", "90B")];
        let (model, reason) =
            substitute_model("ghost-model", &roster, &["gemma", "nemotron"]);
        assert_eq!(model, "gemma-2b", "chain order beats roster size");
        assert!(reason.unwrap().contains("falling back to 'gemma-2b'"));
    }

    #[test]
    fn walk_configs_finds_all_three_nesting_levels_and_skips_the_rest() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("config.json"), "{}").unwrap();
        std::fs::create_dir_all(root.join("a/b")).unwrap();
        std::fs::write(root.join("a/config.json"), "{}").unwrap();
        std::fs::write(root.join("a/b/config.json"), "{}").unwrap();
        std::fs::write(root.join("loose.txt"), "not a config").unwrap();
        std::fs::create_dir_all(root.join("empty-dir")).unwrap();

        let mut found = walk_configs(root).unwrap();
        found.sort();
        assert_eq!(found.len(), 3, "{found:?}");
        assert_eq!(found[0], root.join("a/b/config.json"));
        assert_eq!(found[1], root.join("a/config.json"));
        assert_eq!(found[2], root.join("config.json"));
    }

    #[test]
    #[serial]
    fn empty_model_names_have_no_config_path() {
        let _guard = DiskGuard::new();
        assert_eq!(model_config_path(""), None);
    }

    #[test]
    #[serial]
    fn mlx_layout_matches_case_insensitively_on_the_directory_name() {
        let guard = DiskGuard::new();
        let model_dir = guard._dir.path().join("mlx/TestOrg/LiveModel");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("config.json"), "{}").unwrap();

        // Served ids are lowercased; directories keep their case.
        let found = model_config_path("livemodel")
            .expect("case-insensitive directory match must find the config");
        assert_eq!(found, model_dir.join("config.json"));

        // A name nothing on disk backs up.
        assert_eq!(model_config_path("ghost-model"), None);
    }

    #[test]
    #[serial]
    fn missing_roots_are_skipped_not_errors() {
        let dir = tempfile::tempdir().unwrap();
        let keys = ["MLX_MODELS_DIR", "HF_HOME"];
        // MLX pre-set (sentinel -> was-set restore arm), HF removed before
        // capture (never-set restore arm): both arms covered deterministically.
        std::env::set_var("MLX_MODELS_DIR", "/nonexistent-sentinel");
        std::env::remove_var("HF_HOME");
        let saved: Vec<(&'static str, Option<std::ffi::OsString>)> =
            keys.iter().map(|k| (*k, std::env::var_os(k))).collect();
        struct Restore(Vec<(&'static str, Option<std::ffi::OsString>)>);
        impl Drop for Restore {
            fn drop(&mut self) {
                for (k, v) in self.0.drain(..) {
                    match v {
                        Some(v) => std::env::set_var(k, v),
                        None => std::env::remove_var(k),
                    }
                }
            }
        }
        let _restore = Restore(saved);
        std::env::set_var("MLX_MODELS_DIR", dir.path().join("nope"));
        std::env::set_var("HF_HOME", dir.path().join("also-nope"));
        assert_eq!(
            model_config_path("anything"),
            None,
            "absent roots mean not-found, never a panic"
        );

        // A root that exists but cannot be read is skipped the same way --
        // an unreadable probe is not evidence of absence.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let locked = dir.path().join("locked-root/Org/LiveModel");
            std::fs::create_dir_all(&locked).unwrap();
            std::fs::write(locked.join("config.json"), "{}").unwrap();
            std::fs::set_permissions(
                dir.path().join("locked-root"),
                std::fs::Permissions::from_mode(0o000),
            )
            .unwrap();
            std::env::set_var("MLX_MODELS_DIR", dir.path().join("locked-root"));
            let found = model_config_path("livemodel");
            let _ = std::fs::set_permissions(
                dir.path().join("locked-root"),
                std::fs::Permissions::from_mode(0o755),
            );
            assert_eq!(found, None, "an unreadable root yields nothing, not a panic");
        }
    }

    #[test]
    #[serial]
    fn conf_models_root_prefers_env_then_the_checkout_then_a_relative_path() {
        let keys = ["ZTOOLS_CONF_DIR", "HOME"];
        // ZTOOLS pre-set (was-set restore arm); HOME removed first
        // (never-set restore arm).
        std::env::set_var("ZTOOLS_CONF_DIR", "/nonexistent-sentinel");
        std::env::remove_var("HOME");
        let saved: Vec<(&'static str, Option<std::ffi::OsString>)> =
            keys.iter().map(|k| (*k, std::env::var_os(k))).collect();
        struct Restore(Vec<(&'static str, Option<std::ffi::OsString>)>);
        impl Drop for Restore {
            fn drop(&mut self) {
                for (k, v) in self.0.drain(..) {
                    match v {
                        Some(v) => std::env::set_var(k, v),
                        None => std::env::remove_var(k),
                    }
                }
            }
        }
        let _restore = Restore(saved);

        std::env::set_var("ZTOOLS_CONF_DIR", "/fixture-conf");
        assert_eq!(
            conf_models_root(),
            PathBuf::from("/fixture-conf").join("models"),
            "the env seam wins"
        );

        std::env::remove_var("ZTOOLS_CONF_DIR");
        // A checkout under HOME with a conf/models tree: the home branch.
        let fake_home = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(
            fake_home.path().join("Projects/ztools/conf/models"),
        )
        .unwrap();
        std::env::set_var("HOME", fake_home.path());
        assert_eq!(
            conf_models_root(),
            fake_home.path().join("Projects/ztools/conf/models")
        );

        // An EMPTY HOME: no checkout there, so the documented relative
        // fallback stands. (A truly absent home is not simulatable: home_dir
        // falls back to the passwd entry when HOME is empty or unset.)
        let empty_home = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", empty_home.path());
        assert_eq!(conf_models_root(), PathBuf::from("conf/models"));
    }

    #[test]
    #[serial]
    fn hf_cache_layout_is_recognised_by_its_models_directory_component() {
        let guard = DiskGuard::new();
        let hub = guard._dir.path().join("hf/hub");
        let model_dir = hub.join("models--TestOrg--TestModel");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("config.json"), "{}").unwrap();
        std::env::remove_var("MLX_MODELS_DIR");

        let found = model_config_path("testmodel")
            .expect("the models-- component must be matched case-insensitively");
        assert_eq!(found, model_dir.join("config.json"));
        assert_eq!(model_config_path("othermodel"), None);
    }

    #[test]
    #[serial]
    fn documented_context_window_found_not_found_malformed_or_nonpositive() {
        let guard = DiskGuard::new();

        // Unknown family: no file is ever consulted.
        assert_eq!(documented_context_window("ghost-model"), None);

        guard.write_family_toml("foundation", "context_window = 4096\n");
        assert_eq!(documented_context_window("foundation-something"), Some(4096));

        // Known family but no file for it.
        assert_eq!(documented_context_window("qwen3.8-27b"), None);

        guard.write_family_toml("gemma", "{{{ not toml");
        assert_eq!(documented_context_window("gemma-4-e2b"), None, "malformed toml");

        for content in ["context_window = 0\n", "context_window = -5\n"] {
            guard.write_family_toml("nemotron", content);
            assert_eq!(documented_context_window("nemotron-x"), None, "{content}");
        }

        guard.write_family_toml("laguna", "context_window = \"4096\"\n");
        assert_eq!(documented_context_window("laguna-x"), None, "string is no window");
    }

    #[test]
    #[serial]
    fn generative_verdict_comes_from_the_config_not_the_name() {
        let guard = DiskGuard::new();
        let put = |name: &str, content: &str| {
            let d = guard._dir.path().join("mlx/Org").join(name);
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("config.json"), content).unwrap();
        };

        // Nothing on disk: assume generative rather than silently skipping a
        // model the user installed.
        assert!(is_generative_model("ghost-model"));

        put("Embedder", r#"{"model_type": "Model2Vec"}"#);
        assert!(!is_generative_model("embedder"), "type check is case-insensitive");

        put("StaticArch", r#"{"architectures": ["StaticModel"]}"#);
        assert!(!is_generative_model("staticarch"));

        put("SentTrans", r#"{"architectures": ["SentenceTransformer"]}"#);
        assert!(!is_generative_model("senttrans"));

        put("RealModel", r#"{"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}"#);
        assert!(is_generative_model("realmodel"));

        put("NoArch", r#"{"model_type": "whatever"}"#);
        assert!(is_generative_model("noarch"));

        put("BrokenJson", "{not json");
        assert!(is_generative_model("brokenjson"), "unreadable-as-json keeps probing");

        // An unreadable FILE also keeps probing: same verdict as missing.
        let locked = guard._dir.path().join("mlx/Org/Locked/config.json");
        std::fs::create_dir_all(locked.parent().unwrap()).unwrap();
        std::fs::write(&locked, "{}").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
        }
        assert!(is_generative_model("locked"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o644));
        }
    }

    #[test]
    #[serial]
    fn corroboration_accepts_disk_configs_or_documented_windows_only() {
        let guard = DiskGuard::new();
        assert!(!disk_corroborated("ghost-model"), "nothing on disk backs it");

        guard.write_family_toml("foundation", "context_window = 4096\n");
        assert!(
            disk_corroborated("foundation-x"),
            "a documented window corroborates without any disk config"
        );

        let model_dir = guard._dir.path().join("mlx/Org/DiskModel");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("config.json"), "{}").unwrap();
        assert!(disk_corroborated("diskmodel"));
    }

    #[test]
    #[serial]
    fn drop_uncorroborated_filters_ghosts_but_never_empties_a_roster() {
        let guard = DiskGuard::new();
        let model_dir = guard._dir.path().join("mlx/Org/DiskModel");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("config.json"), "{}").unwrap();

        let roster = vec![
            entry("ghost-a", "7B"),
            entry("diskmodel", "8B"),
            entry("ghost-b", "70B"),
        ];
        let kept = drop_uncorroborated(roster.clone());
        assert_eq!(kept, vec![entry("diskmodel", "8B")], "ghosts are dropped");

        // Nothing survives: the ORIGINAL list comes back -- a fully-ghost
        // roster far more likely means a broken probe than zero models.
        let all_ghost = vec![entry("ghost-a", "7B"), entry("ghost-b", "70B")];
        assert_eq!(drop_uncorroborated(all_ghost.clone()), all_ghost);
    }

    /// One-shot localhost HTTP mock for fetch_roster.
    fn serve_roster(body: &'static str, status_line: &'static str) -> (u16, thread::JoinHandle<()>) {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = vec![0u8; 65_536];
                let _ = stream.read(&mut buf);
                let response = format!(
                    "{status_line}\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });
        thread::sleep(std::time::Duration::from_millis(50));
        (port, handle)
    }

    #[test]
    #[serial]
    fn fetch_roster_keeps_disk_backed_entries_and_drops_ghosts_over_the_wire() {
        let guard = DiskGuard::new();
        let model_dir = guard._dir.path().join("mlx/Org/DiskModel");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("config.json"), "{}").unwrap();

        let body = r#"{"models":[
            {"model":"ghost-a","details":{"parameter_size":"70B"}},
            {"model":"DiskModel","details":{"parameter_size":"8B"}}
        ]}"#;
        let (port, handle) = serve_roster(body, "HTTP/1.1 200 OK");

        // host + port form builds the http:// URL itself.
        let roster = fetch_roster("127.0.0.1", port);
        handle.join().unwrap();
        assert_eq!(roster, vec![entry("DiskModel", "8B")], "the ghost must not survive");
    }

    #[test]
    #[serial]
    fn fetch_roster_accepts_a_full_url_host() {
        let _guard = DiskGuard::new();
        let body = r#"{"models":[{"model":"any-model"}]}"#;
        let (port, handle) = serve_roster(body, "HTTP/1.1 200 OK");
        let host = format!("http://127.0.0.1:{port}");
        let roster = fetch_roster(&host, 0);
        handle.join().unwrap();
        // Not corroborated by anything on disk -> dropped here; the point of
        // this test is that the scheme-form URL reaches the server at all,
        // which an empty roster would silently hide.
        let (port2, handle2) = serve_roster(body, "HTTP/1.1 200 OK");
        let host2 = format!("http://127.0.0.1:{port2}/");
        let roster2 = fetch_roster(&host2, 0);
        handle2.join().unwrap();
        assert_eq!(roster, roster2, "trailing slash is trimmed");
    }

    #[test]
    #[serial]
    fn fetch_roster_answers_empty_when_the_server_cannot_be_asked() {
        let _guard = DiskGuard::new();

        // Non-200 status.
        let (port, handle) = serve_roster("{}", "HTTP/1.1 503 Service Unavailable");
        assert!(fetch_roster("127.0.0.1", port).is_empty());
        handle.join().unwrap();

        // 200 with unparseable JSON.
        let (port, handle) = serve_roster("{not json", "HTTP/1.1 200 OK");
        assert!(fetch_roster("127.0.0.1", port).is_empty());
        handle.join().unwrap();

        // 200 with JSON lacking a models array.
        let (port, handle) = serve_roster(r#"{"other": []}"#, "HTTP/1.1 200 OK");
        assert!(fetch_roster("127.0.0.1", port).is_empty());
        handle.join().unwrap();

        // Connection refused: a bound-then-dropped port.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        assert!(fetch_roster("127.0.0.1", port).is_empty(), "down server == no evidence");
    }
}
