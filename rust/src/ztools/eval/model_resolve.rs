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
fn model_config_path(model: &str) -> Option<PathBuf> {
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

/// config.json files under `root`, two levels deep at most (the shapes above
/// are the only ones that occur; a full recursive walk buys nothing here).
fn walk_configs(root: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut found = Vec::new();
    let direct = root.join("config.json");
    if direct.is_file() {
        found.push(direct);
    }
    for entry in std::fs::read_dir(root)?.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let nested = path.join("config.json");
        if nested.is_file() {
            found.push(nested);
        }
        for sub in std::fs::read_dir(&path)?.flatten() {
            let sub_path = sub.path();
            let deep = sub_path.join("config.json");
            if sub_path.is_dir() && deep.is_file() {
                found.push(deep);
            }
        }
    }
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
    fn size_tiebreak_beats_name_sorting() {
        // "qwen3.10" sorts below "qwen3.8" alphabetically but is bigger.
        let roster = vec![entry("qwen3.8-8b", "8B"), entry("qwen3.10-27b", "27B")];
        let (model, _) = substitute_model("qwen-gone", &roster, &[]);
        assert_eq!(model, "qwen3.10-27b");
    }
}
