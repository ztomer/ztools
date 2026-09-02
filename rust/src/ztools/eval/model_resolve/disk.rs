//! Filesystem probing: model config location, generative-ness, disk corroboration.
//!
//! Split out of model_resolve.rs for the 500-line cap.

use std::path::{Path, PathBuf};

pub(super) fn models_dir() -> PathBuf {
    std::env::var("MLX_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| dirs::home_dir().unwrap_or_default().join("MLXModels"))
}

pub(super) fn hf_cache_dir() -> PathBuf {
    let home = std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap_or_default()
                .join(".cache/huggingface")
        });
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
pub(super) fn walk_configs(root: &Path) -> std::io::Result<Vec<PathBuf>> {
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
pub(super) fn documented_context_window(model: &str) -> Option<u64> {
    let lower = model.to_lowercase();
    let family = crate::ztools::eval::quirks::MODEL_FAMILIES
        .iter()
        .find(|f| lower.contains(*f))?;
    let content =
        std::fs::read_to_string(conf_models_root().join(format!("{family}.toml"))).ok()?;
    let val: toml::Value = toml::from_str(&content).ok()?;
    val.get("context_window")
        .and_then(|w| w.as_integer())
        .filter(|w| *w > 0)
        .map(|w| w as u64)
}

/// Where `conf/models/` lives: the repo checkout beside the compiled-in
/// candidates, same convention as `ZtoolsConfig`'s conf lookups.
pub(super) fn conf_models_root() -> PathBuf {
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
