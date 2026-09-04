//! Model packaging and health diagnostics.
//!
//! Offline detection of packaging defects, unaccelerated MTP speculative
//! drafting shards, missing safetensor shards, and incomplete downloads.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub const THRASHING_DECODE_TOKENS_PER_SEC: f64 = 1.0;

/// Locate the directory for a given model name under MLXModels or HF cache.
pub fn find_model_dir(model_name: &str, base_dir: Option<&Path>) -> Option<PathBuf> {
    let root = if let Some(d) = base_dir {
        d.to_path_buf()
    } else {
        dirs::home_dir()?.join("MLXModels")
    };

    if !root.exists() {
        return None;
    }

    let target = model_name.to_lowercase().replace('/', "-");

    // 1. Direct path check
    let direct = root.join(model_name);
    if direct.is_dir() {
        return Some(direct);
    }

    // 2. Search subdirectories (e.g. MLXModels/<org>/<model>)
    if let Ok(entries) = std::fs::read_dir(&root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = entry.file_name().to_string_lossy().to_lowercase();
                if name == target || name.contains(&target) || target.contains(&name) {
                    return Some(path);
                }
                // Check one level deeper (org/model)
                if let Ok(sub_entries) = std::fs::read_dir(&path) {
                    for sub in sub_entries.flatten() {
                        let sub_path = sub.path();
                        if sub_path.is_dir() {
                            let sub_name = sub.file_name().to_string_lossy().to_lowercase();
                            if sub_name == target
                                || sub_name.contains(&target)
                                || target.contains(&sub_name)
                            {
                                return Some(sub_path);
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

/// Inspect a model directory for packaging defects without loading weights.
pub fn probe_model_dir_defects(dir: &Path) -> Vec<String> {
    let mut defects = Vec::new();

    // 1. Check for unaccelerated / unsupported MTP speculative shards
    let mut mtp_shards = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let fname = entry.file_name().to_string_lossy().to_string();
            if fname.contains("mtp") && fname.ends_with(".safetensors") {
                mtp_shards.push(fname);
            }
        }
    }

    let jang_config_path = dir.join("jang_config.json");
    if jang_config_path.is_file() {
        if let Ok(content) = std::fs::read_to_string(&jang_config_path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                let runtime_avail = json
                    .get("runtime_available")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                let mtp_mode = json.get("mtp_mode").and_then(|v| v.as_str()).unwrap_or("");

                if (!runtime_avail || mtp_mode == "preserved_enabled") && !mtp_shards.is_empty() {
                    defects.push(format!(
                        "unsupported MTP speculative drafting shards present with runtime_available=false ({})",
                        mtp_shards.join(", ")
                    ));
                }
            }
        }
    } else if !mtp_shards.is_empty() {
        defects.push(format!(
            "unintegrated MTP speculative shard(s) present without runtime config: {}",
            mtp_shards.join(", ")
        ));
    }

    // 2. Check for missing safetensor shards referenced in index
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.is_file() {
        if let Ok(content) = std::fs::read_to_string(&index_path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(map) = json.get("weight_map").and_then(|v| v.as_object()) {
                    let mut expected = HashSet::new();
                    for shard_val in map.values() {
                        if let Some(s) = shard_val.as_str() {
                            expected.insert(s.to_string());
                        }
                    }
                    let mut missing = Vec::new();
                    for shard in &expected {
                        if !dir.join(shard).is_file() {
                            missing.push(shard.clone());
                        }
                    }
                    if !missing.is_empty() {
                        missing.sort();
                        defects.push(format!(
                            "missing {} safetensor shard(s) in index: {:?}",
                            missing.len(),
                            missing
                        ));
                    }
                }
            }
        }
    }

    // 3. Incomplete download artifacts (*.incomplete)
    let mut incomplete = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let fname = entry.file_name().to_string_lossy().to_string();
            if fname.ends_with(".incomplete") || fname.ends_with(".lock") {
                incomplete.push(fname);
            }
        }
    }
    let cache_dir = dir.join(".cache");
    if cache_dir.is_dir() {
        if let Ok(entries) = std::fs::read_dir(&cache_dir) {
            for entry in entries.flatten() {
                let fname = entry.file_name().to_string_lossy().to_string();
                if fname.ends_with(".incomplete") {
                    incomplete.push(format!(".cache/{}", fname));
                }
            }
        }
    }
    if !incomplete.is_empty() {
        defects.push(format!(
            "incomplete download artifacts ({} .incomplete file(s) present)",
            incomplete.len()
        ));
    }

    defects
}

/// Probe model defects given a model name and optional base directory.
pub fn probe_model_defects(model_name: &str, base_dir: Option<&Path>) -> Vec<String> {
    if let Some(dir) = find_model_dir(model_name, base_dir) {
        probe_model_dir_defects(&dir)
    } else {
        Vec::new()
    }
}

/// Assess model viability (defect check + decode thrashing check).
pub fn assess_viability(
    model_name: &str,
    decode_tok_per_sec: Option<f64>,
    base_dir: Option<&Path>,
) -> Result<(), String> {
    let defects = probe_model_defects(model_name, base_dir);
    if !defects.is_empty() {
        return Err(format!("broken: {}", defects.join("; ")));
    }

    if let Some(rate) = decode_tok_per_sec {
        if rate < THRASHING_DECODE_TOKENS_PER_SEC {
            return Err(format!(
                "unviable: decode rate {:.2} tok/s is below thrashing threshold ({:.1} tok/s)",
                rate, THRASHING_DECODE_TOKENS_PER_SEC
            ));
        }
    }

    Ok(())
}
