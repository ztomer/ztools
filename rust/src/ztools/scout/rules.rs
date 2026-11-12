//! The four filters of LM.DESIGN section 2, in order.
//!
//! Each is a pure function of what the feeds returned: provenance, the format Osaurus loads, memory fit at the
//! house context, and the disk floor. Every one answers [`Check`], so a rule
//! that could not be judged says `unverified` with why, never a guessed pass.

use serde_json::Value;
use std::collections::BTreeSet;

/// The publishers a candidate may come from (D7 / section 2.1). Adding one needs
/// the owner's one-line ruling.
pub const ALLOWLIST: [&str; 7] = [
    "dealignai",
    "OsaurusAI",
    "mlx-community",
    "lmstudio-community",
    "Jackrong",
    "google",
    "LiquidAI",
];

pub const GIB: u64 = 1 << 30;
/// The design point: house prompts are about 8k tokens, so 32k is 2x headroom.
pub const CTX_TOKENS: u64 = 32_768;
/// The runtime's own share on top of weights and KV.
pub const RUNTIME_BYTES: u64 = 4 * GIB;
/// The share of Osaurus's `gpu_budget` a model may plan to use.
pub const FIT_SHARE_PERCENT: u64 = 85;
/// What must stay free on the models volume after a download.
pub const DISK_FLOOR_BYTES: u64 = 15 * GIB;
/// At most this many candidates staged at once.
pub const MAX_STAGED: usize = 1;

/// One rule's verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Check {
    Pass(String),
    Fail(String),
    /// The rule could not be judged: a feed refused or a field is missing.
    Unverified(String),
}

impl Check {
    #[must_use]
    pub const fn glyph(&self) -> &'static str {
        match self {
            Self::Pass(_) => "✓",
            Self::Fail(_) => "✗",
            Self::Unverified(_) => "⚠",
        }
    }

    #[must_use]
    pub fn text(&self) -> &str {
        match self {
            Self::Pass(t) | Self::Fail(t) | Self::Unverified(t) => t,
        }
    }

    #[must_use]
    pub const fn passed(&self) -> bool {
        matches!(self, Self::Pass(_))
    }
}

/// The publisher of `repo`: everything before the first `/`.
#[must_use]
pub fn author(repo: &str) -> &str {
    repo.split_once('/').map_or(repo, |(a, _)| a)
}

/// The repo name: everything after the first `/`.
#[must_use]
pub fn name(repo: &str) -> &str {
    repo.split_once('/').map_or(repo, |(_, n)| n)
}

/// Provenance: the publisher is on the allowlist (exact case, as HF ids are).
/// A repo of another publisher carrying an allowlisted repo's name is a mirror.
#[must_use]
pub fn provenance(repo: &str, allowlisted_names: &BTreeSet<String>) -> Check {
    let who = author(repo);
    if ALLOWLIST.contains(&who) {
        return Check::Pass(format!("{who} is allowlisted"));
    }
    let lower = name(repo).to_lowercase();
    if allowlisted_names.contains(&lower) {
        Check::Fail(format!(
            "mirror: {who} re-uploads an allowlisted repo's name"
        ))
    } else {
        Check::Fail(format!("{who} is not on the publisher allowlist"))
    }
}

/// One file of a repo's tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct File {
    pub path: String,
    pub size: u64,
}

/// The weights a load maps: every `*.safetensors` file.
#[must_use]
pub fn weights_bytes(files: &[File]) -> u64 {
    files
        .iter()
        .filter(|f| has_ext(&f.path, "safetensors"))
        .map(|f| f.size)
        .sum()
}

/// What a download writes: every file.
#[must_use]
pub fn download_bytes(files: &[File]) -> u64 {
    files.iter().map(|f| f.size).sum()
}

fn has_ext(path: &str, ext: &str) -> bool {
    std::path::Path::new(path)
        .extension()
        .is_some_and(|x| x.eq_ignore_ascii_case(ext))
}

fn has(files: &[File], file: &str) -> bool {
    files.iter().any(|f| f.path == file)
}

/// Quantisation methods that are not MLX's and Osaurus 1.0.1 is not shown to load.
const FOREIGN_METHODS: [&str; 6] = ["gptq", "awq", "fp8", "exl2", "exl3", "bitsandbytes"];
/// MLX quantisation modes Osaurus 1.0.1 loads.
const MLX_MODES: [&str; 3] = ["affine", "mxfp8", "mxfp4"];

/// The `model_type` a config names at its top level.
#[must_use]
pub fn model_type(config: &Value) -> Option<&str> {
    config.get("model_type").and_then(Value::as_str)
}

fn scalar<'a>(config: &'a Value, table: &str, key: &str) -> Option<&'a Value> {
    config.get(table).and_then(|t| t.get(key))
}

/// The format: a model type Osaurus serves, in a quantisation it loads.
/// `served` is the set of model types shown to load (the Osaurus catalog's own
/// and every installed model's).
#[must_use]
pub fn format(files: &[File], config: &Value, served: &BTreeSet<String>) -> Check {
    if files.iter().any(|f| has_ext(&f.path, "gguf")) {
        return Check::Fail("GGUF".into());
    }
    if weights_bytes(files) == 0 {
        return Check::Fail("no safetensors weights".into());
    }
    let Some(kind) = model_type(config) else {
        return Check::Unverified("config.json names no model_type".into());
    };
    if !served.contains(kind) {
        return Check::Fail(format!(
            "model_type {kind} is not one Osaurus is shown to load"
        ));
    }
    if let Some(method) =
        scalar(config, "quantization_config", "quant_method").and_then(Value::as_str)
        && FOREIGN_METHODS.contains(&method.to_lowercase().as_str())
    {
        return Check::Fail(format!("{method} quantisation"));
    }
    let mode = scalar(config, "quantization", "mode").and_then(Value::as_str);
    let bits = scalar(config, "quantization", "bits").and_then(Value::as_u64);
    let family = if has(files, "jang_config.json") {
        "JANG"
    } else if has(files, "optiq_metadata.json") {
        "OptiQ"
    } else {
        "MLX"
    };
    let detail = match (mode, bits) {
        (Some(m), _) if !MLX_MODES.contains(&m) => {
            return Check::Fail(format!("{m} quantisation"));
        }
        (Some("affine") | None, Some(b)) => format!("{b}-bit"),
        (Some(m), _) => m.to_string(),
        (None, None) if family == "MLX" => return Check::Pass(format!("unquantized {kind}")),
        (None, None) => "unstated bits".to_string(),
    };
    Check::Pass(format!("{family} {detail} {kind}"))
}

/// The text model's config: `text_config` when the model wraps one.
fn text(config: &Value) -> &Value {
    config.get("text_config").unwrap_or(config)
}

fn num(v: &Value, key: &str) -> Option<u64> {
    v.get(key).and_then(Value::as_u64)
}

/// The fp16 KV cache upper bound at `ctx` tokens.
///
/// `2 * n_full_layers * kv_heads * head_dim * 2 B * ctx`, plus each sliding
/// layer at its window. Linear-attention layers keep a fixed state the runtime
/// share covers.
///
/// # Errors
///
/// The field the config lacks.
pub fn kv_bytes(config: &Value, ctx: u64) -> Result<u64, String> {
    let t = text(config);
    let need = |k: &str| num(t, k).ok_or_else(|| format!("config.json has no {k}"));
    let layers = need("num_hidden_layers")?;
    let kv_heads = need("num_key_value_heads")?;
    let head_dim = match num(t, "head_dim") {
        Some(d) => d,
        None => need("hidden_size")? / need("num_attention_heads")?.max(1),
    };
    let global_heads = num(t, "num_global_key_value_heads").unwrap_or(kv_heads);
    let global_dim = num(t, "global_head_dim").unwrap_or(head_dim);
    let types: Vec<&str> = t
        .get("layer_types")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let (full, sliding) = if types.is_empty() {
        match num(t, "full_attention_interval") {
            Some(n) if n > 0 => (layers / n, 0),
            _ => (layers, 0),
        }
    } else {
        let count = |want: &str| types.iter().filter(|x| **x == want).count() as u64;
        (count("full_attention"), count("sliding_attention"))
    };
    let window = num(t, "sliding_window").unwrap_or(ctx).min(ctx);
    let per_token = |heads: u64, dim: u64| 2 * heads * dim * 2;
    Ok(full * per_token(global_heads, global_dim) * ctx
        + sliding * per_token(kv_heads, head_dim) * window)
}

/// `n` bytes in GiB, one decimal.
#[must_use]
pub fn gib(n: u64) -> String {
    format!(
        "{:.1}",
        crate::units::unsigned(n) / crate::units::unsigned(GIB)
    )
}

/// Memory fit: `weights + KV(32k) + 4 GiB <= 0.85 * gpu_budget`. Osaurus's own
/// `ram_feasibility` verdict is the second half of the rule and is only known
/// at load time, so a pass says it is still to come.
#[must_use]
pub fn fit(weights: u64, config: &Value, gpu_budget: Option<u64>) -> Check {
    let Some(budget) = gpu_budget else {
        return Check::Unverified("no gpu_budget: Osaurus /health did not answer".into());
    };
    let kv = match kv_bytes(config, CTX_TOKENS) {
        Ok(kv) => kv,
        Err(why) => return Check::Unverified(why),
    };
    let need = weights + kv + RUNTIME_BYTES;
    let line = budget / 100 * FIT_SHARE_PERCENT;
    let sum = format!(
        "{} + KV {} + 4 = {} GiB vs {} GiB",
        gib(weights),
        gib(kv),
        gib(need),
        gib(line)
    );
    if need <= line {
        Check::Pass(format!("{sum} (ram_feasibility at load)"))
    } else {
        Check::Fail(format!("{sum}: over the line"))
    }
}

/// The disk rule: free after the download stays at or above the floor, and no
/// other candidate is staged.
#[must_use]
pub fn disk(download: u64, free: Option<u64>, staged: usize) -> Check {
    let Some(free) = free else {
        return Check::Unverified("free space on the models volume unknown".into());
    };
    if staged >= MAX_STAGED {
        return Check::Fail(format!(
            "{staged} candidate already staged (at most {MAX_STAGED})"
        ));
    }
    let after = free.saturating_sub(download);
    let sum = format!(
        "{} GiB download leaves {} of {} GiB free",
        gib(download),
        gib(after),
        gib(free)
    );
    if download <= free && after >= DISK_FLOOR_BYTES {
        Check::Pass(sum)
    } else {
        Check::Fail(format!(
            "{sum}: under the {} GiB floor",
            gib(DISK_FLOOR_BYTES)
        ))
    }
}

#[cfg(test)]
#[path = "rules_tests.rs"]
mod tests;
