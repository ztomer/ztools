//! Where the scout reads from: the Hugging Face Hub API, the Osaurus catalog
//! (its org fetch on HF and the curated and retired lists in the Osaurus source
//! at the installed tag), and Osaurus's own `/health`.
//!
//! Every read is a plain unauthenticated GET: no token is ever sent, and no
//! request carries anything of this machine but the public repo ids it asks
//! about. Each body is REDUCED to the fields the scout reads before anything
//! looks at it, live or recorded, so a recorded fixture is exactly what the
//! live reader would have seen.

use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::path::Path;
use std::time::Duration;

/// The seam: one GET by its key, already reduced. `Err` is why the feed refused.
pub trait Feed {
    /// # Errors
    ///
    /// Why the feed refused: an HTTP status, a timeout, a body that is not JSON.
    fn get(&self, key: &str) -> Result<Value, String>;
}

pub const HF: &str = "https://huggingface.co";
const OSAURUS_SRC: &str = "https://raw.githubusercontent.com/osaurus-ai/osaurus";
const CATALOG_FILE: &str = "Packages/OsaurusCore/Managers/Model/ModelManager.swift";
/// The key of Osaurus's `/health`, which lives on the local server.
pub const HEALTH: &str = "osaurus:/health";

/// One model's card: `/api/models/<repo>`.
#[must_use]
pub fn model_key(repo: &str) -> String {
    format!("/api/models/{repo}")
}

/// An allowlisted publisher's members of one family.
#[must_use]
pub fn family_key(author: &str, base: &str) -> String {
    format!("/api/models?author={author}&filter=base_model:{base}&limit=100")
}

/// Every repo carrying a name, whoever publishes it: where mirrors show up.
#[must_use]
pub fn search_key(name: &str) -> String {
    format!("/api/models?search={name}&limit=50")
}

/// The Osaurus org fetch, as Osaurus itself makes it.
#[must_use]
pub fn org_key() -> String {
    "/api/models?author=OsaurusAI&sort=downloads&limit=200".into()
}

/// A repo's top-level files with sizes.
#[must_use]
pub fn tree_key(repo: &str) -> String {
    format!("/api/models/{repo}/tree/main")
}

/// A repo's `config.json`.
#[must_use]
pub fn config_key(repo: &str) -> String {
    format!("/{repo}/resolve/main/config.json")
}

/// The Osaurus catalog source at the installed version's tag.
#[must_use]
pub fn catalog_key(version: &str) -> String {
    format!("catalog:{version}")
}

fn pick(v: &Value, keys: &[&str]) -> Value {
    let mut out = serde_json::Map::new();
    for k in keys {
        if let Some(x) = v.get(*k)
            && !x.is_object()
        {
            out.insert((*k).to_string(), x.clone());
        }
    }
    Value::Object(out)
}

fn base_tags(v: &Value) -> Value {
    let tags: Vec<Value> = v
        .get("tags")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|t| t.as_str().is_some_and(|s| s.starts_with("base_model:")))
        .cloned()
        .collect();
    Value::Array(tags)
}

fn listing_row(v: &Value) -> Value {
    let mut row = pick(v, &["id", "pipeline_tag", "createdAt", "downloads"]);
    row["tags"] = base_tags(v);
    row
}

const TEXT_KEYS: [&str; 12] = [
    "model_type",
    "num_hidden_layers",
    "num_key_value_heads",
    "num_attention_heads",
    "hidden_size",
    "head_dim",
    "global_head_dim",
    "num_global_key_value_heads",
    "layer_types",
    "full_attention_interval",
    "sliding_window",
    "max_position_embeddings",
];

/// The fields of a `config.json` the format and fit rules read.
#[must_use]
pub fn reduce_config(v: &Value) -> Value {
    let mut out = pick(v, &TEXT_KEYS);
    for table in ["quantization", "quantization_config", "text_config"] {
        if let Some(t) = v.get(table)
            && t.is_object()
        {
            let keys: &[&str] = if table == "text_config" {
                &TEXT_KEYS
            } else {
                &["mode", "bits", "group_size", "quant_method"]
            };
            out[table] = pick(t, keys);
        }
    }
    out
}

/// Reduce a body by what its key asks for.
#[must_use]
pub fn reduce(key: &str, v: &Value) -> Value {
    if key == HEALTH {
        return json!({
            "gpu_budget_bytes": v
                .pointer("/ram_feasibility/gpu_budget_bytes")
                .or_else(|| v.get("gpu_budget_bytes")),
        });
    }
    if key.ends_with("/config.json") {
        return reduce_config(v);
    }
    if key.contains("/tree/") {
        let files: Vec<Value> = v
            .as_array()
            .into_iter()
            .flatten()
            .filter(|f| f.get("type").is_none_or(|t| t == "file"))
            .map(|f| pick(f, &["path", "size"]))
            .collect();
        return Value::Array(files);
    }
    if key.starts_with("/api/models?") {
        return Value::Array(
            v.as_array()
                .into_iter()
                .flatten()
                .map(listing_row)
                .collect(),
        );
    }
    if key.starts_with("/api/models/") {
        return listing_row(v);
    }
    v.clone()
}

/// Recorded bodies: a JSON Lines file of `{"key", "body"}` or `{"key", "refused"}`.
/// A key with no line refuses, so a fixture run never reaches the network.
pub struct Fixture {
    lines: std::collections::BTreeMap<String, Result<Value, String>>,
}

impl Fixture {
    /// # Errors
    ///
    /// When the file cannot be read or a line is not a record.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("cannot read {}", path.display()))?;
        Self::parse(&text)
    }

    /// # Errors
    ///
    /// When a line is not a `{"key", "body"|"refused"}` record.
    pub fn parse(text: &str) -> Result<Self> {
        let mut lines = std::collections::BTreeMap::new();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let v: Value = serde_json::from_str(line).context("fixture line is not JSON")?;
            let key = v
                .get("key")
                .and_then(Value::as_str)
                .context("fixture line has no key")?
                .to_string();
            let body = match (v.get("body"), v.get("refused").and_then(Value::as_str)) {
                (_, Some(why)) => Err(why.to_string()),
                (Some(b), None) => Ok(b.clone()),
                (None, None) => anyhow::bail!("fixture line {key} has neither body nor refused"),
            };
            lines.insert(key, body);
        }
        Ok(Self { lines })
    }

    /// Make one key refuse with `why`: how a test makes a feed refuse.
    pub fn refuse(&mut self, key: &str, why: &str) {
        self.lines.insert(key.to_string(), Err(why.to_string()));
    }

    /// The recorded body of `key`, if one is recorded.
    #[must_use]
    pub fn body(&self, key: &str) -> Option<&Value> {
        self.lines.get(key).and_then(|r| r.as_ref().ok())
    }

    /// Replace one recorded body.
    pub fn set(&mut self, key: &str, body: Value) {
        self.lines.insert(key.to_string(), Ok(body));
    }
}

impl Feed for Fixture {
    fn get(&self, key: &str) -> Result<Value, String> {
        self.lines
            .get(key)
            .cloned()
            .unwrap_or_else(|| Err("not recorded".into()))
    }
}

const TIMEOUT: Duration = Duration::from_secs(20);

/// The network, read once per key.
pub struct Live {
    pub hf: String,
    pub osaurus_url: String,
    /// When set, every answer is appended here as a fixture line.
    pub record: Option<std::path::PathBuf>,
}

fn http_get(url: &str) -> Result<reqwest::blocking::Response, String> {
    let resp = reqwest::blocking::Client::builder()
        .timeout(TIMEOUT)
        .user_agent("ztools-scout")
        .build()
        .map_err(|e| e.to_string())?
        .get(url)
        .send()
        .map_err(|e| format!("no answer: {e}"))?;
    if resp.status().is_success() {
        Ok(resp)
    } else {
        Err(format!("HTTP {}", resp.status()))
    }
}

impl Live {
    fn fetch(&self, key: &str) -> Result<Value, String> {
        if key == HEALTH {
            let url = format!("{}/health", self.osaurus_url.trim_end_matches('/'));
            let timing = crate::ztools::residency::Timing::default();
            return crate::ztools::residency::locked(
                "ztools models scout: /health",
                timing,
                || http_get(&url)?.json::<Value>().map_err(|e| e.to_string()),
            )
            .map_err(|e| format!("GPU lock: {e:#}"))?;
        }
        if let Some(version) = key.strip_prefix("catalog:") {
            let text = http_get(&format!("{OSAURUS_SRC}/{version}/{CATALOG_FILE}"))?
                .text()
                .map_err(|e| e.to_string())?;
            return super::catalog::parse(&text);
        }
        http_get(&format!("{}{key}", self.hf))?
            .json::<Value>()
            .map_err(|e| format!("not JSON: {e}"))
    }
}

impl Feed for Live {
    fn get(&self, key: &str) -> Result<Value, String> {
        let got = self.fetch(key).map(|v| reduce(key, &v));
        if let Some(path) = &self.record {
            let line = match &got {
                Ok(body) => json!({"key": key, "body": body}),
                Err(why) => json!({"key": key, "refused": why}),
            };
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .map_err(|e| format!("cannot record: {e}"))?;
            std::io::Write::write_all(&mut file, format!("{line}\n").as_bytes())
                .map_err(|e| format!("cannot record: {e}"))?;
        }
        got
    }
}
