//! The Osaurus catalog as its source states it: the curated ids with their
//! `modelType`, and `retiredOsaurusOrgIds`, read from
//! `Packages/OsaurusCore/Managers/Model/ModelManager.swift` at the installed tag.

use regex::Regex;
use serde_json::{Value, json};
use std::sync::LazyLock;

static CURATED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?s)curated\(\s*id:\s*"([^"]+)"(.*?)\n\s*\)"#).expect("static regex")
});
static MODEL_TYPE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"modelType:\s*"([^"]+)""#).expect("static regex"));
static QUOTED: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#""([^"]+)""#).expect("static regex"));

/// `{"curated": [{"id", "model_type"}], "retired": [lowercased ids]}`.
///
/// # Errors
///
/// When the source has no curated entry or no retired list: the shape moved,
/// and a catalog read that found nothing must not read as an empty catalog.
pub fn parse(source: &str) -> Result<Value, String> {
    let curated: Vec<Value> = CURATED
        .captures_iter(source)
        .map(|c| {
            let kind = MODEL_TYPE.captures(&c[2]).map(|m| m[1].to_string());
            json!({"id": &c[1], "model_type": kind})
        })
        .collect();
    let block = source
        .split_once("retiredOsaurusOrgIds: Set<String> = [")
        .and_then(|(_, rest)| rest.split_once(']'))
        .map(|(b, _)| b)
        .ok_or("the Osaurus source has no retiredOsaurusOrgIds list")?;
    let retired: Vec<String> = block
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .flat_map(|l| QUOTED.captures_iter(l).map(|c| c[1].to_lowercase()))
        .collect();
    if curated.is_empty() {
        return Err("the Osaurus source has no curated(id:) entries".into());
    }
    Ok(json!({"curated": curated, "retired": retired}))
}
