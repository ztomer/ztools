//! Fetching the live roster from the server and dropping ghost entries.
//!
//! Split out of model_resolve.rs for the 500-line cap.

use std::time::Duration;

use super::disk::disk_corroborated;
use super::roster::{RosterEntry, API_TAGS, ROSTER_TIMEOUT_SECS};

/// Remove roster entries with nothing on disk behind them.
///
/// Done HERE, at the boundary where the claim enters the process, rather than
/// inside [`substitute_model`] -- keeping the selector pure. If NOTHING is
/// corroborated the original list is returned unchanged: that is far more
/// likely to mean the probe is broken than that every served model is a ghost.
pub(super) fn drop_uncorroborated(entries: Vec<RosterEntry>) -> Vec<RosterEntry> {
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
