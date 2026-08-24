//! Roster entries, substitution scoring, and the missing-model fallback chain.
//!
//! Split out of model_resolve.rs for the 500-line cap.

pub(super) const ROSTER_TIMEOUT_SECS: u64 = 10;
pub(super) const API_TAGS: &str = "/api/tags";

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
