pub mod constants;
pub mod dates;
pub mod dedup;
pub mod enforce;
pub mod family;
pub mod fetch;
pub mod format;
pub mod health;
pub mod holidays;
pub mod listing;
pub mod phases;
pub mod prompts;
pub mod queries;
pub mod refine;
pub mod report;
pub mod score;
pub mod suitability;
pub mod supply;
pub mod transcript;
pub use constants::*;
pub use dates::*;
pub use dedup::merge_duplicate_events;
pub use enforce::*;
pub use fetch::*;
pub use format::*;
pub use health::*;
pub use phases::*;
pub use prompts::*;
pub use queries::*;
pub use refine::{merge_refined, refine_draft};
pub use score::{apply_scores, compute_score};
/// Native Rust Weekend Planner module.
use serde::{Deserialize, Serialize};

pub use followup::*;
pub use report::*;
pub use search::*;
pub use search_order::*;
pub use search_parse::*;
pub use supply::*;
pub use transcript::{PhaseEntry, PhaseLog};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WeekendEvent {
    pub name: String,
    pub location: String,
    pub price: String,
    pub target_ages: String,
    pub day: String,
    pub dates: String,
    pub description: String,
    pub is_transient: bool,
    #[serde(default)]
    pub score: f32,
    /// Raw fields the enforcement suite needs beyond the rendered columns.
    /// `dates` holds the display form; these carry the parseable source values.
    #[serde(default)]
    pub start_date: String,
    #[serde(default)]
    pub end_date: String,
    #[serde(default)]
    pub weather: String,
    #[serde(default)]
    pub duration: String,
}

pub use super::weekend_cache::{
    clean_venue_or_event_title, is_directory_or_list_page, load_cached_activities, load_exclusions,
};

#[derive(serde::Deserialize, Default)]
struct LlmResponse {
    transient_events: Vec<WeekendEventLlm>,
}

#[derive(serde::Deserialize, Default)]
struct WeekendEventLlm {
    #[serde(default)]
    name: String,
    #[serde(default)]
    location: String,
    #[serde(default)]
    target_ages: String,
    #[serde(default)]
    price: String,
    #[serde(default)]
    start_date: String,
    #[serde(default)]
    end_date: String,
    #[serde(default)]
    day: String,
    #[serde(default)]
    weather: String,
    #[serde(default)]
    duration: String,
    #[serde(default)]
    description: String,
}

/// Alternate keys a model may emit for each canonical event field, in
/// priority order. Port of the `*_KEYS` lists in
/// `references/weekend/llm.py::normalize_llm_items`: the shipped prompts pin
/// canonical keys, but a model that answers with gemma-style `activity` /
/// `venue` must not silently lose its name and location at parse time.
const ALT_KEYS: &[(&str, &[&str])] = &[
    (
        "name",
        &[
            "name",
            "activity",
            "activity_name",
            "title",
            "event",
            "event_name",
            "description",
        ],
    ),
    ("location", &["location", "address", "venue", "place"]),
    (
        "target_ages",
        &["target_ages", "age_group", "ages", "age_range"],
    ),
    ("price", &["price", "cost", "pricing", "fee"]),
    ("weather", &["weather", "setting", "type", "indoor_outdoor"]),
    ("day", &["day", "date", "dates", "event_date"]),
    ("duration", &["duration", "end_date", "time"]),
];

/// Fill missing canonical keys on one raw event object from its alternate
/// keys, first present alternate wins. Presence, not emptiness, decides: a
/// present-but-empty canonical key is never overwritten, exactly like the
/// Python original (`if std not in item`).
fn normalize_llm_item(obj: &mut serde_json::Map<String, serde_json::Value>) {
    for (canonical, alts) in ALT_KEYS {
        if obj.contains_key(*canonical) {
            continue;
        }
        if let Some(found) = alts.iter().find_map(|k| obj.get(*k).cloned()) {
            obj.insert((*canonical).to_string(), found);
        }
    }
}

/// Parse an LLM chat-completions response into weekend events.
#[must_use]
pub fn parse_llm_events(resp: &serde_json::Value) -> Option<Vec<WeekendEvent>> {
    let text = resp["choices"][0]["message"]["content"].as_str()?;
    let clean_text = text
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    let mut doc: serde_json::Value = serde_json::from_str(clean_text).ok()?;
    if let Some(events) = doc
        .get_mut("transient_events")
        .and_then(serde_json::Value::as_array_mut)
    {
        for event in events {
            if let Some(obj) = event.as_object_mut() {
                normalize_llm_item(obj);
            }
        }
    }
    let parsed: LlmResponse = serde_json::from_value(doc).ok()?;
    Some(
        parsed
            .transient_events
            .into_iter()
            .map(|e| WeekendEvent {
                // Empty stays empty: the renderer prints the missing-value
                // sentinel (class C4). It used to copy the name here, so
                // "Why It Fits" repeated the event's own title as its reason.
                description: e.description,
                name: e.name,
                location: e.location,
                price: if e.price.is_empty() {
                    "unknown".to_string()
                } else {
                    e.price
                },
                target_ages: if e.target_ages.is_empty() {
                    "unknown".to_string()
                } else {
                    e.target_ages
                },
                day: if e.day.is_empty() {
                    "This Weekend".to_string()
                } else {
                    e.day
                },
                dates: e.start_date.clone(),
                start_date: e.start_date.clone(),
                end_date: e.end_date,
                weather: e.weather,
                duration: e.duration,
                is_transient: true,
                score: 0.0,
            })
            .collect(),
    )
}

pub mod followup;
pub mod relevance;
pub mod search;
pub mod search_order;
pub mod search_parse;

/// Search keywords for the season a month falls in.
///
/// Total: every month maps to a season, and anything unrecognised falls to
/// spring. It returned `Option` and could not produce `None` on any path.
pub(crate) fn seasonal_keywords(month_name: &str) -> &'static str {
    let m = month_name.to_lowercase();
    if m == "june" || m == "july" || m == "august" {
        "summer festival fair"
    } else if m == "september" || m == "october" || m == "november" {
        "harvest festival farm pumpkin"
    } else if m == "december" || m == "january" || m == "february" {
        "winter festival holiday lights"
    } else {
        "spring festival maple syrup"
    }
}

fn parse_snippets_from_html(html: &str) -> Vec<String> {
    let mut snippets = Vec::new();
    let mut search_idx = 0;

    while search_idx < html.len() {
        let find_standard = html[search_idx..].find("class=\"result__snippet\"");
        let find_lite = html[search_idx..].find("class=\"result-snippet\"");

        let (start, pattern_len) = match (find_standard, find_lite) {
            (Some(s1), Some(s2)) => {
                if s1 <= s2 {
                    (s1, 23)
                } else {
                    (s2, 22)
                }
            }
            (Some(s1), None) => (s1, 23),
            (None, Some(s2)) => (s2, 22),
            (None, None) => break,
        };

        let absolute_start = search_idx + start;
        if let Some(tag_end) = html[absolute_start..].find('>') {
            let text_start = absolute_start + tag_end + 1;
            let end_a = html[text_start..].find("</a>");
            let end_td = html[text_start..].find("</td>");
            let text_end_opt = match (end_a, end_td) {
                (Some(a), Some(td)) => Some(a.min(td)),
                (Some(a), None) => Some(a),
                (None, Some(td)) => Some(td),
                (None, None) => None,
            };

            if let Some(text_end) = text_end_opt {
                let snippet = &html[text_start..text_start + text_end];
                let clean_snippet = snippet
                    .replace("<b>", "")
                    .replace("</b>", "")
                    .replace("&#x27;", "'")
                    .replace("&amp;", "&")
                    .replace("&quot;", "\"")
                    .trim()
                    .to_string();
                if !clean_snippet.is_empty() {
                    snippets.push(clean_snippet);
                }
                search_idx = text_start + text_end;
                continue;
            }
        }
        search_idx = absolute_start + pattern_len;
    }
    snippets
}

/// True if HTML content matches known CAPTCHA/WAF bot challenge markers.
#[must_use]
pub fn is_challenged(html: &str) -> bool {
    let lower = html.to_lowercase();
    let markers = [
        "anomaly-modal",
        "anomaly.js",
        "challenge-form",
        "verify you are human",
        "just a moment",
        "attention required",
        "managed challenge",
        "challenges.cloudflare.com",
    ];
    markers.iter().any(|m| lower.contains(m))
}

/// Query DuckDuckGo web search endpoint for live event/venue listings.
/// Format the final weekend markdown plan document.
/// Flag columns that have constant values across all rows (e.g. repetitive prices or ages).
#[cfg(test)]
#[path = "../weekend_tests.rs"]
mod tests;
