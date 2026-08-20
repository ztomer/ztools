pub mod constants;
pub mod dates;
pub mod enforce;
pub mod fetch;
pub mod format;
pub mod phases;
pub mod prompts;
pub mod supply;
pub use constants::*;
pub use dates::*;
pub use enforce::*;
pub use fetch::*;
pub use format::*;
pub use phases::*;
pub use prompts::*;
/// Native Rust Weekend Planner module.
use serde::{Deserialize, Serialize};
pub use supply::*;

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

/// Fetch Open-Meteo weather forecast for Vaughan / GTA (lat 43.8361, lon -79.4982).
/// Clean weather string display for CLI header panel matching Python _format_weather_display.
pub fn apply_scores(events: &mut [WeekendEvent], weather_str: &str, age_range: &str) {
    for ev in events.iter_mut() {
        ev.score = compute_score(ev, weather_str, age_range);
    }
    // Sort descending by score
    events.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

pub(crate) fn compute_score(ev: &WeekendEvent, weather_str: &str, age_range: &str) -> f32 {
    let mut score = 0.0;

    // Populated fields
    let fields = [
        &ev.name,
        &ev.location,
        &ev.price,
        &ev.target_ages,
        &ev.description,
    ];
    let populated = fields.iter().filter(|f| !f.is_empty()).count() as f32;
    score += (populated / 5.0) * 3.0;

    // Ages overlap
    if !age_range.is_empty() && !ev.target_ages.is_empty() {
        let parse_nums = |s: &str| -> Vec<i32> {
            let mut v = Vec::new();
            let mut cur = String::new();
            for c in s.chars() {
                if c.is_ascii_digit() {
                    cur.push(c);
                } else if !cur.is_empty() {
                    if let Ok(n) = cur.parse() {
                        v.push(n);
                    }
                    cur.clear();
                }
            }
            if !cur.is_empty() {
                if let Ok(n) = cur.parse() {
                    v.push(n);
                }
            }
            v.sort();
            v.dedup();
            v
        };
        let age_nums = parse_nums(age_range);
        let target_nums = parse_nums(&ev.target_ages);

        if !age_nums.is_empty() && !target_nums.is_empty() {
            let max_min = std::cmp::max(age_nums[0], target_nums[0]);
            let min_max = std::cmp::min(*age_nums.last().unwrap(), *target_nums.last().unwrap());
            if min_max >= max_min {
                let overlap = min_max - max_min + 1;
                if overlap >= 2 {
                    score += 3.0;
                } else if overlap == 1 {
                    score += 1.5;
                }
            }
        }
    }

    // Weather
    let desc_lower = ev.description.to_lowercase();
    let is_outdoor = desc_lower.contains("outdoor");
    let is_indoor = desc_lower.contains("indoor");
    let is_sunny =
        desc_lower.contains("sunny") || desc_lower.contains("clear") || desc_lower.contains("warm");

    let w_lower = weather_str.to_lowercase();
    let forecast_sunny =
        w_lower.contains("sunny") || w_lower.contains("clear") || w_lower.contains("warm");
    let forecast_cloudy =
        w_lower.contains("cloudy") || w_lower.contains("rain") || w_lower.contains("precipitation");
    let is_cloudy = desc_lower.contains("cloudy")
        || desc_lower.contains("rain")
        || desc_lower.contains("overcast");

    if is_indoor {
        score += 1.0;
    } else if is_outdoor && forecast_sunny {
        score += 2.0;
    } else if is_outdoor && forecast_cloudy {
        // pass
    } else if (is_cloudy && forecast_cloudy) || (is_sunny && forecast_sunny) {
        score += 2.0;
    } else if is_sunny || is_cloudy {
        score += 1.0;
    }

    // Other bonuses
    let p_lower = ev.price.to_lowercase();
    if !p_lower.is_empty() && p_lower != "free" && p_lower != "n/a" && p_lower != "tbd" {
        score += 0.5;
    }
    if ev.location.len() > 5 {
        score += 0.5;
    }

    let final_score = score / 2.0;
    if final_score > 5.0 {
        5.0
    } else {
        final_score
    }
}

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

/// Parse an LLM chat-completions response into weekend events.
pub fn parse_llm_events(resp: &serde_json::Value) -> Option<Vec<WeekendEvent>> {
    let text = resp["choices"][0]["message"]["content"].as_str()?;
    let clean_text = text
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    let parsed: LlmResponse = serde_json::from_str(clean_text).ok()?;
    Some(
        parsed
            .transient_events
            .into_iter()
            .map(|e| WeekendEvent {
                description: if e.description.is_empty() {
                    e.name.clone()
                } else {
                    e.description
                },
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

fn call_osaurus_json(
    prompt: &str,
    config: &crate::config::ZtoolsConfig,
) -> Option<Vec<WeekendEvent>> {
    let resp = phases::call_llm_json(None, prompt, config)?;
    parse_llm_events(&resp)
}

pub(crate) fn _seasonal_keywords(month_name: &str) -> Option<&'static str> {
    let m = month_name.to_lowercase();
    if m == "june" || m == "july" || m == "august" {
        Some("summer festival fair")
    } else if m == "september" || m == "october" || m == "november" {
        Some("harvest festival farm pumpkin")
    } else if m == "december" || m == "january" || m == "february" {
        Some("winter festival holiday lights")
    } else {
        Some("spring festival maple syrup")
    }
}

fn search_duckduckgo_html(query: &str, url: &str) -> Vec<String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap_or_default();

    let user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";

    // Try GET first (html.duckduckgo.com returns HTTP 200 with snippets on GET)
    let res = client
        .get(url)
        .query(&[("q", query)])
        .header("User-Agent", user_agent)
        .send();

    let res = match res {
        Ok(resp) if resp.status().is_success() => Ok(resp),
        _ => client
            .post(url)
            .form(&[("q", query)])
            .header("User-Agent", user_agent)
            .send(),
    };

    let mut snippets = Vec::new();
    if let Ok(resp) = res {
        if let Ok(html) = resp.text() {
            let mut search_idx = 0;
            while let Some(start) = html[search_idx..].find("class=\"result__snippet\"") {
                let absolute_start = search_idx + start;
                if let Some(href_end) = html[absolute_start..].find('>') {
                    let text_start = absolute_start + href_end + 1;
                    if let Some(text_end) = html[text_start..].find("</a>") {
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
                search_idx = absolute_start + 23;
            }
        }
    }
    snippets
}

/// Query DuckDuckGo web search endpoint for live event/venue listings.
/// Format the final weekend markdown plan document.
/// Flag columns that have constant values across all rows (e.g. repetitive prices or ages).
#[cfg(test)]
#[path = "../weekend_tests.rs"]
mod tests;
