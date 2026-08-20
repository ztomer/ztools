//! The multi-phase weekend pipeline: extract -> draft -> refine -> structure.
//!
//! Ported from `weekend/phases.py`. Class C2c: these phases form a CARRIER
//! CHAIN. Each one must pass DATES/PRICE/AGES/LOCATION through verbatim -- the
//! predecessor narrowed the payload at every step, so an event's real dates
//! were gone two phases before the schema that asked for them, and every date
//! column rendered blank.

use std::cmp;

use super::prompts::{
    render, CARRY_FIELDS, PHASE_DRAFT_TRANSIENT, PHASE_EXTRACT_EVENTS, PHASE_REFINE,
    PHASE_STRUCTURE_TRANSIENT_SYSTEM, PHASE_STRUCTURE_USER, PHASE_WEATHER_CONDENSE,
};

pub const WEATHER_PREVIEW_LIMIT: usize = 200;
pub const DEFAULT_BATCH_SIZE: usize = 3;
pub const MAX_BATCH_SIZE: usize = 5;
pub const BATCH_GROWTH_STREAK_LIMIT: usize = 3;

/// The plan-level context every phase shares. Bundled so the phase signatures
/// stay narrow and one consumer cannot pass a plan year that disagrees with
/// another's date range.
#[derive(Clone)]
pub struct PlanContext {
    pub location: String,
    pub ages: String,
    pub date_range: String,
    pub year: i32,
    pub exclusions: String,
}

/// One plain-text LLM call against the configured osaurus endpoint.
pub fn call_llm_text(prompt: &str, config: &crate::config::ZtoolsConfig) -> Option<String> {
    crate::ztools::twitter::call_osaurus(&config.osaurus_url, &config.weekend_model, prompt, config)
        .ok()
        .filter(|s| !s.trim().is_empty())
}

/// One JSON LLM call returning the raw response value.
pub(crate) fn call_llm_json(
    system: Option<&str>,
    user: &str,
    config: &crate::config::ZtoolsConfig,
) -> Option<serde_json::Value> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(config.llm_timeout_secs))
        .build()
        .ok()?;
    let mut messages = Vec::new();
    if let Some(sys) = system {
        messages.push(serde_json::json!({"role": "system", "content": sys}));
    }
    messages.push(serde_json::json!({"role": "user", "content": user}));
    let payload = serde_json::json!({
        "model": config.weekend_model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0.0
    });
    let url = format!(
        "{}/v1/chat/completions",
        config.osaurus_url.trim_end_matches('/')
    );
    client.post(&url).json(&payload).send().ok()?.json().ok()
}

/// Condense a forecast to 1-2 sentences; fall back to a preview on failure.
pub fn condense_weather(weather_str: &str, config: &crate::config::ZtoolsConfig) -> String {
    let prompt = render(PHASE_WEATHER_CONDENSE, &[("weather_str", weather_str)]);
    call_llm_text(&prompt, config)
        .unwrap_or_else(|| weather_str.chars().take(WEATHER_PREVIEW_LIMIT).collect())
}

/// Phase 1: pull clean pipe-separated lines out of the raw scraped corpus.
///
/// Batched with adaptive sizing: a streak of successes grows the batch, a
/// failure halves it and (at batch 1) falls back to passing the line through
/// raw rather than dropping it. The Python original persists batch sizes to a
/// signals file; this port keeps them in-memory per run, which is what the
/// shapes actually depend on.
pub fn extract_sources(
    raw_text: &str,
    location: &str,
    config: &crate::config::ZtoolsConfig,
) -> String {
    let lines: Vec<&str> = raw_text
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            t.starts_with("- ") || t.starts_with("[THIS WEEKEND]")
        })
        .collect();
    if lines.is_empty() {
        return raw_text.to_string();
    }

    let mut results = Vec::new();
    let mut batch_size = DEFAULT_BATCH_SIZE;
    let mut streak = 0;
    let mut i = 0;
    while i < lines.len() {
        let end = cmp::min(i + batch_size, lines.len());
        let chunk = lines[i..end].join("\n");
        let prompt = render(
            PHASE_EXTRACT_EVENTS,
            &[("location", location), ("raw_text", &chunk)],
        );
        if let Some(res) = call_llm_text(&prompt, config) {
            results.push(res);
            streak += 1;
            i = end;
            if streak >= BATCH_GROWTH_STREAK_LIMIT && batch_size < MAX_BATCH_SIZE {
                batch_size += 1;
            }
        } else {
            streak = 0;
            batch_size = cmp::max(batch_size / 2, 1);
            if batch_size == 1 {
                // A single line that even the model rejects is passed through
                // rather than dropped: an empty extract is worse than a raw one.
                results.push(lines[i].to_string());
                i += 1;
                batch_size = DEFAULT_BATCH_SIZE;
            }
        }
    }

    if results.is_empty() {
        raw_text.to_string()
    } else {
        results.join("\n")
    }
}

/// Phase 2: draft candidate activities from the cleaned sources.
pub fn draft_activities(
    weather_condensed: &str,
    cleaned_sources: &str,
    ctx: &PlanContext,
    config: &crate::config::ZtoolsConfig,
) -> Option<String> {
    let prompt = render(
        PHASE_DRAFT_TRANSIENT,
        &[
            ("age_range", &ctx.ages),
            ("location", &ctx.location),
            ("date_range", &ctx.date_range),
            ("year", &ctx.year.to_string()),
            ("weather_condensed", weather_condensed),
            ("cleaned_sources", cleaned_sources),
            ("carry", CARRY_FIELDS),
            ("exclusions", &ctx.exclusions),
        ],
    );
    call_llm_text(&prompt, config)
}

/// Phase 3: merge near-duplicates, keep the best, sort by appeal.
pub fn refine_draft(draft_text: &str, config: &crate::config::ZtoolsConfig) -> String {
    let prompt = render(PHASE_REFINE, &[("draft_text", draft_text)]);
    call_llm_text(&prompt, config).unwrap_or_else(|| draft_text.to_string())
}

/// Phase 4: structure the refined draft into the transient-event JSON schema.
pub fn structure_to_json(
    text: &str,
    weather_condensed: &str,
    year: i32,
    config: &crate::config::ZtoolsConfig,
) -> Option<Vec<super::WeekendEvent>> {
    let sys = render(
        PHASE_STRUCTURE_TRANSIENT_SYSTEM,
        &[
            ("year", &year.to_string()),
            ("weather_condensed", weather_condensed),
        ],
    );
    let usr = render(PHASE_STRUCTURE_USER, &[("draft_text", text)]);
    let resp = call_llm_json(Some(&sys), &usr, config)?;
    super::parse_llm_events(&resp)
}
