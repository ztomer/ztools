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
/// Consecutive extract-batch failures that trip the pass-through breaker.
///
/// Without a breaker a dead model cost FOUR timeouts per
/// line (8 -> 4 -> 2 -> 1 -> raw) across every line of the corpus — at 300s
/// each, a 66-line corpus was a 22-hour run that the scheduler killed at 30
/// minutes, every week, with nothing to show.
pub const EXTRACT_FAILURE_BREAKER: usize = 3;
pub const DEFAULT_BATCH_SIZE: usize = 8;
pub const MAX_BATCH_SIZE: usize = 12;
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

/// Resolve the weekend model against the active Osaurus roster if needed.
#[must_use]
pub fn resolve_weekend_model(base_url: &str, preferred_model: &str) -> String {
    let url = format!("{}/v1/models", base_url.trim_end_matches('/'));
    let Ok(client) = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
    else {
        return preferred_model.to_string();
    };

    let resp = match client.get(&url).send() {
        Ok(r) if r.status().is_success() => r.json::<serde_json::Value>().ok(),
        _ => None,
    };

    let Some(json) = resp else {
        return preferred_model.to_string();
    };

    let models: Vec<String> = json
        .get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id").and_then(|id| id.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default();

    if models.is_empty() || models.iter().any(|m| m == preferred_model) {
        return preferred_model.to_string();
    }

    let pref_lower = preferred_model.to_lowercase();
    let family = if pref_lower.contains("qwen") {
        "qwen"
    } else if pref_lower.contains("gemma") {
        "gemma"
    } else if pref_lower.contains("muse") {
        "muse"
    } else if pref_lower.contains("raptor") {
        "raptor"
    } else {
        ""
    };

    if !family.is_empty() {
        if let Some(matched) = models.iter().find(|m| m.to_lowercase().contains(family)) {
            return matched.clone();
        }
    }

    models
        .first()
        .cloned()
        .unwrap_or_else(|| preferred_model.to_string())
}

/// Wake the weekend model with one tiny request, waiting the LOADING budget.
///
/// Every other call in this pipeline carries a generation timeout (120s /
/// 300s). A cold 25GB model can take minutes to page in, and a client that
/// disconnects mid-load makes the server cancel the load — so the next call
/// starts it again, and the model never becomes ready. One call with the
/// warm-up budget, before any phase, turns that livelock into one wait.
#[must_use]
pub fn warm_model(config: &crate::config::ZtoolsConfig) -> super::ModelHealth {
    let model = resolve_weekend_model(&config.osaurus_url, &config.weekend_model);
    let started = std::time::Instant::now();
    // The stall guard IS the budget here: no token until the model has
    // loaded is the expected shape of a cold start, not a stalled server.
    let outcome = crate::ztools::llm::chat(
        &crate::ztools::llm::ChatRequest {
            base_url: &config.osaurus_url,
            model: &model,
            system: None,
            user: "Reply with the single word: ready",
            json: false,
        },
        &crate::ztools::llm::ChatBudget {
            stall_secs: config.llm_warmup_timeout_secs,
            cap_secs: config.llm_warmup_timeout_secs,
            max_tokens: 8,
        },
    );
    let secs = started.elapsed().as_secs();
    match outcome {
        Ok(_) => super::ModelHealth::Ready { model, secs },
        Err(e) => super::ModelHealth::Unavailable {
            model,
            reason: format!(
                "no answer within {secs}s of a {}s warm-up budget: {}",
                config.llm_warmup_timeout_secs,
                first_line(&e.to_string())
            ),
        },
    }
}

fn first_line(text: &str) -> &str {
    text.lines().next().unwrap_or(text).trim()
}

/// One plain-text LLM call against the configured osaurus endpoint.
#[must_use]
pub fn call_llm_text(prompt: &str, config: &crate::config::ZtoolsConfig) -> Option<String> {
    let model = resolve_weekend_model(&config.osaurus_url, &config.weekend_model);
    crate::ztools::twitter::call_osaurus(
        &config.osaurus_url,
        &model,
        prompt,
        config.llm_extended_timeout_secs,
        config,
    )
    .ok()
    .filter(|s| !s.trim().is_empty())
}

/// One JSON LLM call returning the raw response value.
///
/// The answer is re-wrapped in the completion shape the parsers expect
/// (`choices[0].message.content`), so a streamed answer and a stubbed one
/// look the same to `parse_llm_events`.
pub(crate) fn call_llm_json(
    system: Option<&str>,
    user: &str,
    config: &crate::config::ZtoolsConfig,
) -> Option<serde_json::Value> {
    let model = resolve_weekend_model(&config.osaurus_url, &config.weekend_model);
    let content = crate::ztools::llm::chat(
        &crate::ztools::llm::ChatRequest {
            base_url: &config.osaurus_url,
            model: &model,
            system,
            user,
            json: true,
        },
        &config.chat_budget(config.llm_extended_timeout_secs),
    )
    .ok()?;
    Some(serde_json::json!({"choices": [{"message": {"content": content}}]}))
}

/// Condense a forecast to 1-2 sentences; fall back to a preview on failure.
#[must_use]
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
#[must_use]
pub fn extract_sources(
    raw_text: &str,
    location: &str,
    config: &crate::config::ZtoolsConfig,
) -> String {
    let raw_lines: Vec<&str> = raw_text
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            t.starts_with("- ") || t.starts_with("[THIS WEEKEND]")
        })
        .collect();
    if raw_lines.is_empty() {
        return raw_text.to_string();
    }

    let (marked, general): (Vec<&str>, Vec<&str>) = raw_lines
        .into_iter()
        .partition(|l| l.trim_start().starts_with("[THIS WEEKEND]"));
    let mut lines = marked;
    lines.extend(general.into_iter().take(24));

    let mut results = Vec::new();
    let mut batch_size = DEFAULT_BATCH_SIZE;
    let mut streak = 0;
    let mut failures = 0;
    let mut i = 0;
    while i < lines.len() {
        if failures >= EXTRACT_FAILURE_BREAKER {
            eprintln!(
                "\u{26a0} extract: {failures} consecutive model failures; passing the \
                 remaining {} line(s) through raw",
                lines.len() - i
            );
            results.extend(lines[i..].iter().map(ToString::to_string));
            break;
        }
        let end = cmp::min(i + batch_size, lines.len());
        let chunk = lines[i..end].join("\n");
        let prompt = render(
            PHASE_EXTRACT_EVENTS,
            &[("location", location), ("raw_text", &chunk)],
        );
        if let Some(res) = call_llm_text(&prompt, config) {
            results.push(res);
            streak += 1;
            failures = 0;
            i = end;
            if streak >= BATCH_GROWTH_STREAK_LIMIT && batch_size < MAX_BATCH_SIZE {
                batch_size += 1;
            }
        } else {
            streak = 0;
            failures += 1;
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
#[must_use]
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
#[must_use]
pub fn refine_draft(draft_text: &str, config: &crate::config::ZtoolsConfig) -> String {
    let prompt = render(PHASE_REFINE, &[("draft_text", draft_text)]);
    call_llm_text(&prompt, config).unwrap_or_else(|| draft_text.to_string())
}

/// Phase 4: structure the refined draft into the transient-event JSON schema.
#[must_use]
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
