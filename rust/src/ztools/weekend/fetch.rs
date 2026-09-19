use chrono::NaiveDate;

use super::WeekendEvent;
use super::{
    condense_weather, draft_activities, extract_sources, in_window_count, prioritise_in_window,
    refine_draft, seasonal_keywords, structure_to_json, PlanContext, SearchResult,
};
use super::{follow_aggregators, search_engines, warm_model};
use super::{ModelHealth, PlanHealth, SearchHealth};

/// Body truncation bound, mirrored from `WEEKEND_MAX_BODY_LENGTH`.
///
/// The corpus byte-parity gate pins this at the Python default (300) — a
/// change here without the gate going red-with-reason is drift.
pub const MAX_BODY_LENGTH: usize = 300;

/// Dedupe raw scrape results by normalised TITLE and drop the ones with no
/// in-region evidence. Byte-exact port of `data.py::_clean_search_results`.
///
/// The semantics that must NOT drift, all pinned by the parity fixtures:
/// - the dedupe key is the TITLE only (lowercased, trailing `.,!?:; ` and
///   whitespace stripped) — two results sharing a title are one result,
///   regardless of their bodies;
/// - an empty title means the result is dropped outright (there is no "Event"
///   fallback label on the corpus path);
/// - the body is truncated to `max_body` chars, then stripped;
/// - region evidence is judged on `"{title} {body}"` against the passed
///   [`RegionLists`](crate::ztools::weekend_cache::RegionLists) (loaded from
///   `conf/weekend.toml [region]` — data, not code), and out-of-region
///   results are counted and reported, never silently kept.
#[must_use]
pub fn clean_search_results(
    results: &[SearchResult],
    default_label: &str,
    max_body: usize,
    region: &crate::ztools::weekend_cache::RegionLists,
) -> String {
    let mut seen = std::collections::HashSet::new();
    let mut cleaned = Vec::new();
    let mut dropped = 0usize;
    for r in results {
        let title = r.title.trim();
        let norm: String = if title.is_empty() {
            String::new()
        } else {
            // CHAR-SET, not substring: a `&str` pattern to trim_end_matches
            // would only strip the exact contiguous suffix, but Python's
            // rstrip(".,!?:; ") strips any of the set. "Vaughan Fall Fair!!!"
            // must collapse to "vaughan fall fair" to dedupe against the plain
            // title, exactly as the parity fixture demands.
            title
                .to_lowercase()
                .trim_end_matches(['.', ',', '!', '?', ':', ';', ' '])
                .to_string()
        };
        if norm.is_empty() || seen.contains(&norm) {
            continue;
        }
        seen.insert(norm);
        let body: String = r
            .body
            .chars()
            .take(max_body)
            .collect::<String>()
            .trim()
            .to_string();
        if !crate::ztools::weekend_cache::has_region_evidence(&format!("{title} {body}"), region) {
            dropped += 1;
            continue;
        }
        let label = if title.is_empty() {
            default_label
        } else {
            title
        };
        cleaned.push(format!("- {label}: {body}"));
    }
    if dropped > 0 {
        println!("\u{2192} Dropped {dropped} out-of-region search result(s)");
    }
    cleaned.join("\n")
}

/// Search the aggregator for event snippets and return the cleaned, deduped,
/// in-window-prioritised corpus.
///
/// This is the ground truth the provenance gate judges extracted rows against.
/// Build search queries for a target weekend starting on d1.
#[must_use]
pub fn build_search_queries(d1: NaiveDate) -> Vec<String> {
    let month_name = d1.format("%B").to_string();
    let year = d1.format("%Y").to_string();

    let municipalities = ["Vaughan", "Markham", "Richmond Hill", "Toronto"];
    let mut queries = Vec::new();

    for city in municipalities {
        queries.push(format!("kids activities {city} {month_name} {year}"));
        queries.push(format!(
            "{city} community centre kids programs {month_name}"
        ));
        queries.push(format!("{city} family events {month_name} {year}"));
    }

    let region = "GTA";
    queries.push(format!("{region} family events {month_name} {year}"));
    queries.push(format!("{region} Zoo special events {month_name} {year}"));
    queries.push(format!(
        "{region} museum family programs {month_name} {year}"
    ));
    {
        let seasonal = seasonal_keywords(&month_name);
        queries.push(format!("{region} {seasonal} {month_name} {year}"));
    }
    queries
}

/// Search the aggregator for event snippets and return the cleaned, deduped,
/// in-window-prioritised corpus, plus what the engines said while building
/// it. The corpus is the ground truth the provenance gate judges extracted
/// rows against; the health is what the plan says when that corpus is empty.
fn fetch_events_corpus(
    d1: NaiveDate,
    d2: NaiveDate,
    config: &crate::config::ZtoolsConfig,
) -> (String, SearchHealth) {
    let queries = build_search_queries(d1);

    let mut all_results = Vec::<SearchResult>::new();
    let mut health = SearchHealth::default();
    for chunk in queries.chunks(4) {
        let mut handles = Vec::new();
        for q in chunk {
            let q_clone = q.clone();
            let ddg = config.duckduckgo_url.clone();
            let bing = config.bing_url.clone();
            handles.push(std::thread::spawn(move || {
                search_engines(&q_clone, &ddg, &bing)
            }));
        }
        for h in handles {
            if let Ok(outcome) = h.join() {
                health.record(&outcome);
                all_results.extend(outcome.results);
            }
        }
    }
    if let Some(summary) = health.summary() {
        println!("\u{2192} Search: {summary}");
    }

    // The scrape results become the corpus through the SAME byte-exact
    // cleaning the Python pipeline applies (title-only dedupe, max_body
    // truncation, region evidence on "{title} {body}"); see
    // `clean_search_results`. The parity gate byte-compares this output
    // against `_clean_search_results`. Region lists come from the configured
    // `weekend.toml` files, never from literals.
    let region = crate::ztools::weekend_cache::load_region_lists(&config.weekend_region_paths);
    let cleaned = clean_search_results(&all_results, "Event", MAX_BODY_LENGTH, &region);

    // The directory pages themselves are not activities, but the events are
    // INSIDE them: follow the promising ones and add their text to the corpus
    // so the extractor can read individual listings out of them.
    let followed = follow_aggregators(&all_results, &region);
    let mut lines: Vec<String> = cleaned
        .lines()
        .map(std::string::ToString::to_string)
        .collect();
    if !followed.is_empty() {
        lines.extend(followed.lines().map(std::string::ToString::to_string));
    }
    let raw_text = lines.join("\n");

    // Make the in-window candidates visible WITHOUT removing the rest. Filtering
    // here instead was tried and reverted (in the Python pipeline): it starved
    // the draft and the model invented events. The marked corpus is also what
    // the provenance gate judges against, matching the Python flow.
    let total = raw_text.lines().filter(|l| !l.trim().is_empty()).count();
    let in_window = in_window_count(&raw_text, d1, d2);
    let marked_text = prioritise_in_window(&raw_text, d1, d2);
    if total > 0 {
        println!("→ Candidates: {in_window}/{total} mention a date this weekend");
    }
    (marked_text, health)
}

/// The monolithic single-shot extraction used as a fallback when the 4-phase
/// pipeline stalls. The Python original falls back to this when a draft fails.
fn monolithic_transient(
    corpus: &str,
    location: &str,
    d1: NaiveDate,
    d2: NaiveDate,
    config: &crate::config::ZtoolsConfig,
) -> Vec<WeekendEvent> {
    let (d1_str, d2_str) = (
        d1.format("%Y-%m-%d").to_string(),
        d2.format("%Y-%m-%d").to_string(),
    );
    let prompt = format!(
        "You are an expert family activity planner. Extract up to 10 time-limited events happening STRICTLY this weekend (between {d1_str} and {d2_str}) in {location} from the text below.\n\
        Output JSON now. Use EXACT schema:\n\
        {{\"transient_events\": [{{\"name\": \"str\", \"location\": \"str\", \"target_ages\": \"str\", \"price\": \"str\", \"start_date\": \"str\", \"end_date\": \"str\", \"duration\": \"str\", \"weather\": \"str\", \"day\": \"str\", \"description\": \"str\"}}]}}\n\n\
        Rules for every field:\n\
        - Suggest up to 10 specific weekend activities. Do NOT stop after just 1 or 2 events. Find as many as you can.\n\
        - Only extract events that occur within or overlap with the dates {d1_str} to {d2_str}. Discard events from past or future weekends.\n\
        - Copy values from the source text. NEVER invent one.\n\
        - If the source does not state a value, output an empty string \"\".\n\
        - start_date / end_date: ISO YYYY-MM-DD\n\
        - description: 1-2 sentence short summary of what makes it appealing for kids.\n\
        - Lines prefixed [THIS WEEKEND] definitely happen in the plan window; prefer them.\n\
        \n\
        Search results:\n\
        {corpus}\n\
        \n\
        Output ONLY JSON."
    );
    super::call_osaurus_json(&prompt, config).unwrap_or_default()
}

/// Run the full transient pipeline for a weekend: fetch -> prioritise ->
/// extract -> draft -> refine -> structure, with a monolithic fallback when a
/// phase yields nothing.
///
/// The model is warmed on its own thread WHILE the search runs, so a cold
/// start overlaps the network work instead of following it; if the warm-up
/// never answers, no phase is attempted and the health record says so.
///
/// Returns the structured events, the corpus they were judged against, and
/// the health record the plan's warning is written from.
#[must_use]
#[expect(
    clippy::option_if_let_else,
    reason = "the else branch is the monolithic-prompt fallback, with the \
              comment explaining why a dead draft phase must not starve the \
              plan. That belongs beside the branch, not inside a closure"
)]
pub fn fetch_duckduckgo_events(
    location: &str,
    d1: NaiveDate,
    d2: NaiveDate,
    weather_str: &str,
    ctx: &PlanContext,
    config: &crate::config::ZtoolsConfig,
) -> (Vec<WeekendEvent>, String, PlanHealth) {
    let warm = {
        let config = config.clone();
        std::thread::spawn(move || warm_model(&config))
    };
    let (corpus, search) = fetch_events_corpus(d1, d2, config);
    let model = warm.join().unwrap_or_else(|_| ModelHealth::Unavailable {
        model: config.weekend_model.clone(),
        reason: "the warm-up thread panicked".to_string(),
    });
    match &model {
        ModelHealth::Ready { model, secs } => println!("\u{2192} Model {model} ready ({secs}s)"),
        ModelHealth::Unavailable { model, reason } => {
            eprintln!("\u{26a0} Model {model} unavailable: {reason}; skipping extraction");
        }
    }
    let health = PlanHealth { search, model };
    if !health.model.is_ready() {
        return (Vec::new(), corpus, health);
    }

    let weather_condensed = condense_weather(weather_str, config);
    let cleaned = extract_sources(&corpus, location, config);

    let events = if let Some(draft) = draft_activities(&weather_condensed, &cleaned, ctx, config) {
        let refined = refine_draft(&draft, config);
        structure_to_json(&refined, &weather_condensed, ctx.year, config).unwrap_or_default()
    } else {
        // A dead draft phase must not starve the plan: fall back to the
        // monolithic prompt rather than returning nothing.
        monolithic_transient(&corpus, location, d1, d2, config)
    };

    (events, corpus, health)
}

/// Default Open-Meteo URL builder for Vaughan / GTA.
fn open_meteo_url(friday_date: &str, sunday_date: &str) -> String {
    format!(
        "https://api.open-meteo.com/v1/forecast?latitude=43.8361&longitude=-79.4982&daily=temperature_2m_max,precipitation_sum&timezone=America/New_York&start_date={friday_date}&end_date={sunday_date}"
    )
}

/// Fallback forecast when the fetch fails.
fn fallback_forecast() -> String {
    "Daily Forecast: Friday: 24.5°C Clear, Saturday: 26.0°C Clear, Sunday: 23.0°C Clear".to_string()
}

/// Parse the Open-Meteo JSON response into a forecast string.
#[must_use]
pub fn parse_weather_json(json: &serde_json::Value) -> Option<String> {
    let daily = json.get("daily")?;
    let times = daily.get("time").and_then(|t| t.as_array())?;
    let temps = daily.get("temperature_2m_max").and_then(|t| t.as_array())?;
    let precips = daily.get("precipitation_sum").and_then(|t| t.as_array())?;
    let mut lines = Vec::new();
    for (i, t_val) in times.iter().enumerate() {
        let t_str = t_val.as_str().unwrap_or("");
        let temp = temps
            .get(i)
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(22.0);
        let precip = precips
            .get(i)
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0);
        let cond = if precip > 0.5 {
            "Precipitation"
        } else {
            "Clear"
        };
        lines.push(format!("{t_str}: {temp:.1}°C, {cond} ({precip:.1}mm)"));
    }
    if lines.is_empty() {
        None
    } else {
        Some(format!("Daily Forecast:\n{}", lines.join("\n")))
    }
}

#[must_use]
pub fn fetch_weather(friday_date: &str, sunday_date: &str) -> String {
    let Ok(client) = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
    else {
        return fallback_forecast();
    };

    let url = open_meteo_url(friday_date, sunday_date);

    if let Ok(resp) = client.get(&url).send() {
        if let Ok(json) = resp.json::<serde_json::Value>() {
            if let Some(forecast) = parse_weather_json(&json) {
                return forecast;
            }
        }
    }

    fallback_forecast()
}
