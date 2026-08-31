use chrono::NaiveDate;

use super::WeekendEvent;
use super::{
    _seasonal_keywords, condense_weather, draft_activities, extract_sources, in_window_count,
    prioritise_in_window, refine_draft, search_duckduckgo_html, structure_to_json, PlanContext,
};

/// Search the aggregator for event snippets and return the cleaned, deduped,
/// in-window-prioritised corpus. This is the ground truth the provenance gate
/// judges extracted rows against.
/// Build search queries for a target weekend starting on d1.
pub fn build_search_queries(d1: NaiveDate) -> Vec<String> {
    let month_name = d1.format("%B").to_string();
    let year = d1.format("%Y").to_string();

    let municipalities = ["Vaughan", "Markham", "Richmond Hill", "Toronto"];
    let mut queries = Vec::new();

    for city in municipalities {
        queries.push(format!("kids activities {} {} {}", city, month_name, year));
        queries.push(format!(
            "{} community centre kids programs {}",
            city, month_name
        ));
        queries.push(format!("{} family events {} {}", city, month_name, year));
    }

    let region = "GTA";
    queries.push(format!("{} family events {} {}", region, month_name, year));
    queries.push(format!(
        "{} Zoo special events {} {}",
        region, month_name, year
    ));
    queries.push(format!(
        "{} museum family programs {} {}",
        region, month_name, year
    ));
    if let Some(seasonal) = _seasonal_keywords(&month_name) {
        queries.push(format!("{} {} {} {}", region, seasonal, month_name, year));
    }
    queries
}

/// Search the aggregator for event snippets and return the cleaned, deduped,
/// in-window-prioritised corpus. This is the ground truth the provenance gate
/// judges extracted rows against.
fn fetch_events_corpus(
    d1: NaiveDate,
    d2: NaiveDate,
    config: &crate::config::ZtoolsConfig,
) -> String {
    let queries = build_search_queries(d1);

    let mut handles = Vec::new();
    for q in queries {
        let q_clone = q.clone();
        let url = config.duckduckgo_url.clone();
        handles.push(std::thread::spawn(move || {
            search_duckduckgo_html(&q_clone, &url)
        }));
    }

    let mut all_results = Vec::new();
    for h in handles {
        if let Ok(res) = h.join() {
            all_results.extend(res);
        }
    }

    let mut seen = std::collections::HashSet::new();
    let mut cleaned = Vec::new();
    for snippet in all_results {
        let norm = snippet.to_lowercase();
        if norm.is_empty() || seen.contains(&norm) {
            continue;
        }
        seen.insert(norm);

        if crate::ztools::weekend_cache::has_region_evidence(&snippet) {
            cleaned.push(format!("- Event: {}", snippet));
        }
    }
    let raw_text = cleaned.join("\n");

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
    marked_text
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
        "You are an expert family activity planner. Extract up to 10 time-limited events happening STRICTLY this weekend (between {} and {}) in {} from the text below.\n\
        Output JSON now. Use EXACT schema:\n\
        {{\"transient_events\": [{{\"name\": \"str\", \"location\": \"str\", \"target_ages\": \"str\", \"price\": \"str\", \"start_date\": \"str\", \"end_date\": \"str\", \"duration\": \"str\", \"weather\": \"str\", \"day\": \"str\", \"description\": \"str\"}}]}}\n\n\
        Rules for every field:\n\
        - Suggest up to 10 specific weekend activities. Do NOT stop after just 1 or 2 events. Find as many as you can.\n\
        - Only extract events that occur within or overlap with the dates {} to {}. Discard events from past or future weekends.\n\
        - Copy values from the source text. NEVER invent one.\n\
        - If the source does not state a value, output an empty string \"\".\n\
        - start_date / end_date: ISO YYYY-MM-DD\n\
        - description: 1-2 sentence short summary of what makes it appealing for kids.\n\
        - Lines prefixed [THIS WEEKEND] definitely happen in the plan window; prefer them.\n\
        \n\
        Search results:\n\
        {}\n\
        \n\
        Output ONLY JSON.",
        d1_str, d2_str, location, d1_str, d2_str, corpus
    );
    super::call_osaurus_json(&prompt, config).unwrap_or_default()
}

/// Run the full transient pipeline for a weekend: fetch -> prioritise ->
/// extract -> draft -> refine -> structure, with a monolithic fallback when a
/// phase yields nothing. Returns the structured events plus the corpus they
/// were judged against.
pub fn fetch_duckduckgo_events(
    location: &str,
    d1: NaiveDate,
    d2: NaiveDate,
    weather_str: &str,
    ctx: &PlanContext,
    config: &crate::config::ZtoolsConfig,
) -> (Vec<WeekendEvent>, String) {
    let corpus = fetch_events_corpus(d1, d2, config);

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

    (events, corpus)
}

/// Default Open-Meteo URL builder for Vaughan / GTA.
fn open_meteo_url(friday_date: &str, sunday_date: &str) -> String {
    format!(
        "https://api.open-meteo.com/v1/forecast?latitude=43.8361&longitude=-79.4982&daily=temperature_2m_max,precipitation_sum&timezone=America/New_York&start_date={}&end_date={}",
        friday_date, sunday_date
    )
}

/// Fallback forecast when the fetch fails.
fn fallback_forecast() -> String {
    "Daily Forecast: Friday: 24.5°C Clear, Saturday: 26.0°C Clear, Sunday: 23.0°C Clear".to_string()
}

/// Parse the Open-Meteo JSON response into a forecast string.
pub fn parse_weather_json(json: &serde_json::Value) -> Option<String> {
    let daily = json.get("daily")?;
    let times = daily.get("time").and_then(|t| t.as_array())?;
    let temps = daily.get("temperature_2m_max").and_then(|t| t.as_array())?;
    let precips = daily.get("precipitation_sum").and_then(|t| t.as_array())?;
    let mut lines = Vec::new();
    for (i, t_val) in times.iter().enumerate() {
        let t_str = t_val.as_str().unwrap_or("");
        let temp = temps.get(i).and_then(|v| v.as_f64()).unwrap_or(22.0);
        let precip = precips.get(i).and_then(|v| v.as_f64()).unwrap_or(0.0);
        let cond = if precip > 0.5 {
            "Precipitation"
        } else {
            "Clear"
        };
        lines.push(format!(
            "{}: {:.1}°C, {} ({:.1}mm)",
            t_str, temp, cond, precip
        ));
    }
    if lines.is_empty() {
        None
    } else {
        Some(format!("Daily Forecast:\n{}", lines.join("\n")))
    }
}

pub fn fetch_weather(friday_date: &str, sunday_date: &str) -> String {
    let client = match reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
    {
        Ok(c) => c,
        Err(_) => return fallback_forecast(),
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
