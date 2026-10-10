use chrono::NaiveDate;

use super::WeekendEvent;
use super::{DemotionPolicy, SearchRecord};
use super::{ModelHealth, PlanHealth, SearchHealth};
use super::{
    PlanContext, SearchResult, build_search_queries, condense_weather, draft_activities,
    extract_sources, in_window_count, prioritise_in_window, refine_draft, structure_to_json,
};
use super::{follow_aggregators, search_engines_in, warm_model};

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

/// The holiday inside `d1..=d2` by the configured province's table, if any.
/// No province (or no table for it) means no holiday query, never a failure:
/// the month-wide and window queries still run.
fn holiday_in_window(
    d1: NaiveDate,
    d2: NaiveDate,
    config: &crate::config::ZtoolsConfig,
) -> Option<&'static str> {
    let province = super::holidays::load_province(&config.weekend_region_paths).ok()?;
    d1.iter_days()
        .take_while(|d| *d <= d2)
        .find_map(|d| province.holiday_on(d))
        .map(|h| h.name)
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
    let queries = build_search_queries(d1, d2, holiday_in_window(d1, d2, config));

    // The order is learned from the runs before this one (search_order.rs):
    // an engine that walled most of its queries lately goes last.
    let record_path = crate::manifest::expand_tilde(&config.search_record_path);
    let mut record = SearchRecord::load(&record_path);
    let policy = DemotionPolicy::load(&config.weekend_region_paths);
    let order = record.order(&policy);
    if let Some(line) = record.describe(&policy, &order) {
        println!("\u{2192} Search order: {line}");
    }

    let mut all_results = Vec::<SearchResult>::new();
    let mut health = SearchHealth::default();
    for chunk in queries.chunks(4) {
        let mut handles = Vec::new();
        for q in chunk {
            let q_clone = q.clone();
            let urls = super::EngineUrls {
                duckduckgo: config.duckduckgo_url.clone(),
                bing: config.bing_url.clone(),
                brave: config.brave_url.clone(),
            };
            handles.push(std::thread::spawn(move || {
                search_engines_in(&q_clone, &urls, &order)
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
    record.record(&record_path, &d1.format("%Y-%m-%d").to_string(), &health);

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
    record_corpus(&raw_text, (d1, d2), &queries, (in_window, total));
    (marked_text, health)
}

/// Keep the corpus this run judged beside the plans (`store_corpus.rs`), so a
/// thin plan can be explained from what the engines actually returned. A
/// failure to write it is reported and never fails the run.
fn record_corpus(
    corpus: &str,
    (d1, d2): (NaiveDate, NaiveDate),
    queries: &[String],
    (in_window, total): (usize, usize),
) {
    let mut header = format!("# window {d1}..{d2}\n# in-window {in_window}/{total}");
    for q in queries {
        header.push_str("\n# query ");
        header.push_str(q);
    }
    let store = crate::ztools::store::weekend_output_dir();
    let run = chrono::Local::now().naive_local();
    match crate::ztools::store_corpus::save_corpus(&store, run, &header, corpus) {
        Ok(path) => println!("\u{2192} Corpus kept at {}", path.display()),
        Err(e) => eprintln!("\u{26a0} corpus not kept under {}: {e}", store.display()),
    }
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
        ModelHealth::NotInstalled { model, installed } => {
            eprintln!(
                "\u{26a0} Model {model} is not installed (server lists: {}); \
                 skipping extraction -- set weekend_model to an installed model",
                installed.join(", ")
            );
        }
    }
    let health = PlanHealth {
        search,
        model,
        provenance: super::Provenance::default(),
    };
    if !health.model.is_ready() {
        return (Vec::new(), corpus, health);
    }

    let weather_condensed = condense_weather(weather_str, config);
    let cleaned = extract_sources(&corpus, location, config);

    let events = draft_activities(&weather_condensed, &cleaned, ctx, config).map_or_else(
        // A dead draft phase must not starve the plan: fall back to the
        // monolithic prompt rather than returning nothing.
        || monolithic_transient(&corpus, location, d1, d2, config),
        |draft| {
            let refined = refine_draft(&draft, config);
            structure_to_json(&refined, &weather_condensed, ctx.year, config).unwrap_or_default()
        },
    );

    (events, corpus, health)
}

/// The forecast URL for the weekend window, against `endpoint`.
///
/// `endpoint` is an ORIGIN — scheme, host, and any path prefix — exactly as
/// [`crate::config::ZtoolsConfig::weather_url`] holds it, so a mirror, a proxy or
/// a self-hosted instance substitutes the host and still gets Open-Meteo's
/// `/v1/forecast` contract. The trailing slash is TRIMMED rather than
/// concatenated: an operator who writes `https://host/` meant one slash, and
/// `//v1/forecast` is a 404 that reads as a dead forecast.
fn open_meteo_url(endpoint: &str, friday_date: &str, sunday_date: &str) -> String {
    format!(
        "{}/v1/forecast?latitude=43.8361&longitude=-79.4982&daily=temperature_2m_max,precipitation_sum&timezone=America/New_York&start_date={friday_date}&end_date={sunday_date}",
        endpoint.trim_end_matches('/')
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

/// Default Open-Meteo forecast for the weekend window, or [`fallback_forecast`]
/// when the configured endpoint is unreachable or the body does not parse.
///
/// Takes `config` rather than a bare endpoint string because every OTHER
/// third-party host this module reaches is threaded the same way
/// (`fetch_events_corpus` reads `duckduckgo_url`/`bing_url`/`brave_url` off it),
/// and because a caller with no override must not be able to forget to have one.
#[must_use]
pub fn fetch_weather(
    friday_date: &str,
    sunday_date: &str,
    config: &crate::config::ZtoolsConfig,
) -> String {
    fetch_weather_from(&open_meteo_url(
        &config.weather_url,
        friday_date,
        sunday_date,
    ))
}

/// [`fetch_weather`] against an explicit endpoint.
///
/// The window and the endpoint are separate inputs, so the URL is the seam and
/// the caller supplies it — the same shape as
/// [`fetch_page_text`](super::fetch_page_text) and
/// [`resolve_weekend_model`](super::resolve_weekend_model). A stub bound to
/// `127.0.0.1:0` therefore drives the whole round trip (client build, GET,
/// decode, and the fallback) instead of only the parser.
#[must_use]
pub fn fetch_weather_from(url: &str) -> String {
    let Ok(client) = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
    else {
        eprintln!("\u{26a0} forecast client not built; rendering the fixed forecast");
        return fallback_forecast();
    };

    // The fallback is indistinguishable from a real forecast to every caller,
    // so every failure here was silent: a dead endpoint, a refused
    // certificate and an HTML error page all rendered a plausible 24.5°C and
    // nothing said which. Name the failure class instead, and use `{e:?}` over
    // `{e}` because reqwest's `Display` stops at "error sending request" --
    // the certificate verdict lives in the source chain only.
    match client.get(url).send() {
        Ok(resp) => match resp.json::<serde_json::Value>() {
            Ok(json) => parse_weather_json(&json).unwrap_or_else(|| {
                eprintln!("\u{26a0} forecast from {url} carried no daily block");
                fallback_forecast()
            }),
            Err(e) => {
                eprintln!("\u{26a0} forecast from {url} did not decode as JSON: {e:?}");
                fallback_forecast()
            }
        },
        Err(e) => {
            eprintln!("\u{26a0} forecast fetch failed for {url}: {e:?}");
            fallback_forecast()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_env::TestEnv;
    use serial_test::serial;

    /// The whole content of the URL builder is its query, so it is pinned
    /// whole: the window must reach the wire in the right slots and the right
    /// ORDER (an inverted range is a window the endpoint will not honour, and
    /// nothing else in the suite sees these two arguments).
    ///
    /// The origin is read off the DEFAULT CONFIG rather than written here, so
    /// this test also covers the join -- the value the default hands over is
    /// the one the builder is handed -- and cannot drift from
    /// `config::tests::the_default_weather_endpoint_is_the_host_that_was_hardcoded`,
    /// which pins those same bytes.
    ///
    /// Reading the default config is what makes this a hazard test, so it takes
    /// the sandbox with it: `test_env::audit` is right that a config's `~/…`
    /// defaults are resolved through a process-global `dirs::home_dir()`, and a
    /// test that reads one without the lock can observe another test's home.
    #[test]
    #[serial]
    fn the_weekend_window_reaches_the_forecast_endpoint_in_order() {
        let env = TestEnv::new();
        assert_eq!(
            open_meteo_url(
                crate::config::ZtoolsConfig::default().weather_url.as_str(),
                "2026-08-07",
                "2026-08-09"
            ),
            concat!(
                "https://api.open-meteo.com/v1/forecast",
                "?latitude=43.8361&longitude=-79.4982",
                "&daily=temperature_2m_max,precipitation_sum",
                "&timezone=America/New_York",
                "&start_date=2026-08-07&end_date=2026-08-09",
            )
        );
        drop(env);
    }

    /// The ORIGIN substitutes; the path and the window do not. This is the arm
    /// the hardcoded literal could not have: a mirror or a self-hosted instance
    /// has to land on the same contract, and a trailing slash an operator typed
    /// must not become `//v1/forecast` — which is a 404, and a 404 reads as a
    /// dead forecast behind a fallback that looks like weather.
    ///
    /// The expectation is a LITERAL rather than built from the endpoint on
    /// purpose: an expectation assembled from the same value the function trims
    /// agrees with the function whether or not the trim happens, which is a
    /// test that cannot fail (calibrated — removing `trim_end_matches` left it
    /// green).
    #[test]
    fn the_configured_origin_replaces_the_host_and_survives_a_trailing_slash() {
        let expected = concat!(
            "http://127.0.0.1:9/v1/forecast",
            "?latitude=43.8361&longitude=-79.4982",
            "&daily=temperature_2m_max,precipitation_sum",
            "&timezone=America/New_York",
            "&start_date=2026-08-07&end_date=2026-08-09",
        );
        for endpoint in ["http://127.0.0.1:9", "http://127.0.0.1:9/"] {
            assert_eq!(
                open_meteo_url(endpoint, "2026-08-07", "2026-08-09"),
                expected,
                "an endpoint of {endpoint:?} must not change the path or the window"
            );
        }
    }
}
