//! The multi-phase weekend pipeline: extract -> draft -> refine -> structure.
//!
//! Ported from `weekend/phases.py`. Class C2c: these phases form a CARRIER
//! CHAIN. Each one must pass DATES/PRICE/AGES/LOCATION through verbatim -- the
//! predecessor narrowed the payload at every step, so an event's real dates
//! were gone two phases before the schema that asked for them, and every date
//! column rendered blank.

use std::cmp;

use super::PhaseLog;
use super::prompts::{
    ACTIVITY_RULE, CARRY_FIELDS, FAMILY_RULE, PHASE_DRAFT_TRANSIENT, PHASE_EXTRACT_EVENTS,
    PHASE_STRUCTURE_TRANSIENT_SYSTEM, PHASE_STRUCTURE_USER, PHASE_WEATHER_CONDENSE, WEATHER_RULE,
    render,
};

pub(crate) use super::calls::call_llm_json;
pub use super::calls::{call_llm_text, phase_retries};

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

/// The configured weekend model, checked against the server's roster.
///
/// THERE IS NO SUBSTITUTE. This used to fall back, silently, to a model of the
/// same "family", then to any raptor/qwen/gemma, then to `models.first()` —
/// whatever the server happened to list first. When the configured model was
/// not installed, every scheduled plan through 2026-10-09 was drafted by that
/// arbitrary model, extracted nothing, and said only "Plan Degraded" with no
/// hint why. A model that is not installed is now a stated failure the plan
/// names; the fix is a config change, and the plan says which.
///
/// `Ok(model)` when the roster lists it — or when the roster cannot be read
/// at all, since then nothing is known to be absent and the warm-up call will
/// say whether the server answers.
///
/// # Errors
///
/// [`super::ModelHealth::NotInstalled`] when the roster is readable and does
/// not list the configured model.
pub fn resolve_weekend_model(
    base_url: &str,
    configured: &str,
) -> Result<String, super::ModelHealth> {
    let url = format!("{}/v1/models", base_url.trim_end_matches('/'));
    let roster = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .ok()
        .and_then(|client| client.get(&url).send().ok())
        .filter(|r| r.status().is_success())
        .and_then(|r| r.json::<serde_json::Value>().ok());
    let installed: Vec<String> = roster
        .as_ref()
        .and_then(|json| json.get("data"))
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id").and_then(|id| id.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default();
    if installed.is_empty() || installed.iter().any(|m| m == configured) {
        return Ok(configured.to_string());
    }
    Err(super::ModelHealth::NotInstalled {
        model: configured.to_string(),
        installed,
    })
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
    let model = match resolve_weekend_model(&config.osaurus_url, &config.weekend_model) {
        Ok(model) => model,
        Err(not_installed) => return not_installed,
    };
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

/// Condense a forecast to 1-2 sentences; fall back to a preview on failure.
#[must_use]
pub fn condense_weather(
    weather_str: &str,
    config: &crate::config::ZtoolsConfig,
    log: &PhaseLog,
) -> String {
    let prompt = render(PHASE_WEATHER_CONDENSE, &[("weather_str", weather_str)]);
    let answer = call_llm_text(&prompt, config);
    log.record("condense weather", &prompt, answer.as_deref());
    answer.unwrap_or_else(|| weather_str.chars().take(WEATHER_PREVIEW_LIMIT).collect())
}

/// The fewest pipe-separated fields a row can have and still be about one
/// event: NAME | LOCATION | DATES.
const MIN_ROW_FIELDS: usize = 3;

/// The candidate rows in one extract answer, and nothing else.
///
/// A row is a line in the pipe format with at least [`MIN_ROW_FIELDS`] fields
/// and a NAME: not empty, not "unknown", not the format's own header echoed
/// back, not a markdown rule. Surrounding table pipes are tolerated.
/// Everything else in an answer is the model talking -- "I cannot extract
/// ...", "Per the instructions, I skipped ..." -- and is not a source the draft
/// should be shown as an event. Whether a NAMED row is an activity is not
/// decided here: that is the draft's judgement, made with the activity rule.
#[must_use]
pub fn extracted_rows(answer: &str) -> Vec<&str> {
    answer
        .lines()
        .map(str::trim)
        .filter(|line| {
            let inner = line.trim_matches('|');
            let fields: Vec<&str> = inner.split('|').map(str::trim).collect();
            fields.len() >= MIN_ROW_FIELDS && is_a_name(fields[0])
        })
        .collect()
}

fn is_a_name(field: &str) -> bool {
    let name = field.trim_start_matches(['-', '*', ' ']).trim();
    name.chars().any(char::is_alphanumeric)
        && !name.eq_ignore_ascii_case("unknown")
        && !name.eq_ignore_ascii_case("name")
}

/// Phase 1: pull clean pipe-separated lines out of the raw scraped corpus.
///
/// Batched with adaptive sizing: a streak of successes grows the batch, a
/// failure halves it and (at batch 1) falls back to passing the line through
/// raw rather than dropping it. The Python original persists batch sizes to a
/// signals file; this port keeps them in-memory per run, which is what the
/// shapes actually depend on.
///
/// THIS PHASE REFORMATS; IT NEVER REDUCES SUPPLY (`supply.rs`: a candidate
/// removed before the draft makes the model invent, or makes the plan thin).
/// Every candidate line is put to the model -- there used to be a cap of 24
/// unmarked lines, which on 2026-10-10 dropped 88 of 112 -- and a batch whose
/// answer holds no row ([`extracted_rows`]) reaches the draft as its raw
/// lines, exactly as a batch whose call failed does. A refusal is not an
/// answer about those lines: that run's extractor answered the batch naming
/// four in-window family events with "I cannot extract ...", the paragraph was
/// kept as the batch's result, and the plan had one event. Commentary in an
/// answer that does hold rows is dropped; the rows go on.
#[must_use]
pub fn extract_sources(
    raw_text: &str,
    location: &str,
    config: &crate::config::ZtoolsConfig,
    log: &PhaseLog,
) -> String {
    // In CORPUS ORDER, which `prioritise_in_window` already set: in-window
    // search results first, each followed page's lines together. This used to
    // float every marked line again here, which split a listing page's entry
    // from its own name a second time even once the corpus kept it whole.
    let lines: Vec<&str> = raw_text
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            t.starts_with("- ") || t.starts_with(super::IN_WINDOW_MARK)
        })
        .collect();
    if lines.is_empty() {
        return raw_text.to_string();
    }

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
            &[
                ("activity_rule", ACTIVITY_RULE),
                ("location", location),
                ("raw_text", &chunk),
            ],
        );
        let answer = call_llm_text(&prompt, config);
        log.record(
            &format!("extract lines {}-{end} of {}", i + 1, lines.len()),
            &prompt,
            answer.as_deref(),
        );
        if let Some(res) = answer {
            let rows = extracted_rows(&res);
            if rows.is_empty() {
                results.extend(lines[i..end].iter().map(ToString::to_string));
            } else {
                results.push(rows.join("\n"));
            }
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
    log: &PhaseLog,
) -> Option<String> {
    let prompt = render(
        PHASE_DRAFT_TRANSIENT,
        &[
            ("activity_rule", ACTIVITY_RULE),
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
    let answer = call_llm_text(&prompt, config);
    log.record("draft", &prompt, answer.as_deref());
    answer
}

/// Rows per structure call.
///
/// A structure answer is about 350 chars a row, and one call over every row is
/// a call whose size the draft decides: on a replay
/// of 2026-10-10 the draft returned ~80 rows, the structure prompt was 13.8K
/// chars, and the call produced nothing within its budget on either attempt --
/// every event lost at the last phase. Refine's old "keep the best 8" had been
/// bounding this by accident; its cap is gone (`refine.rs`), so the bound is
/// here, where the size matters.
pub const STRUCTURE_BATCH: usize = 10;

/// Phase 4: structure the refined rows into the transient-event JSON schema,
/// [`STRUCTURE_BATCH`] rows per call.
///
/// A REFORMAT, like extract: a batch whose call fails loses only its own rows,
/// says so on the operator's channel, and is in the phase transcript; the
/// other batches' events still reach the plan. `None` only when no batch
/// answered. Text with no pipe rows goes in one call, as it is.
///
/// Takes no forecast: the weather LABEL is a fact about where the activity
/// happens ([`WEATHER_RULE`]), and a forecast in this prompt is what turned a
/// hotel workshop "outdoor" under a clear sky.
#[must_use]
pub fn structure_to_json(
    text: &str,
    year: i32,
    config: &crate::config::ZtoolsConfig,
    log: &PhaseLog,
) -> Option<Vec<super::WeekendEvent>> {
    let sys = structure_system(year);
    let rows = extracted_rows(text);
    let batches: Vec<String> = if rows.is_empty() {
        vec![text.to_string()]
    } else {
        rows.chunks(STRUCTURE_BATCH).map(|c| c.join("\n")).collect()
    };
    let total = batches.len();
    let mut events: Option<Vec<super::WeekendEvent>> = None;
    for (n, batch) in batches.iter().enumerate() {
        let usr = structure_user(batch);
        let resp = call_llm_json(Some(&sys), &usr, config);
        log.record(
            &format!("structure batch {} of {total}", n + 1),
            &format!("{sys}\n\n{usr}"),
            resp.as_ref().and_then(answer_content),
        );
        match resp.as_ref().and_then(super::parse_llm_events) {
            Some(found) => events.get_or_insert_with(Vec::new).extend(found),
            None => eprintln!(
                "\u{26a0} structure batch {} of {total}: no usable answer; its {} row(s) are lost",
                n + 1,
                batch.lines().count()
            ),
        }
    }
    events
}

/// The schema and field rules every JSON event answer is asked under.
///
/// The structure phase and the monolithic fallback both send this, so neither
/// can drift from the other (the fallback's own inline copy once lacked the
/// weather rule, and every event it produced came back unlabelled). The model
/// eval sends it too (`eval::tasks`), so the json slot is ranked on this text.
#[must_use]
pub fn structure_system(year: i32) -> String {
    render(
        PHASE_STRUCTURE_TRANSIENT_SYSTEM,
        &[
            ("year", &year.to_string()),
            ("weather_rule", WEATHER_RULE),
            ("family_rule", FAMILY_RULE),
        ],
    )
}

/// The user message of one structure batch.
///
/// `rows` are pipe rows, `NAME | LOCATION | DATES | PRICE | AGES | description`.
/// One renderer for the
/// phase and the model eval, so the eval cannot rank a wording this phase
/// stopped sending.
#[must_use]
pub fn structure_user(rows: &str) -> String {
    render(PHASE_STRUCTURE_USER, &[("draft_text", rows)])
}

/// The answer text inside the completion shape [`call_llm_json`] returns.
pub(crate) fn answer_content(resp: &serde_json::Value) -> Option<&str> {
    resp["choices"][0]["message"]["content"].as_str()
}
