//! The eval task roster — port of `eval/tasks_core.py::TASKS`.
//!
//! Every task is a prompt (from [`super::prompts`], the canonical text) plus
//! ONE graded validator carrying the source it judges against. Names, prompt
//! composition, `parse_json` flags and validator arguments follow the Python
//! table row for row, so a sweep measures the same thing it measured before
//! the port. `json` is the alias the table also carried.
//!
//! THE WEEKEND ROWS ARE THE EXCEPTION TO "THE PYTHON TABLE" (W5, 2026-10-10):
//! they rank the `json` slot, which is the model the planner runs on, so they
//! send the planner's own prompts through its own renderers -- see
//! [`weekend_tasks`]. `weekend_fixed`, `weekend_fixed_mixed` and their
//! `detailed_json` alias were retired with that change: they asked for a
//! year-round venue list production has never structured with a model.
//!
//! The single non-constant input is the filename task's text, read from
//! `conf/eval_inputs.toml [test_inputs].filename` — data, like the Python
//! `get_eval_input("filename")`. The vision task's images are drawn from
//! `conf/eval_vision.toml` at load time (see [`super::vision`]).

use anyhow::{Context, Result};
use chrono::Datelike;
use std::path::Path;

use super::graded::Graded;
use super::prompts::file_summary;
use super::prompts::{
    CONTRADICTION_PHRASE, FALSEHOOD_PHRASES, FILE_SUMMARY_PROMPT, FILE_SUMMARY_PROMPT_MIXED,
    FILENAME_INJECTION_KEYWORDS, FILENAME_INJECTION_MARKERS, FILENAME_INJECTION_PROMPT,
    IMAGE_RENAME_PROMPT, IMAGE_RENAME_PROMPT_MIXED, KEY_FACTS, RENAME_PROMPT, RENAME_PROMPT_MIXED,
    RENAME_TEXT_SLOT, SUMMARIZE_INJECTION_KEYWORDS, SUMMARIZE_INJECTION_MARKERS, TWITTER_PROMPT,
    TWITTER_PROMPT_ACCURACY, TWITTER_PROMPT_CONTRADICTION, TWITTER_PROMPT_INJECTION,
    TWITTER_PROMPT_MISATTRIBUTION, TWITTER_PROMPT_MIXED, WEEKEND_EVAL_LOCATION,
    WEEKEND_FABRICATION_LURES, WEEKEND_FABRICATION_RESULTS, WEEKEND_INJECTION_KEYWORDS,
    WEEKEND_INJECTION_MARKERS, WEEKEND_INJECTION_ROWS, WEEKEND_NOISE_RESULTS,
    WEEKEND_SEARCH_RESULTS, WEEKEND_SIGNAL_EVENTS, WEEKEND_STRUCTURE_ROWS, weekend_eval_window,
};
use super::task_loader::{ChatMessage, Check, EvalTask};

/// The file-summary tasks' system prompt, verbatim from the Python table.
const FILE_SUMMARY_SYSTEM: &str = "Output JSON now. No preamble, no markdown.\n\n\
Required format: {\"path\": \"description\", ...} OR [{\"path\": \"x\", \"desc\": \"y\"}, ...]\n\n\
Summarize each file in one line. Be specific - mention actual functionality, not just file type.";

/// The vision task's prompt, verbatim from the Python table.
const IMAGE_REAL_PROMPT: &str = "Describe each of the three images separately, in a few words \
each. Name the main colour and shape of each. No preamble.";

/// `RENAME_PROMPT` rendered with a real input — never the bare template, which
/// once asked the model to summarise the literal string `{text}` and scored
/// 100 on a shape-only validator.
fn filename_prompt_for(text: &str) -> String {
    RENAME_PROMPT.replace(RENAME_TEXT_SLOT, text)
}

/// `conf/eval_inputs.toml [test_inputs].<task>`.
///
/// # Errors
///
/// When the file or the key is missing: a task built on an absent input would
/// send the bare template, the exact defect the render exists to prevent.
pub fn eval_input(inputs_file: &Path, task: &str) -> Result<String> {
    let path = inputs_file;
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("eval inputs at {}", path.display()))?;
    let val: toml::Value = toml::from_str(&text).context("eval_inputs.toml parses")?;
    val.get("test_inputs")
        .and_then(|t| t.get(task))
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .with_context(|| format!("eval_inputs.toml has no [test_inputs].{task}"))
}

fn strs(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| (*s).to_string()).collect()
}

fn graded(check: Graded) -> Vec<Check> {
    vec![Check::Graded(check)]
}

fn user_task(name: &str, prompt: &str, check: Graded) -> EvalTask {
    EvalTask::new(name, prompt, graded(check))
}

fn system_task(name: &str, system: &str, user: &str, check: Graded) -> EvalTask {
    EvalTask::with_system(name, system, user, graded(check))
}

fn src(g: impl Fn(String) -> Graded, source: &str) -> Graded {
    g(source.to_string())
}

/// `check` behind the planner's parse gate: an answer the weekend planner
/// would drop scores 0 whatever its content.
fn planner(check: Graded) -> Graded {
    Graded::PlannerEvents {
        inner: Box::new(check),
    }
}

/// One structure batch exactly as `weekend::phases::structure_to_json` sends it:
/// `(system, user)`.
fn weekend_structure_prompt() -> (String, String) {
    let (start, _) = weekend_eval_window();
    (
        crate::ztools::weekend::structure_system(start.year()),
        crate::ztools::weekend::structure_user(WEEKEND_STRUCTURE_ROWS),
    )
}

/// The monolithic fallback exactly as `weekend::fetch` sends it: the search
/// results floated and marked by `prioritise_in_window`, as
/// `fetch_events_corpus` hands them on, under the same system prompt.
fn weekend_monolithic_prompt() -> (String, String) {
    weekend_monolithic_over(WEEKEND_SEARCH_RESULTS)
}

/// The monolithic fallback over `results`, as `weekend::fetch` sends it.
fn weekend_monolithic_over(results: &str) -> (String, String) {
    let (start, end) = weekend_eval_window();
    let corpus = crate::ztools::weekend::prioritise_in_window(results, start, end);
    (
        crate::ztools::weekend::structure_system(start.year()),
        crate::ztools::weekend::monolithic_user(&corpus, WEEKEND_EVAL_LOCATION, start, end),
    )
}

/// Rows 1-2: the planner's two JSON calls.
///
/// `weekend_transient` is one structure batch -- a reformat of ten carried
/// rows, graded for count, detail, distinct names, grounding in the rows, and
/// the placeholder caps (generic location, constant column, near duplicate).
/// `weekend_transient_mixed` is the monolithic fallback over search results with
/// noise among them -- the one production JSON call that must FILTER, graded
/// for precision and recall against the stated signal and noise.
fn weekend_tasks() -> Vec<EvalTask> {
    let (system, rows) = weekend_structure_prompt();
    let (mono_system, search) = weekend_monolithic_prompt();
    vec![
        system_task(
            "weekend_transient",
            &system,
            &rows,
            planner(Graded::DetailedJson {
                source: rows.clone(),
            }),
        )
        .json(),
        system_task(
            "weekend_transient_mixed",
            &mono_system,
            &search,
            planner(Graded::MixedSignal {
                source: search.clone(),
                signal: strs(WEEKEND_SIGNAL_EVENTS),
                noise: strs(WEEKEND_NOISE_RESULTS),
            }),
        )
        .json(),
    ]
}

/// The file-summary prompts with every listed file's own content spliced in.
///
/// The prompt instructs the model to rely ONLY on provided content and never to
/// infer from a name; until 2026-10-05 no content was provided, so the rule was
/// unfollowable and `validate_file_summary` graded name guesses. Rendering here —
/// at roster build time, where `filename_prompt_for` already renders the one
/// other data-driven prompt — is the existing seam, so the same `EvalTask` shape,
/// the same transport and the same validator see a prompt that can be obeyed.
///
/// THE CHECKOUT IS A PARAMETER, RESOLVED ONCE IN [`roster`]. The rows are
/// repo-relative, so the renderer needs a root; resolving it here (rather than
/// inside the renderer) is what makes ONE resolution serve both rows AND the
/// scorer's copy of the prompt — see the `source` invariant below.
///
/// # Errors
///
/// When no checkout holds every listed row, or when a row cannot be read, or a
/// template lost its content slot: all three would ship a prompt that names files
/// and explains none of them, or explains a DIFFERENT checkout's files, which is
/// the defect. Failing is the honest outcome; there is no degraded form to fall
/// back to, because a path-only prompt is exactly what this replaced.
fn file_summary_tasks(checkout: &Path) -> Result<Vec<EvalTask>> {
    let (plain, mixed) =
        file_summary::render_both_from(checkout, FILE_SUMMARY_PROMPT, FILE_SUMMARY_PROMPT_MIXED)?;
    Ok(vec![
        system_task(
            "file_summary",
            FILE_SUMMARY_SYSTEM,
            &plain,
            Graded::FileSummary,
        )
        .json(),
        system_task(
            "file_summary_mixed",
            FILE_SUMMARY_SYSTEM,
            &mixed,
            Graded::MixedFileSummary {
                // The rendered bytes, not the bytes the files hold now. The
                // validator reads its signal set OUT of this text, so a
                // re-read at score time would grade the model against a prompt
                // it was never shown.
                source: mixed.clone(),
            },
        ),
    ])
}

/// Rows 3-11: filename, image-rename, summarize and file-summary, plain and
/// mixed, in the table's interleaved order.
///
/// Built by pushing rather than by one `vec![]` literal because the two
/// file-summary rows come from [`file_summary_tasks`], which renders file
/// contents and can fail. The table's ORDER is pinned by `tasks_tests.rs`, and it
/// is the Python table's order — the two file-summary rows are NOT adjacent there
/// (`rename_mixed` and `summarize_mixed` sit between them), so they are placed
/// by index rather than by being written next to each other.
fn text_tasks(
    filename_input: &str,
    filename_prompt: &str,
    checkout: &Path,
) -> Result<Vec<EvalTask>> {
    let mixed_filename = |name: &str, prompt: &str| {
        user_task(
            name,
            prompt,
            src(|source| Graded::MixedFilename { source }, prompt),
        )
        .json()
    };
    let mut tasks = vec![
        user_task(
            "filename",
            filename_prompt,
            src(|source| Graded::Filename { source }, filename_input),
        ),
        mixed_filename("image_rename", IMAGE_RENAME_PROMPT),
        user_task(
            "summarize",
            TWITTER_PROMPT,
            src(|source| Graded::Summary { source }, TWITTER_PROMPT),
        ),
    ];
    let mut file_summary = file_summary_tasks(checkout)?.into_iter();
    tasks.push(
        file_summary
            .next()
            .expect("file_summary_tasks yields the plain row first"),
    );
    tasks.extend([
        mixed_filename("rename_mixed", RENAME_PROMPT_MIXED),
        user_task(
            "summarize_mixed",
            TWITTER_PROMPT_MIXED,
            src(
                |source| Graded::MixedSummary { source },
                TWITTER_PROMPT_MIXED,
            ),
        ),
    ]);
    tasks.push(
        file_summary
            .next()
            .expect("file_summary_tasks yields the mixed row second"),
    );
    tasks.extend([
        mixed_filename("filename_mixed", RENAME_PROMPT_MIXED),
        mixed_filename("image_rename_mixed", IMAGE_RENAME_PROMPT_MIXED),
    ]);
    // Two rows, two slots: the iterator is exhausted on purpose, so the surplus
    // is checked rather than dropped -- a third row added to
    // `file_summary_tasks` would otherwise vanish from the roster silently.
    let surplus: Vec<EvalTask> = file_summary.collect();
    assert!(
        surplus.is_empty(),
        "a file-summary row with no slot in the pinned order: {:?}",
        surplus.iter().map(|t| t.name.as_str()).collect::<Vec<_>>()
    );
    Ok(tasks)
}

/// Rows 12-22: the faithfulness, vision and adversarial probes.
fn probe_tasks(filename_prompt: &str, images: Vec<String>) -> Vec<EvalTask> {
    let (system, rows) = weekend_structure_prompt();
    let (fab_system, fab_listing) = weekend_monolithic_over(WEEKEND_FABRICATION_RESULTS);
    let injection_rows = crate::ztools::weekend::structure_user(WEEKEND_INJECTION_ROWS);
    vec![
        system_task(
            "weekend_transient_schema",
            &system,
            &rows,
            planner(Graded::StrictSchema {
                kind: "json".to_string(),
            }),
        ),
        user_task(
            "summarize_contradiction",
            TWITTER_PROMPT_CONTRADICTION,
            Graded::NoContradiction {
                phrase: CONTRADICTION_PHRASE.to_string(),
            },
        ),
        user_task("filename_leak", filename_prompt, Graded::NoLeak),
        user_task(
            "summarize_factual_accuracy",
            TWITTER_PROMPT_ACCURACY,
            Graded::FactualAccuracy {
                falsehoods: strs(FALSEHOOD_PHRASES),
            },
        ),
        EvalTask {
            name: "image_real".to_string(),
            messages: vec![ChatMessage::user_with_images(IMAGE_REAL_PROMPT, images)],
            checks: graded(Graded::ImageDescription),
            parse_json: false,
        },
        // M20: the monolithic fallback over a six-result listing, under its
        // own "up to 10" -- production's padding pressure, not an eval count.
        system_task(
            "weekend_fabrication",
            &fab_system,
            &fab_listing,
            planner(Graded::NoFabrication {
                source: fab_listing.clone(),
                lures: strs(WEEKEND_FABRICATION_LURES),
            }),
        )
        .json(),
        user_task(
            "filename_injection",
            FILENAME_INJECTION_PROMPT,
            Graded::ResistsInjection {
                source: FILENAME_INJECTION_PROMPT.to_string(),
                markers: strs(FILENAME_INJECTION_MARKERS),
                keywords: strs(FILENAME_INJECTION_KEYWORDS),
            },
        ),
        user_task(
            "summarize_misattribution",
            TWITTER_PROMPT_MISATTRIBUTION,
            src(
                |source| Graded::Attribution { source },
                TWITTER_PROMPT_MISATTRIBUTION,
            ),
        ),
        user_task(
            "summarize_factual_coverage",
            TWITTER_PROMPT,
            Graded::FactualCoverage {
                key_facts: strs(KEY_FACTS),
            },
        ),
        // Each slot's own injection gate (2026-09-19):
        // `filename_injection` had been the proxy for tools that read tweets
        // and search snippets, and it measured a different shape.
        user_task(
            "summarize_injection",
            TWITTER_PROMPT_INJECTION,
            Graded::ResistsInjection {
                source: TWITTER_PROMPT_INJECTION.to_string(),
                markers: strs(SUMMARIZE_INJECTION_MARKERS),
                keywords: strs(SUMMARIZE_INJECTION_KEYWORDS),
            },
        ),
        // M20: one structure batch with the instruction planted in a row.
        system_task(
            "weekend_injection",
            &system,
            &injection_rows,
            planner(Graded::ResistsInjection {
                source: injection_rows.clone(),
                markers: strs(WEEKEND_INJECTION_MARKERS),
                keywords: strs(WEEKEND_INJECTION_KEYWORDS),
            }),
        )
        .json(),
    ]
}

/// The two data files the roster is built from, resolved separately so a
/// user overlay may carry one without the other.
#[derive(Debug, Clone)]
pub struct RosterInputs {
    /// `eval_inputs.toml`
    pub inputs: std::path::PathBuf,
    /// `eval_vision.toml`
    pub vision: std::path::PathBuf,
}

impl RosterInputs {
    /// Both files under one directory (the shipped `conf/` layout).
    #[must_use]
    pub fn in_dir(conf_dir: &Path) -> Self {
        Self {
            inputs: conf_dir.join("eval_inputs.toml"),
            vision: conf_dir.join("eval_vision.toml"),
        }
    }
}

/// The roster, in the Python table's order, plus its `json` alias.
///
/// THE FILE-SUMMARY CHECKOUT IS RESOLVED ONCE, HERE, ABOVE EVERY ROW THAT NEEDS
/// IT. `file_summary::live_root` asks the checkout derivation which candidate
/// actually holds every listed row; doing it once rather than inside the renderer
/// is what lets the SAME root render both the plain and the mixed prompt and
/// hand the mixed bytes to the scorer as its source — a second resolution could
/// only return a different answer, and then the validator would read its signal
/// set out of a prompt the model was never shown.
///
/// # Errors
///
/// When the filename input or the vision fixtures cannot be loaded — both
/// are data this roster refuses to guess — when no checkout holds every file the
/// file-summary prompts list, or when one of them cannot be read.
pub fn roster(files: &RosterInputs) -> Result<Vec<EvalTask>> {
    let checkout = file_summary::live_root()?;
    let filename_input = eval_input(&files.inputs, "filename")?;
    let filename_prompt = filename_prompt_for(&filename_input);
    let vision = super::vision::load_vision_spec(&files.vision)?;
    let images = super::vision::fixture_images(&vision)?;

    let mut tasks = weekend_tasks();
    tasks.extend(text_tasks(&filename_input, &filename_prompt, &checkout)?);
    tasks.extend(probe_tasks(&filename_prompt, images));
    // The table's alias: `json` is weekend_transient under the slot's name.
    // `detailed_json` was weekend_fixed's, and went with it (W5).
    let mut json_alias = tasks[0].clone();
    json_alias.name = "json".to_string();
    tasks.push(json_alias);
    Ok(tasks)
}

#[cfg(test)]
#[path = "tasks_file_summary_tests.rs"]
mod file_summary_render_tests;

#[cfg(test)]
#[path = "tasks_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "tasks_weekend_tests.rs"]
mod weekend_parity_tests;
