//! The eval task roster — port of `eval/tasks_core.py::TASKS`.
//!
//! Every task is a prompt (from [`super::prompts`], the canonical text) plus
//! ONE graded validator carrying the source it judges against. Names, prompt
//! composition, `parse_json` flags and validator arguments follow the Python
//! table row for row, so a sweep measures the same thing it measured before
//! the port. `json` and `detailed_json` are the aliases the table also carried.
//!
//! The single non-constant input is the filename task's text, read from
//! `conf/eval_inputs.toml [test_inputs].filename` — data, like the Python
//! `get_eval_input("filename")`. The vision task's images are drawn from
//! `conf/eval_vision.toml` at load time (see [`super::vision`]).

use anyhow::{Context, Result};
use std::path::Path;

use super::graded::Graded;
use super::prompts::file_summary;
use super::prompts::{
    CONTRADICTION_PHRASE, FALSEHOOD_PHRASES, FILE_SUMMARY_PROMPT, FILE_SUMMARY_PROMPT_MIXED,
    FILENAME_INJECTION_KEYWORDS, FILENAME_INJECTION_MARKERS, FILENAME_INJECTION_PROMPT,
    IMAGE_RENAME_PROMPT, IMAGE_RENAME_PROMPT_MIXED, KEY_FACTS, RENAME_PROMPT, RENAME_PROMPT_MIXED,
    RENAME_TEXT_SLOT, SUMMARIZE_INJECTION_KEYWORDS, SUMMARIZE_INJECTION_MARKERS, TWITTER_PROMPT,
    TWITTER_PROMPT_ACCURACY, TWITTER_PROMPT_CONTRADICTION, TWITTER_PROMPT_INJECTION,
    TWITTER_PROMPT_MISATTRIBUTION, TWITTER_PROMPT_MIXED, WEEKEND_FABRICATION_LURES,
    WEEKEND_FABRICATION_PROMPT, WEEKEND_INJECTION_KEYWORDS, WEEKEND_INJECTION_MARKERS,
    WEEKEND_INJECTION_PROMPT, WEEKEND_SYS_FIXED, WEEKEND_SYS_TRANSIENT, WEEKEND_USR_FIXED,
    WEEKEND_USR_FIXED_MIXED, WEEKEND_USR_TRANSIENT, WEEKEND_USR_TRANSIENT_MIXED,
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

/// Rows 1-4: weekend extraction, plain and mixed.
fn weekend_tasks() -> Vec<EvalTask> {
    vec![
        system_task(
            "weekend_transient",
            WEEKEND_SYS_TRANSIENT,
            WEEKEND_USR_TRANSIENT,
            src(
                |source| Graded::DetailedJson { source },
                WEEKEND_USR_TRANSIENT,
            ),
        )
        .json(),
        system_task(
            "weekend_fixed",
            WEEKEND_SYS_FIXED,
            WEEKEND_USR_FIXED,
            src(|source| Graded::DetailedJson { source }, WEEKEND_USR_FIXED),
        )
        .json(),
        system_task(
            "weekend_transient_mixed",
            WEEKEND_SYS_TRANSIENT,
            WEEKEND_USR_TRANSIENT_MIXED,
            src(
                |source| Graded::MixedSignal { source },
                WEEKEND_USR_TRANSIENT_MIXED,
            ),
        )
        .json(),
        system_task(
            "weekend_fixed_mixed",
            WEEKEND_SYS_FIXED,
            WEEKEND_USR_FIXED_MIXED,
            src(
                |source| Graded::MixedSignal { source },
                WEEKEND_USR_FIXED_MIXED,
            ),
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
/// # Errors
///
/// When a listed file cannot be read, or a template lost its content slot: both
/// would ship a prompt that names files and explains none of them, which is the
/// defect. Failing is the honest outcome; there is no degraded form to fall back
/// to, because a path-only prompt is exactly what this replaced.
fn file_summary_tasks() -> Result<Vec<EvalTask>> {
    let (plain, mixed) = file_summary::render_both(FILE_SUMMARY_PROMPT, FILE_SUMMARY_PROMPT_MIXED)?;
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
                source: mixed.clone(),
            },
        ),
    ])
}

/// Rows 5-13: filename, image-rename, summarize and file-summary, plain and
/// mixed, in the table's interleaved order.
///
/// Built by pushing rather than by one `vec![]` literal because the two
/// file-summary rows come from [`file_summary_tasks`], which renders file
/// contents and can fail. The table's ORDER is pinned by `tasks_tests.rs`, and it
/// is the Python table's order — the two file-summary rows are NOT adjacent there
/// (`rename_mixed` and `summarize_mixed` sit between them), so they are placed
/// by index rather than by being written next to each other.
fn text_tasks(filename_input: &str, filename_prompt: &str) -> Result<Vec<EvalTask>> {
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
    let mut file_summary = file_summary_tasks()?.into_iter();
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

/// Rows 14-22: the faithfulness, vision and adversarial probes.
fn probe_tasks(filename_prompt: &str, images: Vec<String>) -> Vec<EvalTask> {
    vec![
        system_task(
            "weekend_transient_schema",
            WEEKEND_SYS_TRANSIENT,
            WEEKEND_USR_TRANSIENT,
            Graded::StrictSchema {
                kind: "json".to_string(),
            },
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
        user_task(
            "weekend_fabrication",
            WEEKEND_FABRICATION_PROMPT,
            Graded::NoFabrication {
                source: WEEKEND_FABRICATION_PROMPT.to_string(),
                lures: strs(WEEKEND_FABRICATION_LURES),
            },
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
        user_task(
            "weekend_injection",
            WEEKEND_INJECTION_PROMPT,
            Graded::ResistsInjection {
                source: WEEKEND_INJECTION_PROMPT.to_string(),
                markers: strs(WEEKEND_INJECTION_MARKERS),
                keywords: strs(WEEKEND_INJECTION_KEYWORDS),
            },
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

/// The roster, in the Python table's order, plus its two aliases.
///
/// # Errors
///
/// When the filename input or the vision fixtures cannot be loaded — both
/// are data this roster refuses to guess.
pub fn roster(files: &RosterInputs) -> Result<Vec<EvalTask>> {
    let filename_input = eval_input(&files.inputs, "filename")?;
    let filename_prompt = filename_prompt_for(&filename_input);
    let vision = super::vision::load_vision_spec(&files.vision)?;
    let images = super::vision::fixture_images(&vision)?;

    let mut tasks = weekend_tasks();
    tasks.extend(text_tasks(&filename_input, &filename_prompt)?);
    tasks.extend(probe_tasks(&filename_prompt, images));
    // The table's aliases: `json` is weekend_transient, `detailed_json` is
    // weekend_fixed, under the names the model slots are measured by.
    let mut json_alias = tasks[0].clone();
    json_alias.name = "json".to_string();
    let mut detailed_alias = tasks[1].clone();
    detailed_alias.name = "detailed_json".to_string();
    tasks.push(json_alias);
    tasks.push(detailed_alias);
    Ok(tasks)
}

#[cfg(test)]
#[path = "tasks_file_summary_tests.rs"]
mod file_summary_render_tests;

#[cfg(test)]
#[path = "tasks_tests.rs"]
mod tests;
