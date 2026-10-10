//! The weekend phases' calls to the model: retried, every failure named, and
//! a line-per-row answer cut at the token limit kept to its whole lines.
//!
//! Split from `phases.rs` (the 500-line cap); `phases` re-exports both calls,
//! so their callers are unchanged.

/// How many times a phase call is retried after a transport failure or an
/// empty answer, from `conf/weekend.toml [llm] phase_retries`.
///
/// The Python planner retried five times; the stall-guarded client removed
/// the main cause (abandoned calls that wedged the server), so the default is
/// modest. A retry is never free — each one is another bounded call against
/// a serial server — and a phase that fails twice running is not flaky.
#[must_use]
pub fn phase_retries(paths: &[String]) -> u32 {
    for raw in paths {
        let path = crate::manifest::expand_tilde(raw);
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(val) = toml::from_str::<toml::Value>(&content) else {
            continue;
        };
        if let Some(n) = val
            .get("llm")
            .and_then(|l| l.get("phase_retries"))
            .and_then(toml::Value::as_integer)
        {
            return u32::try_from(n).unwrap_or(0);
        }
    }
    1
}

/// Run `call` up to `1 + retries` times until it answers, saying on the
/// operator's channel each time it did not, and why.
fn with_retries<T>(
    what: &str,
    retries: u32,
    mut call: impl FnMut() -> anyhow::Result<T>,
) -> Option<T> {
    for attempt in 0..=retries {
        match call() {
            Ok(v) => return Some(v),
            // The cause, every time: "no answer" alone hid that the 17:17 draft
            // was cut at the token limit, twice.
            Err(e) if attempt < retries => eprintln!(
                "\u{26a0} {what}: {e:#} (attempt {}); retrying ({} left)",
                attempt + 1,
                retries - attempt
            ),
            Err(e) => eprintln!(
                "\u{26a0} {what}: {e:#} (attempt {}); giving up",
                attempt + 1
            ),
        }
    }
    None
}

/// The whole lines of an answer cut at the token limit, without the line the
/// cut fell in; `None` when no line is whole.
fn whole_lines(partial: &str) -> Option<String> {
    let whole = partial
        .rsplit_once('\n')
        .map_or("", |(head, _)| head)
        .trim_end();
    (!whole.trim().is_empty()).then(|| whole.to_string())
}

/// One plain-text LLM call against the configured osaurus endpoint, retried
/// per `phase_retries`.
///
/// Every caller asks for one row per line (extract, draft, refine) or a line
/// of prose (the forecast), so an answer cut at the token limit keeps its
/// whole lines, and says how many: each is a complete row (W3).
#[must_use]
pub fn call_llm_text(prompt: &str, config: &crate::config::ZtoolsConfig) -> Option<String> {
    // The configured model, verbatim: `warm_model` checked it against the
    // roster before any phase ran, and no phase runs when it is missing.
    let model = &config.weekend_model;
    let retries = phase_retries(&config.weekend_region_paths);
    with_retries("llm text call", retries, || {
        let answer = match crate::ztools::twitter::call_osaurus(
            &config.osaurus_url,
            model,
            prompt,
            config.llm_extended_timeout_secs,
            config,
        ) {
            Ok(answer) => answer,
            Err(e) => {
                let Some(cut) = e.downcast_ref::<crate::ztools::llm::CutAnswer>() else {
                    return Err(e);
                };
                let Some(kept) = whole_lines(&cut.partial) else {
                    return Err(e);
                };
                println!(
                    "\u{2192} llm text call: {cut}; kept its {} whole line(s)",
                    kept.lines().count()
                );
                kept
            }
        };
        if answer.trim().is_empty() {
            anyhow::bail!("the model answered with nothing");
        }
        Ok(answer)
    })
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
    let model = &config.weekend_model;
    let retries = phase_retries(&config.weekend_region_paths);
    let content = with_retries("llm json call", retries, || {
        crate::ztools::llm::chat(
            &crate::ztools::llm::ChatRequest {
                base_url: &config.osaurus_url,
                model,
                system,
                user,
                json: true,
            },
            &config.chat_budget(config.llm_extended_timeout_secs),
        )
    })?;
    Some(serde_json::json!({"choices": [{"message": {"content": content}}]}))
}

#[cfg(test)]
#[path = "phases_cut_tests.rs"]
mod cut_tests;
