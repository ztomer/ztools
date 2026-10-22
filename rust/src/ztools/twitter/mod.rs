//! Native Rust Twitter summarizer and browser collection module.

pub mod browser;
pub mod browser_bin;
pub mod browser_parse;
pub mod budget;
pub mod capture;
pub mod chain;
pub mod collect;
pub mod cookies;
pub mod endpoints;
pub mod fallback;
pub mod native;
pub mod quality;
mod rejected;
pub mod repair;
pub use repair::{Answer, MIN_BULLETS, handle_model_output};
pub mod session;

pub use browser::{BrowserCollector, CamoufoxConfig, MockBrowserCollector};
pub use browser_parse::parse_tweets_from_response;
pub use cookies::{
    Cookie, DEFAULT_DOMAINS, SESSION_COOKIE_NAME, find_firefox_profile_dbs, has_session_cookie,
};
pub use quality::{Quality, Refusal, check_summary_quality, drop_recited, unmatched_citations};

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use chrono::Local;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Tweet {
    /// Tweet ID (`legacy.id_str`). The A/B gate compares ID sets across
    /// collectors, so this must survive parsing — a dropped ID reads as a
    /// missing tweet. Defaulted for old cache files written before it existed.
    #[serde(default)]
    pub id: String,
    pub screen_name: String,
    pub text: String,
    pub created_at: String,
    pub favorite_count: u64,
    pub retweet_count: u64,
    pub reply_to: Option<String>,
}

/// Deduplicate tweets by normalized text content and RT signatures.
#[must_use]
pub fn deduplicate_tweets(tweets: &[Tweet]) -> Vec<Tweet> {
    let mut seen_sigs = HashSet::new();
    let mut deduped = Vec::new();

    for t in tweets {
        let text = t.text.trim();
        if text.is_empty() {
            continue;
        }

        // Clean RT prefix and URLs
        let mut clean = text.to_string();
        if clean.to_lowercase().starts_with("rt @")
            && let Some(pos) = clean.find(':')
        {
            clean = clean[pos + 1..].trim().to_string();
        }

        // Strip URLs and non-alphanumeric chars for signature
        let mut norm = String::new();
        for ch in clean.chars() {
            if ch.is_alphanumeric() || ch.is_whitespace() {
                norm.push(ch.to_ascii_lowercase());
            }
        }
        let norm_sig: String = norm.split_whitespace().collect::<Vec<_>>().join(" ");
        let sig: String = norm_sig.chars().take(90).collect();
        if sig.chars().count() < 15 {
            deduped.push(t.clone());
            continue;
        }

        if seen_sigs.contains(&sig) {
            continue;
        }

        seen_sigs.insert(sig);
        deduped.push(t.clone());
    }

    deduped
}

/// Build executive summary prompt for LLM. `instructions` is the shared
/// instruction block (canonical text: `conf/prompts.toml` `[twitter.summarize]`).
#[must_use]
pub fn build_prompt(tweets: &[Tweet], max_chars: usize, instructions: &str) -> (String, usize) {
    let deduped = deduplicate_tweets(tweets);
    let mut lines = Vec::new();
    let mut used = 0;

    for t in deduped.iter().rev() {
        let mut prefix_parts = vec![format!("@{} | {}", t.screen_name, t.created_at)];
        if t.favorite_count > 0 || t.retweet_count > 0 {
            prefix_parts.push(format!(
                "{} favs, {} RTs",
                t.favorite_count, t.retweet_count
            ));
        }
        if let Some(ref r) = t.reply_to {
            prefix_parts.push(format!("-> @{r}"));
        }
        let line = format!("[{}]: {}", prefix_parts.join(" | "), t.text.trim());
        if used + line.len() + 1 > max_chars {
            continue;
        }
        used += line.len() + 1;
        lines.push(line);
    }
    lines.reverse();
    let timeline = lines.join("\n");

    let prompt = format!(
        "{instructions}\n\n\
        <timeline>\n{timeline}\n</timeline>"
    );

    (prompt, lines.len())
}

/// How the summarizer decodes: not greedy, and no frequency penalty.
///
/// At temperature 0 the ~1B-active summarizer fell into repetition loops (43
/// tweets became 58 bullets, 35 of them repeats), and greedy decoding is the
/// documented loop-prone setting (`docs/MODEL_QUIRKS.md`). A small temperature
/// breaks the tie that sustains a loop.
///
/// A `frequency_penalty` of 0.3 was tried and REMOVED the same day: the
/// citation `(@handle | Sat Oct 10 15:33:43 +0000 2026)` repeats the same
/// tokens in every bullet, and both live runs with the penalty garbled exactly
/// those -- `Sat Oct 10 336:51`, `+0000 2о` with a Cyrillic о, `Sat Oct 1`. A
/// penalty taxes the format the gate requires. The gate (`quality.rs`), not
/// sampling, is what keeps a loop or a garbled citation out of the store.
pub const SUMMARY_SAMPLING: crate::ztools::llm::Sampling = crate::ztools::llm::Sampling {
    temperature: 0.1,
    frequency_penalty: None,
};

/// Environment override for the summary's temperature, for one run: the seam
/// the loop-rate measurement (`docs/MODEL_QUIRKS.md`, roadmap T2) runs
/// through, like `TWITTER_FALLBACK_MODELS` for the chain.
pub const TEMPERATURE_ENV: &str = "TWITTER_TEMPERATURE";

/// [`SUMMARY_SAMPLING`], or the temperature `TWITTER_TEMPERATURE` asks for when
/// it is a number a server takes (0 to 2). Anything else keeps the shipped
/// policy and says so.
#[must_use]
pub fn summary_sampling() -> crate::ztools::llm::Sampling {
    let Ok(raw) = std::env::var(TEMPERATURE_ENV) else {
        return SUMMARY_SAMPLING;
    };
    match raw.trim().parse::<f64>() {
        Ok(t) if (0.0..=2.0).contains(&t) => crate::ztools::llm::Sampling {
            temperature: t,
            ..SUMMARY_SAMPLING
        },
        _ => {
            eprintln!(
                "\u{26a0} {TEMPERATURE_ENV}={raw:?} is not a temperature from 0 to 2; using {}",
                SUMMARY_SAMPLING.temperature
            );
            SUMMARY_SAMPLING
        }
    }
}

/// Call the local Osaurus server for one plain-text answer.
///
/// Streams through [`crate::ztools::llm::chat`]: `timeout_secs` is the CAP
/// on a call whose tokens keep flowing; the stall guard is the configured
/// default, because a call that stopped producing is failed the same way
/// whoever made it.
///
/// # Errors
///
/// When the request cannot be sent, the stream stalls, or the body is not
/// the API's shape (see `ztools::llm`).
pub fn call_osaurus(
    base_url: &str,
    model: &str,
    prompt: &str,
    timeout_secs: u64,
    config: &crate::config::ZtoolsConfig,
) -> Result<String> {
    crate::ztools::llm::chat_with(
        &crate::ztools::llm::ChatRequest {
            base_url,
            model,
            system: None,
            user: prompt,
            json: false,
        },
        &config.chat_budget(timeout_secs),
        &summary_sampling(),
    )
}

/// Where one candidate is asked: the on-device model by `fm` when the policy
/// routes it there, every other model by the server.
struct Route<'a> {
    policy: &'a chain::FallbackPolicy,
    base_url: &'a str,
    timeout_secs: u64,
    config: &'a crate::config::ZtoolsConfig,
}

impl Route<'_> {
    /// The candidate's raw answer and whether it was cut at the length limit.
    fn ask(&self, candidate: &str, prompt: &str, processed: usize) -> Result<(String, bool)> {
        if let Some(fm) = self.policy.on_device_for(candidate) {
            eprintln!("· Summarizing {processed} tweets with {candidate} on this Mac...");
            let raw = crate::ztools::on_device::respond(fm, prompt, self.timeout_secs)?;
            return Ok((raw, false));
        }
        eprintln!(
            "· Summarizing {processed} tweets with {candidate} on {}...",
            self.base_url
        );
        repair::ask(
            self.base_url,
            candidate,
            prompt,
            self.timeout_secs,
            self.config,
        )
    }
}

/// The run's error when every model in `plan` failed.
///
/// When every model is served by the ONE server at `base_url`, "every model
/// failed" while it is down is one failure, not N: say so, or the log reads as
/// N independent model faults. A chain with an on-device model did have
/// somewhere else to go.
fn chain_failure(
    e: anyhow::Error,
    plan: &[String],
    policy: &chain::FallbackPolicy,
    base_url: &str,
) -> anyhow::Error {
    if plan.iter().any(|m| policy.on_device_for(m).is_some()) {
        return e.context("no summary written: every model in the fallback chain failed");
    }
    e.context(format!(
        "no summary written: every model in the fallback chain ({}) is served by the same \
         Osaurus server at {base_url}, so the chain has no fallback outside that server \
         (set conf/twitter.toml [fallback] on_device to reach foundation without it)",
        plan.join(", ")
    ))
}

/// Run full Twitter summary flow and save the markdown into `output_dir`.
///
/// The directory is REQUIRED, never defaulted here: whether a run writes the
/// production store (`store::twitter_output_dir`, which the dashboard and the
/// status page read) is the caller's decision, made once where the tweet
/// source is known (`cli_ztools_twitter::destination_for`). A library default
/// to the real store is how test and `--json` runs became the summary the
/// dashboard showed.
///
/// # Errors
///
/// When there is no tweet to summarise, when every model in the chain failed
/// or was rejected by the quality gate, and from writing the summary.
pub fn run_summary(
    tweets: &[Tweet],
    output_dir: &Path,
    base_url: Option<&str>,
    model: Option<&str>,
    config: &crate::config::ZtoolsConfig,
) -> Result<PathBuf> {
    let default_url = config.osaurus_url.clone();
    let default_model = config.twitter_model.clone();
    let base_url = base_url.unwrap_or(&default_url);
    let model = model.unwrap_or(&default_model);

    let mut tweets_vec = tweets.to_vec();
    if tweets_vec.is_empty() {
        let cache_path = crate::manifest::expand_tilde(&config.twitter_cache_path);
        if cache_path.exists()
            && let Ok(text) = fs::read_to_string(&cache_path)
            && let Ok(parsed) = serde_json::from_str::<Vec<Tweet>>(&text)
        {
            tweets_vec = parsed;
        }
    }

    // Nothing to summarise is a refusal, not a document: five "Please provide
    // the timeline" answers were saved as summaries of 0 tweets in August 2026.
    if tweets_vec.is_empty() {
        anyhow::bail!("no tweets to summarise: nothing was collected, and no summary was written");
    }
    let deduped = deduplicate_tweets(&tweets_vec);
    let clustered = crate::ztools::embeddings::cluster_tweets(&deduped, base_url, config)
        .unwrap_or_else(|_| deduped.iter().map(|t| vec![t.clone()]).collect());

    // Re-flatten from clusters, ordering by cluster size to put biggest narratives first
    let mut sorted_clusters = clustered;
    sorted_clusters.sort_by_key(|c| std::cmp::Reverse(c.len()));

    let mut final_tweets = Vec::new();
    for cluster in sorted_clusters {
        for t in cluster {
            final_tweets.push(t);
        }
    }

    let (prompt, processed) = build_prompt(
        &final_tweets,
        config.twitter_prompt_max_chars,
        &config.twitter_summarize_prompt,
    );
    // What the answer may cite: exactly the source lines the prompt showed.
    let sources: Vec<(String, String)> = final_tweets[..processed.min(final_tweets.len())]
        .iter()
        .map(|t| (t.screen_name.clone(), t.created_at.clone()))
        .collect();
    // The chain: intended model resolved against what the server serves,
    // then the configured fallbacks. A roster the server will not give us is
    // an unknown roster, not an empty one — the intent stands unfiltered.
    let policy = chain::load_fallback_policy(&config.twitter_config_paths)?;
    let available =
        crate::ztools::model_eval::get_available_models(base_url, config).unwrap_or_default();
    let plan = chain::plan_chain(model, &available, &policy);
    let timeout_secs = budget::estimate_timeout(
        prompt.chars().count(),
        &budget::TimeoutInputs::pessimistic(),
    );
    let (answer, mut provenance) = chain::run_chain(&plan, model, |candidate| {
        let route = Route {
            policy: &policy,
            base_url,
            timeout_secs,
            config,
        };
        let (raw, cut) = route
            .ask(candidate, &prompt, processed)
            .map_err(|e| Refusal {
                verdict: "the call failed".to_string(),
                detail: format!("the call failed: {e}"),
            })?;
        // A reasoning model's `<thinking>` block must not land verbatim in
        // the saved markdown, and an answer the quality gate rejects must not
        // be saved at all: the rejection is this model's recorded reason, and
        // the chain moves on to the next model.
        handle_model_output(&raw, &sources, cut)
            .map(Some)
            .map_err(|refusal| {
                let stamp = Local::now().format("%Y-%m-%d_%H%M%S").to_string();
                // The refused answer is kept with the FULL reason in its
                // header: the banner carries the verdict, this file carries
                // the finding.
                let kept = rejected::keep(output_dir, &stamp, candidate, &raw, &refusal.detail)
                    .map(|p| format!(" (answer kept at {})", p.display()))
                    .unwrap_or_default();
                Refusal {
                    verdict: refusal.verdict,
                    detail: format!("the answer was refused: {}{kept}", refusal.detail),
                }
            })
    })
    .map_err(|e| chain_failure(e, &plan, &policy, base_url))?;
    provenance.backend = policy.backend_of(&provenance.model);
    if provenance.degraded() {
        // The same line the document opens with, then every failure in full:
        // the line is what a reader sees, the reasons are the run record.
        eprintln!("⚠ {}", provenance.notice(&answer.repairs));
        for reason in provenance.reasons() {
            eprintln!("  → {reason}");
        }
    }

    let now = Local::now();
    let filename = format!("{}_summary.md", now.format("%Y-%m-%d_%H%M"));
    fs::create_dir_all(output_dir)?;
    let out_path = output_dir.join(filename);

    let total = tweets_vec.len();
    // The blank line before `{banner}` is load-bearing, not spacing: without it
    // `**Model:**` is a lazy continuation of the `**Tweets:**` paragraph and
    // every CommonMark renderer folds the two into one run-on line. `>` may
    // interrupt a paragraph, so the degraded banner happened to survive; the
    // quiet one did not.
    let content = format!(
        "# Twitter Timeline Summary\n\n\
         **Period:** {}\n\
         **Tweets:** {} fetched, {} processed\n\n\
         {}\n\n\
         {}\n",
        // The local clock, labelled with its real offset. It used to carry a
        // literal "UTC" after a LOCAL time, which was wrong by the offset.
        now.format("%Y-%m-%d %H:%M %:z"),
        total,
        processed,
        provenance.banner(&answer.repairs),
        summary_section_for(&answer.body)
    );

    fs::write(&out_path, content)?;
    crate::ztools::settings::retain(output_dir, crate::ztools::settings::TWITTER_RETENTION);
    Ok(out_path)
}

/// The section under the summary header. The `## Summary` preamble exists only
/// to give a heading to a body that has none: the model's output already opens
/// with `## Executive Summary`, and prepending the preamble on top of it
/// rendered a heading with nothing underneath — two titles, one blank pane, on
/// every dashboard page (the Swift renderer skips such sections, but the
/// writer should not mint them in the first place).
fn summary_section_for(summary_body: &str) -> String {
    let body = summary_body.trim();
    if body.is_empty()
        || body.starts_with("# ")
        || body.starts_with("## ")
        || body.starts_with("### ")
    {
        body.to_string()
    } else {
        format!("## Summary\n\n{body}")
    }
}

/// Split a `<thinking>...</thinking>` block out of model output.
///
/// Returns the stripped thinking and the body with thinking blocks removed.
/// With no block, returns the text untouched (not stripped). Port of
/// `lib/osaurus_lib.py::extract_thinking`.
#[must_use]
pub fn extract_thinking(text: &str) -> (String, String) {
    static THINKING_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?s)<thinking[^>]*>(.+?)</thinking>").expect("valid regex"));
    let Some(caps) = THINKING_RE.captures(text) else {
        return (String::new(), text.to_string());
    };
    let thinking = caps
        .get(1)
        .map(|m| m.as_str().trim().to_string())
        .unwrap_or_default();
    let cleaned = crate::ztools::eval::clean::remove_thinking_blocks(text);
    (thinking, cleaned)
}

/// Append extracted thinking under an `## Analysis` heading. Empty thinking
/// returns the summary unchanged. Port of
/// `lib/osaurus_lib.py::merge_thinking_with_summary`.
#[must_use]
pub fn merge_thinking_with_summary(thinking: &str, summary: &str) -> String {
    if thinking.is_empty() {
        summary.to_string()
    } else {
        format!("{summary}\n\n## Analysis\n{thinking}")
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
