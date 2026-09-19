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
pub mod session;

pub use browser::{BrowserCollector, CamoufoxConfig, MockBrowserCollector};
pub use browser_parse::parse_tweets_from_response;
pub use cookies::{
    find_firefox_profile_dbs, has_session_cookie, Cookie, DEFAULT_DOMAINS, SESSION_COOKIE_NAME,
};

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
        if clean.to_lowercase().starts_with("rt @") {
            if let Some(pos) = clean.find(':') {
                clean = clean[pos + 1..].trim().to_string();
            }
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
    crate::ztools::llm::chat(
        &crate::ztools::llm::ChatRequest {
            base_url,
            model,
            system: None,
            user: prompt,
            json: false,
        },
        &config.chat_budget(timeout_secs),
    )
}

/// Run full Twitter summary flow and save markdown artifact.
///
/// # Errors
///
/// From the model call, and from writing the summary to the output
/// directory.
pub fn run_summary(
    tweets: &[Tweet],
    output_dir: Option<&Path>,
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
        if cache_path.exists() {
            if let Ok(text) = fs::read_to_string(&cache_path) {
                if let Ok(parsed) = serde_json::from_str::<Vec<Tweet>>(&text) {
                    tweets_vec = parsed;
                }
            }
        }
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
    let ((summary_body, _), provenance) = chain::run_chain(&plan, model, |candidate| {
        eprintln!("· Summarizing {processed} tweets with {candidate} on {base_url}...");
        let raw = call_osaurus(base_url, candidate, &prompt, timeout_secs, config)?;
        // A reasoning model's `<thinking>` block must not land verbatim in
        // the saved markdown, and an unstructured answer must not be saved
        // as success: a critical-quality attempt yields nothing, which the
        // chain spends on the next model.
        Ok(handle_model_output(&raw, processed))
    })?;
    if provenance.degraded() {
        eprintln!("⚠ Degraded summary: {}", provenance.describe());
        for reason in &provenance.reasons {
            eprintln!("  → {reason}");
        }
    }

    let now = Local::now();
    let filename = format!("{}_summary.md", now.format("%Y-%m-%d_%H%M"));
    let default_dir = crate::ztools::store::twitter_store_dir();
    let dir = output_dir.unwrap_or(&default_dir);
    fs::create_dir_all(dir)?;
    let out_path = dir.join(filename);

    let total = tweets_vec.len();
    let content = format!(
        "# Twitter Timeline Summary\n\n\
         **Period:** {}\n\
         **Tweets:** {} fetched, {} processed\n\
         {}\n\n\
         {}\n",
        now.format("%Y-%m-%d %H:%M UTC"),
        total,
        processed,
        provenance.banner(),
        summary_section_for(&summary_body)
    );

    fs::write(&out_path, content)?;
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

/// One model attempt's post-call branch: split thinking, then merge or strip.
///
/// The UNMERGED body faces the quality gate, never the merged text: an
/// appended `## Analysis` heading must not rescue an empty body. A critical
/// body yields nothing even when thinking is present; the caller tries the
/// next model. Port of `_summarize_with_model`'s post-call branch in
/// `references/twitter/summarize.py`; the transport stays with the caller so
/// this remains unit-testable without a server.
#[must_use]
pub fn handle_model_output(content: &str, processed: usize) -> Option<(String, usize)> {
    let (thinking, cleaned) = extract_thinking(content);
    if thinking.is_empty() {
        let stripped = crate::ztools::eval::clean::remove_thinking_blocks(&cleaned);
        let (_warnings, critical) = check_summary_quality(&stripped);
        if critical {
            None
        } else {
            Some((stripped, processed))
        }
    } else {
        let (_warnings, critical) = check_summary_quality(&cleaned);
        if critical {
            None
        } else {
            Some((merge_thinking_with_summary(&thinking, &cleaned), processed))
        }
    }
}

/// Validate summary formatting quality (headers, bullet count, length).
#[must_use]
pub fn check_summary_quality(summary: &str) -> (Vec<String>, bool) {
    if summary.trim().is_empty() {
        return (vec!["Summary is empty".to_string()], true);
    }
    let mut warnings = Vec::new();
    let mut header_count = 0;
    let mut bullet_count = 0;
    let mut char_count = 0;

    for line in summary.lines() {
        let stripped = line.trim();
        char_count += stripped.len();
        if stripped.starts_with("##") {
            header_count += 1;
        } else if stripped.starts_with("- ") || stripped.starts_with("* ") {
            bullet_count += 1;
        }
    }

    if header_count == 0 {
        warnings.push("No ## headers".to_string());
    }
    if bullet_count < 3 {
        warnings.push(format!("Only {bullet_count} bullet points"));
    }
    if char_count < 100 {
        warnings.push(format!("Very short ({char_count} chars)"));
    }

    let critical = header_count == 0 && bullet_count == 0;
    (warnings, critical)
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
