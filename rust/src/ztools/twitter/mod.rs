//! Native Rust Twitter summarizer and browser collection module.

pub mod browser;
pub mod browser_parse;
pub mod cookies;

pub use browser::{BrowserCollector, CamoufoxConfig, MockBrowserCollector};
pub use browser_parse::parse_tweets_from_response;
pub use cookies::{
    find_firefox_profile_dbs, has_session_cookie, Cookie, DEFAULT_DOMAINS, SESSION_COOKIE_NAME,
};

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use chrono::Local;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Tweet {
    pub screen_name: String,
    pub text: String,
    pub created_at: String,
    pub favorite_count: u64,
    pub retweet_count: u64,
    pub reply_to: Option<String>,
}

/// Deduplicate tweets by normalized text content and RT signatures.
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
            prefix_parts.push(format!("-> @{}", r));
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
        <timeline>\n{}\n</timeline>",
        timeline
    );

    (prompt, lines.len())
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    temperature: f64,
    stream: bool,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Deserialize)]
struct ChatMessageResponse {
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

/// Call local Osaurus LLM server at localhost:1337.
pub fn call_osaurus(
    base_url: &str,
    model: &str,
    prompt: &str,
    config: &crate::config::ZtoolsConfig,
) -> Result<String> {
    let client = Client::builder()
        .timeout(Duration::from_secs(config.llm_extended_timeout_secs))
        .build()?;

    let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));
    let req = ChatRequest {
        model,
        messages: vec![ChatMessage {
            role: "user",
            content: prompt,
        }],
        temperature: 0.0,
        stream: false,
    };

    let raw = client
        .post(&url)
        .json(&req)
        .send()
        .context("Failed to send request to Osaurus server")?
        .text()
        .context("Failed to read Osaurus server response text")?;

    let resp: ChatResponse = serde_json::from_str(&raw)
        .map_err(|e| anyhow::anyhow!("Failed to parse Osaurus server response JSON: {e}, raw: {raw}"))?;

    let text = resp
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .unwrap_or_default();

    Ok(text)
}

/// Run full Twitter summary flow and save markdown artifact.
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
    eprintln!("· Summarizing {} tweets with {} on {}...", processed, model, base_url);
    let summary_body = call_osaurus(base_url, model, &prompt, config)?;

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
         **Model:** {}\n\n\
         ## Summary\n\n\
         {}\n",
        now.format("%Y-%m-%d %H:%M UTC"),
        total,
        processed,
        model,
        summary_body.trim()
    );

    fs::write(&out_path, content)?;
    Ok(out_path)
}

/// Validate summary formatting quality (headers, bullet count, length).
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
        warnings.push(format!("Only {} bullet points", bullet_count));
    }
    if char_count < 100 {
        warnings.push(format!("Very short ({} chars)", char_count));
    }

    let critical = header_count == 0 && bullet_count == 0;
    (warnings, critical)
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
