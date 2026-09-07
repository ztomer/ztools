//! The `twitter-summarize` command, and where it finds tweets.
//!
//! Split out of `cli_ztools` for the 500-line cap, along the seam that was
//! already there: this command owns its own options struct and three private
//! sources (a JSON blob, a file, the first non-empty cache candidate) that
//! nothing else calls.

use std::path::PathBuf;

use anyhow::Result;

use crate::config::ZtoolsConfig;

/// The `twitter-summarize` flag set, grouped so the function signature stays
/// one stable struct instead of a growing argument list.
pub(crate) struct TwitterSummarizeOpts {
    pub json: Option<String>,
    pub model: Option<String>,
    pub md_out: Option<PathBuf>,
    pub use_cache: bool,
    pub fetch_only: bool,
    pub debug: bool,
    pub since: Option<String>,
    pub login: bool,
    pub fetch_latest: bool,
    pub last_updated: bool,
}

/// Parse a tweet array, or nothing.
///
/// Deliberately lossy in one direction only: unparseable input yields an empty
/// list so the caller falls through to its other sources, and never a partial
/// one. Half a timeline read as the whole timeline is a summary that is
/// confidently wrong about what was said.
fn tweets_from_json(text: &str) -> Vec<crate::ztools::twitter::Tweet> {
    serde_json::from_str::<Vec<crate::ztools::twitter::Tweet>>(text).unwrap_or_default()
}

/// Tweets from one file, or nothing when it is absent or unreadable.
fn tweets_from_file(path: &std::path::Path) -> Vec<crate::ztools::twitter::Tweet> {
    match std::fs::read_to_string(path) {
        Ok(content) => tweets_from_json(&content),
        Err(_) => Vec::new(),
    }
}

/// The first cache file that yields a NON-EMPTY tweet list, and where it came
/// from.
///
/// The non-empty condition is the point: an empty cache file is not an answer,
/// and stopping at one would report "0 cached tweets" while a populated
/// candidate sat unread behind it. Returning the path too is what lets the
/// caller say which file it used instead of leaving the operator to guess
/// between two hard-coded locations.
///
/// The candidate list is a parameter rather than being built from `$HOME`
/// inside, so every branch here is provable without writing into the
/// developer's own home directory.
fn tweets_from_cache(
    candidates: &[PathBuf],
) -> Option<(PathBuf, Vec<crate::ztools::twitter::Tweet>)> {
    for candidate in candidates {
        let parsed = tweets_from_file(candidate);
        if !parsed.is_empty() {
            return Some((candidate.clone(), parsed));
        }
    }
    None
}

pub(crate) fn twitter_summarize(config: &ZtoolsConfig, opts: TwitterSummarizeOpts) -> Result<()> {
    let TwitterSummarizeOpts {
        json,
        model,
        md_out,
        use_cache,
        fetch_only,
        debug,
        since,
        login,
        fetch_latest,
        last_updated,
    } = opts;
    if fetch_latest || last_updated {
        return crate::ztools::store::twitter_latest(last_updated);
    }
    if login {
        println!("· Launching browser for x.com sign-in...");
        return crate::ztools::twitter::browser::login_live();
    }

    let mut tweets = Vec::new();
    let mut explicit_source = false;

    if let Some(path_or_dash) = json {
        if path_or_dash == "-" {
            let mut buffer = String::new();
            if std::io::Read::read_to_string(&mut std::io::stdin(), &mut buffer).is_ok() {
                tweets = tweets_from_json(&buffer);
            }
        } else {
            tweets = tweets_from_file(std::path::Path::new(&path_or_dash));
        }
        explicit_source = true;
    }

    if !explicit_source {
        if use_cache {
            let candidates: Vec<PathBuf> = [
                dirs::home_dir().map(|h| h.join(".twitter_summary_debug_cache.json")),
                dirs::home_dir().map(|h| h.join(".cache/twitter/debug_tweets.json")),
            ]
            .into_iter()
            .flatten()
            .collect();
            if let Some((from, cached)) = tweets_from_cache(&candidates) {
                println!(
                    "· Using {} cached tweets from {}",
                    cached.len(),
                    from.display()
                );
                tweets = cached;
            }
            if tweets.is_empty() {
                anyhow::bail!(
                    "No cached tweets found. Run without --use-cache first to scrape live tweets."
                );
            }
        } else {
            tweets = crate::ztools::twitter::browser::collect_tweets_live(since.as_deref(), debug)?;
            if tweets.is_empty() {
                println!("· No tweets found in the timeline window.");
                return Ok(());
            }
            if fetch_only {
                println!(
                    "✓ {} tweets fetched and cached. (--fetch-only provided, exiting)",
                    tweets.len()
                );
                return Ok(());
            }
        }
    }

    let path = crate::ztools::twitter::run_summary(&tweets, None, None, model.as_deref(), config)?;
    if let Ok(doc) = std::fs::read_to_string(&path) {
        println!("{doc}");
    }
    println!("✓ twitter summary generated at {}", path.display());
    if let Some(out_path) = md_out {
        std::fs::copy(&path, &out_path)?;
        println!("✓ copy saved to {}", out_path.display());
    }
    Ok(())
}

#[cfg(test)]
#[path = "cli_ztools_twitter_tests.rs"]
mod tests;
