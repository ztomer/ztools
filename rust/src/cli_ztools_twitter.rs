//! The `twitter-summarize` command, and where it finds tweets.
//!
//! Split out of `cli_ztools` for the 500-line cap, along the seam that was
//! already there: this command owns its own options struct and three private
//! sources (a JSON blob, a file, the first non-empty cache candidate) that
//! nothing else calls.

use std::path::PathBuf;

use anyhow::Result;

use crate::config::ZtoolsConfig;

/// What `twitter-summarize` was asked to do. The CLI accepts every flag
/// together; `cli::run` resolves their precedence once, at the dispatch,
/// instead of the body doing it by the order of its early returns.
pub(crate) enum TwitterCommand {
    /// `--fetch-latest` / `--last-updated`: the stored summary, read-only.
    Latest { last_updated: bool },
    /// `--login`: open the browser to sign in to x.com.
    Login,
    /// `--clean`: delete the stored summaries.
    Clean,
    /// The run itself.
    Summarize(TwitterSummarizeOpts),
}

/// Where the tweets come from. `--json` names a source the caller chose, so
/// the cache and a live fetch are never fallbacks for it.
pub(crate) enum TweetSource {
    /// A JSON file path, or `-` for stdin.
    Json(String),
    /// `--use-cache`: the previous live fetch.
    Cache,
    /// A live timeline scrape; `fetch_only` stops after caching it.
    Live {
        since: Option<String>,
        debug: bool,
        fetch_only: bool,
    },
}

/// The summarize run: its source and its two outputs.
pub(crate) struct TwitterSummarizeOpts {
    pub source: TweetSource,
    pub model: Option<String>,
    pub md_out: Option<PathBuf>,
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
    std::fs::read_to_string(path).map_or_else(|_| Vec::new(), |content| tweets_from_json(&content))
}

/// The primary debug-cache path: the same file Python's `--fetch-only`
/// writes, so either collector's output feeds the other's `--use-cache`
/// and the A/B harness can compare both sides' tweet sets.
fn debug_cache_path() -> Option<std::path::PathBuf> {
    dirs::home_dir().map(|h| h.join(".twitter_summary_debug_cache.json"))
}

/// Persist collected tweets for `--use-cache` runs and the A/B harness.
///
/// # Errors
///
/// When the JSON cannot be serialized (cannot happen for `Tweet`) or the
/// file cannot be written.
fn save_tweets_json(
    tweets: &[crate::ztools::twitter::Tweet],
    path: &std::path::Path,
) -> Result<()> {
    let body = serde_json::to_string_pretty(tweets)?;
    std::fs::write(path, body)?;
    Ok(())
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

pub(crate) fn twitter_summarize(config: &ZtoolsConfig, command: TwitterCommand) -> Result<()> {
    let opts = match command {
        TwitterCommand::Latest { last_updated } => {
            return crate::ztools::store::twitter_latest(last_updated);
        }
        TwitterCommand::Login => {
            println!("· Launching browser for x.com sign-in...");
            return crate::ztools::twitter::browser::login_live();
        }
        TwitterCommand::Clean => {
            // Housekeeping before a run, never the run: clear stored summaries,
            // report what happened, and exit successfully either way.
            let dir = crate::ztools::store::twitter_store_dir();
            let report = crate::ztools::store::clean_folder(&dir);
            for warning in &report.warnings {
                eprintln!("⚠ {warning}");
            }
            println!(
                "· Cleanup complete: {} .md file(s) removed from {}.",
                report.deleted,
                dir.display()
            );
            return Ok(());
        }
        TwitterCommand::Summarize(opts) => opts,
    };
    let TwitterSummarizeOpts {
        source,
        model,
        md_out,
    } = opts;

    let tweets = match source {
        // An explicit `--json` is a source the caller NAMED, so it is never
        // fallen back from, even when it yielded nothing: that would summarise
        // a different set of tweets than was asked for.
        TweetSource::Json(path_or_dash) if path_or_dash == "-" => {
            let mut buffer = String::new();
            if std::io::Read::read_to_string(&mut std::io::stdin(), &mut buffer).is_ok() {
                tweets_from_json(&buffer)
            } else {
                Vec::new()
            }
        }
        TweetSource::Json(path) => tweets_from_file(std::path::Path::new(&path)),
        TweetSource::Cache => {
            let candidates: Vec<PathBuf> = [
                dirs::home_dir().map(|h| h.join(".twitter_summary_debug_cache.json")),
                dirs::home_dir().map(|h| h.join(".cache/twitter/debug_tweets.json")),
            ]
            .into_iter()
            .flatten()
            .collect();
            let Some((from, cached)) =
                tweets_from_cache(&candidates).filter(|(_, t)| !t.is_empty())
            else {
                anyhow::bail!(
                    "No cached tweets found. Run without --use-cache first to scrape live tweets."
                );
            };
            println!(
                "· Using {} cached tweets from {}",
                cached.len(),
                from.display()
            );
            cached
        }
        TweetSource::Live {
            since,
            debug,
            fetch_only,
        } => {
            let tweets = crate::ztools::twitter::browser::collect_tweets_live(
                since.as_deref(),
                debug,
                config,
            )?;
            if tweets.is_empty() {
                println!("· No tweets found in the timeline window.");
                return Ok(());
            }
            if fetch_only {
                // Persist for `--use-cache` runs and the A/B harness, mirroring
                // Python's `save_debug_cache` to the same path. The message
                // used to claim this while writing nothing.
                if let Some(cache_path) = debug_cache_path() {
                    save_tweets_json(&tweets, &cache_path)?;
                }
                println!(
                    "✓ {} tweets fetched and cached. (--fetch-only provided, exiting)",
                    tweets.len()
                );
                return Ok(());
            }
            tweets
        }
    };

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
