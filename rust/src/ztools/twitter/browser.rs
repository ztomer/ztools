//! Camoufox / Browser driver for collecting Twitter timeline data.
//!
//! Port of `twitter/browser.py` and `twitter/browser_launch.py`.

use anyhow::Result;

use std::path::PathBuf;

use super::cookies::Cookie;
use super::Tweet;

#[derive(Debug, Clone)]
pub struct CamoufoxConfig {
    pub headless: bool,
    pub cookies: Vec<Cookie>,
    pub timeout_secs: u64,
}

impl Default for CamoufoxConfig {
    fn default() -> Self {
        Self {
            headless: true,
            cookies: Vec::new(),
            timeout_secs: 30,
        }
    }
}

pub trait BrowserCollector: Send + Sync {
    fn collect_timeline(&self, target_count: usize) -> Result<Vec<Tweet>>;
}

/// Live browser collector driving the Camoufox / Playwright timeline scraper.
pub struct LiveBrowserCollector {
    pub since: Option<String>,
    pub debug: bool,
    /// Injection seam so the passthrough logic is testable without spawning
    /// Python; production always wires [`collect_tweets_live`].
    runner: fn(Option<&str>, bool) -> Result<Vec<Tweet>>,
}

impl LiveBrowserCollector {
    pub fn new(since: Option<String>, debug: bool) -> Self {
        Self {
            since,
            debug,
            runner: collect_tweets_live,
        }
    }
}

impl BrowserCollector for LiveBrowserCollector {
    fn collect_timeline(&self, _target_count: usize) -> Result<Vec<Tweet>> {
        (self.runner)(self.since.as_deref(), self.debug)
    }
}

/// Run browser login to authenticate and store cookies.
pub fn login_live() -> Result<()> {
    let mut cmd = std::process::Command::new("python3");
    setup_python_env(&mut cmd);
    cmd.args([
        "-c",
        "import sys; sys.argv = ['twitter', '--login']; from twitter.cli import main; main()",
    ]);
    let status = cmd.status()?;
    if !status.success() {
        anyhow::bail!("Browser login exited with code {:?}", status.code());
    }
    Ok(())
}

/// Helper to configure PYTHONPATH to reach shipped modules in references/ or environment.
fn setup_python_env(cmd: &mut std::process::Command) {
    let mut paths = Vec::new();
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let p = std::path::Path::new(&manifest_dir)
            .parent()
            .map(|p| p.join("references"));
        if let Some(p) = p {
            if p.exists() {
                paths.push(p.display().to_string());
            }
        }
    }
    if let Some(home) = dirs::home_dir() {
        let p = home.join("Projects/ztools/references");
        if p.exists() {
            paths.push(p.display().to_string());
        }
    }
    if !paths.is_empty() {
        let combined = paths.join(":");
        if let Ok(existing) = std::env::var("PYTHONPATH") {
            cmd.env("PYTHONPATH", format!("{combined}:{existing}"));
        } else {
            cmd.env("PYTHONPATH", combined);
        }
    }
}

/// Build the inline Python statement that drives the fetch-only CLI.
fn build_fetch_stmt(debug: bool, since: Option<&str>) -> String {
    let mut py_stmt = "import sys; sys.argv = ['twitter', '--fetch-only'".to_string();
    if debug {
        py_stmt.push_str(", '--debug'");
    }
    if let Some(s) = since {
        py_stmt.push_str(&format!(", '--since', '{s}'"));
    }
    py_stmt.push_str("]; from twitter.cli import main; main()");
    py_stmt
}

/// Return the first cache file that parses to a non-empty tweet list.
///
/// Narrow seam over the post-collection cache scan so the fallback ordering
/// (first hit wins; empty/malformed/missing entries are skipped) is testable
/// against fixture files instead of a live browser session.
fn tweets_from_cache_candidates(candidates: Vec<PathBuf>) -> Option<Vec<Tweet>> {
    for candidate in candidates {
        if candidate.exists() {
            if let Ok(content) = std::fs::read_to_string(&candidate) {
                if let Ok(tweets) = serde_json::from_str::<Vec<Tweet>>(&content) {
                    if !tweets.is_empty() {
                        return Some(tweets);
                    }
                }
            }
        }
    }
    None
}

/// Collect timeline tweets via the live headless browser driver.
pub fn collect_tweets_live(since: Option<&str>, debug: bool) -> Result<Vec<Tweet>> {
    let mut cmd = std::process::Command::new("python3");
    setup_python_env(&mut cmd);
    cmd.args(["-c", &build_fetch_stmt(debug, since)]);
    let status = cmd
        .status()
        .map_err(|e| anyhow::anyhow!("failed to execute browser scraper: {e}"))?;

    if !status.success() {
        anyhow::bail!(
            "Browser collection failed with exit code: {:?}",
            status.code()
        );
    }

    // Check standard cache locations
    let candidates = [
        dirs::home_dir().map(|h| h.join(".twitter_summary_debug_cache.json")),
        dirs::home_dir().map(|h| h.join(".cache/twitter/debug_tweets.json")),
    ];

    match tweets_from_cache_candidates(candidates.into_iter().flatten().collect()) {
        Some(tweets) => Ok(tweets),
        None => anyhow::bail!("No tweets collected from browser session."),
    }
}

/// Mock browser collector for offline deterministic testing without launching GUI/headless browsers.
pub struct MockBrowserCollector {
    pub canned_tweets: Vec<Tweet>,
}

impl BrowserCollector for MockBrowserCollector {
    fn collect_timeline(&self, target_count: usize) -> Result<Vec<Tweet>> {
        let count = self.canned_tweets.len().min(target_count);
        Ok(self.canned_tweets[..count].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_collector_respects_target_count() {
        let collector = MockBrowserCollector {
            canned_tweets: vec![
                Tweet {
                    screen_name: "user1".to_string(),
                    text: "Tweet 1".to_string(),
                    created_at: "Thu Aug 20 12:00:00 +0000 2026".to_string(),
                    favorite_count: 10,
                    retweet_count: 2,
                    reply_to: None,
                },
                Tweet {
                    screen_name: "user2".to_string(),
                    text: "Tweet 2".to_string(),
                    created_at: "Thu Aug 20 12:01:00 +0000 2026".to_string(),
                    favorite_count: 20,
                    retweet_count: 5,
                    reply_to: None,
                },
            ],
        };

        let result = collector.collect_timeline(1).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].screen_name, "user1");

        let all = collector.collect_timeline(10).unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_default_config_is_headless_without_cookies() {
        let config = CamoufoxConfig::default();
        assert!(config.headless);
        assert!(config.cookies.is_empty());
        assert_eq!(config.timeout_secs, 30);
    }

    #[test]
    fn test_new_collector_keeps_construction_args() {
        let collector = LiveBrowserCollector::new(Some("2026-08-01".to_string()), true);
        assert_eq!(collector.since.as_deref(), Some("2026-08-01"));
        assert!(collector.debug);
    }

    #[test]
    fn test_build_fetch_stmt_flags_and_since_quoting() {
        assert_eq!(
            build_fetch_stmt(false, None),
            "import sys; sys.argv = ['twitter', '--fetch-only']; from twitter.cli import main; main()"
        );
        // `since` is interpolated inside single quotes in the argv list.
        assert_eq!(
            build_fetch_stmt(false, Some("2026-08-01")),
            "import sys; sys.argv = ['twitter', '--fetch-only', '--since', '2026-08-01']; from twitter.cli import main; main()"
        );
        assert_eq!(
            build_fetch_stmt(true, None),
            "import sys; sys.argv = ['twitter', '--fetch-only', '--debug']; from twitter.cli import main; main()"
        );
    }

    #[test]
    fn test_live_collector_passes_since_and_debug_to_runner() {
        static CAPTURED: std::sync::Mutex<Vec<(Option<String>, bool)>> =
            std::sync::Mutex::new(Vec::new());
        fn capturing_runner(since: Option<&str>, debug: bool) -> Result<Vec<Tweet>> {
            CAPTURED
                .lock()
                .unwrap()
                .push((since.map(str::to_string), debug));
            Ok(Vec::new())
        }
        let collector = LiveBrowserCollector {
            since: Some("2026-08-15".to_string()),
            debug: true,
            runner: capturing_runner,
        };

        let tweets = collector.collect_timeline(5).unwrap();

        assert!(tweets.is_empty());
        assert_eq!(
            *CAPTURED.lock().unwrap(),
            vec![(Some("2026-08-15".to_string()), true)]
        );
    }

    #[test]
    fn test_cache_scan_first_non_empty_wins_and_bad_entries_are_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let empty_cache = dir.path().join("empty.json");
        let bad_cache = dir.path().join("bad.json");
        let good_cache = dir.path().join("good.json");
        std::fs::write(&empty_cache, "[]").unwrap();
        std::fs::write(&bad_cache, "{not json").unwrap();
        std::fs::write(
            &good_cache,
            r#"[{"screen_name":"user1","text":"hi","created_at":"Thu Aug 20 12:00:00 +0000 2026","favorite_count":1,"retweet_count":0}]"#,
        )
        .unwrap();

        let tweets =
            tweets_from_cache_candidates(vec![empty_cache.clone(), bad_cache, good_cache.clone()])
                .expect("good cache must win");

        assert_eq!(tweets.len(), 1);
        assert_eq!(tweets[0].screen_name, "user1");

        // Ordering matters: a non-empty earlier file preempts a later one.
        let first = tweets_from_cache_candidates(vec![good_cache, empty_cache]).unwrap();
        assert_eq!(first[0].screen_name, "user1");
    }

    #[test]
    fn test_cache_scan_none_when_every_candidate_is_missing_or_bad() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("nope.json");
        let bad = dir.path().join("bad.json");
        std::fs::write(&bad, "[1, 2, 3]").unwrap(); // valid JSON, wrong shape

        assert!(tweets_from_cache_candidates(vec![]).is_none());
        assert!(tweets_from_cache_candidates(vec![missing, bad]).is_none());
    }

    #[test]
    fn test_setup_python_env_puts_references_dirs_on_pythonpath() {
        let mut cmd = std::process::Command::new("true");
        setup_python_env(&mut cmd);

        let pythonpath = cmd
            .get_envs()
            .find(|(k, _)| k.to_string_lossy() == "PYTHONPATH")
            .and_then(|(_, v)| v)
            .expect("cargo test always provides CARGO_MANIFEST_DIR with references/")
            .to_string_lossy()
            .to_string();
        let manifest_refs = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("references");
        if manifest_refs.exists() {
            assert!(
                pythonpath.starts_with(manifest_refs.display().to_string().as_str()),
                "PYTHONPATH '{pythonpath}' must lead with {manifest_refs:?}"
            );
        }
    }
}
