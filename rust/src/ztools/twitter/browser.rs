//! Camoufox / Browser driver for collecting Twitter timeline data.
//!
//! Port of `twitter/browser.py` and `twitter/browser_launch.py`.

use anyhow::Result;

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
}

impl BrowserCollector for LiveBrowserCollector {
    fn collect_timeline(&self, _target_count: usize) -> Result<Vec<Tweet>> {
        collect_tweets_live(self.since.as_deref(), self.debug)
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

/// Collect timeline tweets via the live headless browser driver.
pub fn collect_tweets_live(since: Option<&str>, debug: bool) -> Result<Vec<Tweet>> {
    let mut cmd = std::process::Command::new("python3");
    setup_python_env(&mut cmd);

    let mut py_stmt = "import sys; sys.argv = ['twitter', '--fetch-only'".to_string();
    if debug {
        py_stmt.push_str(", '--debug'");
    }
    if let Some(s) = since {
        py_stmt.push_str(&format!(", '--since', '{s}'"));
    }
    py_stmt.push_str("]; from twitter.cli import main; main()");

    cmd.args(["-c", &py_stmt]);
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

    for candidate in candidates.into_iter().flatten() {
        if candidate.exists() {
            if let Ok(content) = std::fs::read_to_string(&candidate) {
                if let Ok(tweets) = serde_json::from_str::<Vec<Tweet>>(&content) {
                    if !tweets.is_empty() {
                        return Ok(tweets);
                    }
                }
            }
        }
    }

    anyhow::bail!("No tweets collected from browser session.")
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
}
