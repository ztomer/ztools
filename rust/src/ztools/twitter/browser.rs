//! Browser collector seam for the Twitter timeline.
//!
//! Port of `twitter/browser.py` and `twitter/browser_launch.py`. The live
//! driver is the native `camoufox-rs` collector in [`native`]; the Python
//! subprocess it replaced was retired 2026-09-13 once the Phase 3 A/B passed
//! on the calibrated criterion (shared-record agreement — see
//! `bin/ab_test::compare_collect_records` for why ID-set equality was the
//! wrong instrument).

use anyhow::Result;

use super::cookies::Cookie;
use super::native;
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
    /// # Errors
    ///
    /// When the timeline cannot be collected: the browser session failed to
    /// start, was not logged in, or produced no tweets at all. An implementation
    /// returning FEWER than `target_count` tweets is not an error -- the
    /// timeline simply ended.
    fn collect_timeline(&self, target_count: usize) -> Result<Vec<Tweet>>;
}

/// Live browser collector driving the Camoufox / Playwright timeline scraper.
pub struct LiveBrowserCollector {
    pub since: Option<String>,
    pub debug: bool,
    pub config: crate::config::ZtoolsConfig,
    /// Injection seam so the passthrough logic is testable without launching
    /// a browser; production always wires [`collect_tweets_live`].
    runner: fn(Option<&str>, bool, &crate::config::ZtoolsConfig) -> Result<Vec<Tweet>>,
}

impl LiveBrowserCollector {
    #[must_use]
    pub fn new(since: Option<String>, debug: bool, config: crate::config::ZtoolsConfig) -> Self {
        // `collect_tweets_live` is the single choke point every caller uses;
        // the runner stays fixed here so tests keep a stable seam.
        Self {
            since,
            debug,
            config,
            runner: collect_tweets_live,
        }
    }
}

impl BrowserCollector for LiveBrowserCollector {
    fn collect_timeline(&self, _target_count: usize) -> Result<Vec<Tweet>> {
        (self.runner)(self.since.as_deref(), self.debug, &self.config)
    }
}

/// Run browser login to authenticate and store the persistent profile.
///
/// # Errors
///
/// When the headed browser cannot be started, or the user closes the window
/// without completing sign-in.
pub fn login_live() -> Result<()> {
    native::login_native()
}

/// Collect timeline tweets via the live native browser driver.
///
/// # Errors
///
/// When the browser cannot be started, the session is not signed in, the
/// captured traffic never came from the Following endpoint, or no tweets
/// were collected. An empty timeline is an error here rather than an empty
/// list: every caller is about to summarise what it gets back, and
/// summarising nothing produces a confident summary of no data.
pub fn collect_tweets_live(
    since: Option<&str>,
    debug: bool,
    config: &crate::config::ZtoolsConfig,
) -> Result<Vec<Tweet>> {
    native::collect_tweets_native(since, debug, config)
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
                    id: String::new(),
                    screen_name: "user1".to_string(),
                    text: "Tweet 1".to_string(),
                    created_at: "Thu Aug 20 12:00:00 +0000 2026".to_string(),
                    favorite_count: 10,
                    retweet_count: 2,
                    reply_to: None,
                },
                Tweet {
                    id: String::new(),
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
        let collector = LiveBrowserCollector::new(
            Some("2026-08-01".to_string()),
            true,
            crate::config::ZtoolsConfig::default(),
        );
        assert_eq!(collector.since.as_deref(), Some("2026-08-01"));
        assert!(collector.debug);
    }

    #[test]
    fn test_live_collector_passes_since_and_debug_to_runner() {
        static CAPTURED: std::sync::Mutex<Vec<(Option<String>, bool)>> =
            std::sync::Mutex::new(Vec::new());
        // A non-capturing closure coerces to the runner's fn pointer; it
        // records its arguments and answers an empty timeline.
        let collector = LiveBrowserCollector {
            since: Some("2026-08-15".to_string()),
            debug: true,
            config: crate::config::ZtoolsConfig::default(),
            runner: |since, debug, _config| {
                CAPTURED
                    .lock()
                    .unwrap()
                    .push((since.map(str::to_string), debug));
                Ok(Vec::new())
            },
        };

        let tweets = collector.collect_timeline(5).unwrap();

        assert!(tweets.is_empty());
        assert_eq!(
            *CAPTURED.lock().unwrap(),
            vec![(Some("2026-08-15".to_string()), true)]
        );
    }
}
