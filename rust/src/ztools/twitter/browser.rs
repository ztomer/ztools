//! Browser collector seam for the Twitter timeline.
//!
//! Port of `twitter/browser.py` and `twitter/browser_launch.py`. The live
//! driver is the native `camoufox-rs` collector in [`native`]; the Python
//! subprocess it replaced was retired 2026-09-13 once the Phase 3 A/B passed
//! on the calibrated criterion (shared-record agreement — see
//! `bin/ab_test::compare_collect_records` for why ID-set equality was the
//! wrong instrument).

use anyhow::Result;

use super::Tweet;
use super::cookies::Cookie;
use super::native;

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

    use crate::test_env::TestEnv;

    /// The collector's runner field, spelled out once so a test can coerce a
    /// fn ITEM to the same type before comparing: `fn_addr_eq` takes
    /// `FnPtr`, which fn items do not implement.
    type Runner = fn(Option<&str>, bool, &crate::config::ZtoolsConfig) -> Result<Vec<Tweet>>;

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
        assert_empty!(config.cookies);
        assert_eq!(config.timeout_secs, 30);
    }

    /// `LiveBrowserCollector` holds a whole `ZtoolsConfig`, and that config's
    /// defaults name `~/…` for the cache, the collector checkout and the
    /// weekend data — resolved through `dirs::home_dir()`, which the shared
    /// sandbox redirects. The guard and the config are therefore built TOGETHER
    /// in the two tests below: a collector whose config resolved its paths
    /// against the operator's home is a collector that could read their cache.
    #[test]
    fn test_new_collector_keeps_construction_args() {
        let env = TestEnv::new();
        let collector = LiveBrowserCollector::new(
            Some("2026-08-01".to_string()),
            true,
            crate::config::ZtoolsConfig::default(),
        );
        assert_eq!(collector.since.as_deref(), Some("2026-08-01"));
        assert!(collector.debug);
        drop(env);
    }

    /// The wiring the whole module exists for, pinned. `new()` is one line and
    /// it was never asserted: swapping its runner for a stub keeps every other
    /// test green, and the collector then reports an empty timeline forever
    /// while looking perfectly healthy. `fn_addr_eq` rather than `==` because
    /// plain fn-pointer comparison warns under `-D warnings` (addresses are
    /// not guaranteed unique) -- and the answer is still exact here, because
    /// both sides name the same item.
    #[test]
    fn new_wires_the_production_runner_rather_than_a_stub() {
        let env = TestEnv::new();
        let collector =
            LiveBrowserCollector::new(None, false, crate::config::ZtoolsConfig::default());
        assert!(
            std::ptr::fn_addr_eq(collector.runner, collect_tweets_live as Runner),
            "LiveBrowserCollector::new must hold collect_tweets_live: a collector whose runner is \
             a stub never reaches the browser driver, and nothing else in the suite would say so"
        );
        drop(env);
    }

    const TRIPWIRE_SINCE: &str = "2026-08-15";
    const TRIPWIRE_MODEL: &str = "tripwire-model";

    /// A runner that REFUSES to answer unless it is handed the collector's own
    /// `since`, `debug` and `config`, and that answers with a timeline no
    /// other runner in this file produces.
    ///
    /// The previous test recorded its arguments into a static and then compared
    /// that static against the values the test itself had put into the
    /// collector: a self-echo, true whether or not `collect_timeline` ever
    /// called it. Answering instead of recording makes the assertion about
    /// `collect_timeline`'s RETURN VALUE, which no amount of dropping or
    /// replacing the runner leaves intact.
    fn tripwire_runner(
        since: Option<&str>,
        debug: bool,
        config: &crate::config::ZtoolsConfig,
    ) -> Result<Vec<Tweet>> {
        if since != Some(TRIPWIRE_SINCE) || !debug || config.twitter_model != TRIPWIRE_MODEL {
            anyhow::bail!(
                "runner was not handed the collector's own arguments: since={since:?} \
                 debug={debug} model={}",
                config.twitter_model
            );
        }
        Ok(vec![Tweet {
            id: "tripwire-1".to_string(),
            screen_name: "tripwire_user".to_string(),
            text: "answered by the tripwire runner".to_string(),
            created_at: "Thu Aug 20 12:00:00 +0000 2026".to_string(),
            favorite_count: 7,
            retweet_count: 1,
            reply_to: None,
        }])
    }

    /// The collector, AND the sandbox its config was built under.
    ///
    /// Returned together rather than as two statements because this is the only
    /// place in the file that builds a config for a caller: a caller that took
    /// the guard itself could forget, and the config's `~/…` defaults would
    /// resolve against whichever home was current. Holding the guard here also
    /// means the sandbox outlives every assertion that reads the collector.
    fn tripwire_collector() -> (LiveBrowserCollector, TestEnv) {
        let env = TestEnv::new();
        let collector = LiveBrowserCollector {
            since: Some(TRIPWIRE_SINCE.to_string()),
            debug: true,
            config: crate::config::ZtoolsConfig {
                twitter_model: TRIPWIRE_MODEL.to_string(),
                ..crate::config::ZtoolsConfig::default()
            },
            runner: tripwire_runner,
        };
        (collector, env)
    }

    #[test]
    fn collect_timeline_answers_with_exactly_what_the_runner_returned() {
        let (collector, env) = tripwire_collector();
        let tweets = collector.collect_timeline(5).unwrap();

        // Wrong arguments reach the runner as an Err (the unwrap above names
        // them); a runner that is never called yields nothing at all. Only the
        // one path that keeps both properties gets here.
        assert_eq!(tweets.len(), 1);
        assert_eq!(tweets[0].id, "tripwire-1");
        assert_eq!(tweets[0].screen_name, "tripwire_user");
        assert_eq!(tweets[0].text, "answered by the tripwire runner");
        assert_eq!(tweets[0].favorite_count, 7);
        drop(env);
    }

    /// A runner that fails must fail the collection. Returning an empty
    /// timeline instead would be the one unrecoverable answer: every caller
    /// goes on to summarise what it gets back, and an empty list summarises
    /// to a confident account of no data.
    #[test]
    fn a_failed_runner_reaches_the_caller_instead_of_an_empty_timeline() {
        let (tripwire, env) = tripwire_collector();
        let collector = LiveBrowserCollector {
            runner: |_, _, _| anyhow::bail!("browser session did not start"),
            ..tripwire
        };

        let err = collector
            .collect_timeline(5)
            .expect_err("a dead browser must not read as a quiet timeline");
        assert!(
            err.to_string().contains("browser session did not start"),
            "{err}"
        );
        drop(env);
    }
}

/// The saved twitter summary document's byte-level golden. `#[path]` because it
/// lives in `twitter/` next to the code it pins, while `twitter/mod.rs` is owned
/// elsewhere: wiring it from a module this file owns keeps the new file from
/// needing an edit to that one.
#[cfg(test)]
#[path = "summary_doc_tests.rs"]
mod summary_doc_tests;
