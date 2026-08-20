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
