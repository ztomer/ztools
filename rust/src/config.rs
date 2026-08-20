//! The `[ztools]` config block: model choices, timeouts and the paths and
//! endpoints the ztools subsystems talk to. Split from `config.rs` for the
//! house 400-line cap.
//!
//! Every external location the subsystems touch is a knob here rather than a
//! hardcoded `~/…` or third-party host, so a test can point them at fixtures
//! and a run does not depend on which machine it is on.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZtoolsConfig {
    #[serde(default = "default_osaurus_url")]
    pub osaurus_url: String,
    /// Where the weekend planner's web search goes. Configurable for the same
    /// reason `osaurus_url` is: a hardcoded third-party host cannot be pointed
    /// at a stub, which leaves the planner's whole fetch path untestable and
    /// makes every run depend on someone else's uptime.
    #[serde(default = "default_duckduckgo_url")]
    pub duckduckgo_url: String,
    /// Where the weekend planner looks for its exclusion list, in order; the
    /// first file that yields entries wins, and an empty list means "use the
    /// built-in defaults". Configurable because a hardcoded `~/…` path makes
    /// the loader read whichever machine happens to be running the tests.
    #[serde(default = "default_weekend_exclusions_paths")]
    pub weekend_exclusions_paths: Vec<String>,
    /// Where the summarizer looks for a previously captured timeline when it is
    /// handed no tweets. Configurable so a test can point it at a fixture: it
    /// used to be a hardcoded `~/.cache/…` path, and the test that exercised it
    /// wrote into the developer's real cache to do so.
    #[serde(default = "default_twitter_cache_path")]
    pub twitter_cache_path: String,
    /// Project directory holding the Playwright collector the summarizer falls
    /// back to when it has no tweets and no cache. Configurable so a test can
    /// point it somewhere harmless — it used to be a hardcoded `~/Projects/…`,
    /// which meant a unit test could launch the operator's real browser
    /// scraper.
    #[serde(default = "default_twitter_collector_dir")]
    pub twitter_collector_dir: String,
    #[serde(default = "default_twitter_model")]
    pub twitter_model: String,
    #[serde(default = "default_weekend_model")]
    pub weekend_model: String,
    #[serde(default = "default_image_renamer_model")]
    pub image_renamer_model: String,
    #[serde(default = "default_llm_timeout_secs")]
    pub llm_timeout_secs: u64,
    #[serde(default = "default_llm_extended_timeout_secs")]
    pub llm_extended_timeout_secs: u64,
    #[serde(default = "default_llm_quick_timeout_secs")]
    pub llm_quick_timeout_secs: u64,
    #[serde(default = "default_twitter_prompt_max_chars")]
    pub twitter_prompt_max_chars: usize,
    #[serde(default = "default_max_image_filename_len")]
    pub max_image_filename_len: usize,
}

fn default_osaurus_url() -> String {
    "http://localhost:1337".to_string()
}
fn default_duckduckgo_url() -> String {
    "https://html.duckduckgo.com/html/".to_string()
}
fn default_twitter_cache_path() -> String {
    "~/.cache/twitter/debug_tweets.json".to_string()
}
fn default_twitter_collector_dir() -> String {
    "~/Projects/ztools".to_string()
}
fn default_weekend_exclusions_paths() -> Vec<String> {
    vec![
        "~/.config/weekend.toml".to_string(),
        "~/Projects/ztools/conf/weekend.toml".to_string(),
    ]
}
fn default_twitter_model() -> String {
    "gemma-4-e2b-it-8bit".to_string()
}
fn default_weekend_model() -> String {
    "qwen3.8-27b-8bit".to_string()
}
fn default_image_renamer_model() -> String {
    "gemma-4-e2b-it-8bit".to_string()
}
fn default_llm_timeout_secs() -> u64 {
    120
}
fn default_llm_extended_timeout_secs() -> u64 {
    300
}
fn default_llm_quick_timeout_secs() -> u64 {
    10
}
fn default_twitter_prompt_max_chars() -> usize {
    24000
}
fn default_max_image_filename_len() -> usize {
    50
}

impl Default for ZtoolsConfig {
    fn default() -> Self {
        Self {
            osaurus_url: default_osaurus_url(),
            duckduckgo_url: default_duckduckgo_url(),
            weekend_exclusions_paths: default_weekend_exclusions_paths(),
            twitter_cache_path: default_twitter_cache_path(),
            twitter_collector_dir: default_twitter_collector_dir(),
            twitter_model: default_twitter_model(),
            weekend_model: default_weekend_model(),
            image_renamer_model: default_image_renamer_model(),
            llm_timeout_secs: default_llm_timeout_secs(),
            llm_extended_timeout_secs: default_llm_extended_timeout_secs(),
            llm_quick_timeout_secs: default_llm_quick_timeout_secs(),
            twitter_prompt_max_chars: default_twitter_prompt_max_chars(),
            max_image_filename_len: default_max_image_filename_len(),
        }
    }
}

impl ZtoolsConfig {
    /// Attempt to load dynamic `[best_models]` from ztools config if present.
    pub fn with_ztools_best_models(mut self) -> Self {
        let candidates = [
            dirs::home_dir().map(|h| h.join(".config/ztools/config.toml")),
            dirs::home_dir().map(|h| h.join("Projects/ztools/conf/config.toml")),
        ];
        for cand in candidates.into_iter().flatten() {
            if cand.is_file() {
                if let Ok(content) = std::fs::read_to_string(&cand) {
                    if let Ok(toml_val) = toml::from_str::<toml::Value>(&content) {
                        if let Some(best) = toml_val.get("best_models") {
                            if let Some(m) = best.get("summarize").and_then(|v| v.as_str()) {
                                self.twitter_model = m.to_string();
                            }
                            if let Some(m) = best.get("json").and_then(|v| v.as_str()) {
                                self.weekend_model = m.to_string();
                            }
                            if let Some(m) = best.get("filename").and_then(|v| v.as_str()) {
                                self.image_renamer_model = m.to_string();
                            }
                            break;
                        }
                    }
                }
            }
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_values() {
        let cfg = ZtoolsConfig::default();
        assert_eq!(cfg.twitter_model, "gemma-4-e2b-it-8bit");
        assert_eq!(cfg.weekend_model, "qwen3.8-27b-8bit");
        assert_eq!(cfg.image_renamer_model, "gemma-4-e2b-it-8bit");
        assert_eq!(cfg.llm_timeout_secs, 120);
    }

    #[test]
    fn test_with_ztools_best_models_preserves_on_missing() {
        let cfg = ZtoolsConfig::default().with_ztools_best_models();
        assert!(!cfg.twitter_model.is_empty());
        assert!(!cfg.weekend_model.is_empty());
        assert!(!cfg.image_renamer_model.is_empty());
    }
}
