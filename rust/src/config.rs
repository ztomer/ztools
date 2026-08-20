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
    /// Vision model for naming images with no readable text. Empty means the
    /// VLM path is unavailable (the Python CLI requires an explicit
    /// `--vlm-model` too), and such images fall back to a clean of the stem.
    #[serde(default = "default_image_renamer_vlm_model")]
    pub image_renamer_vlm_model: String,
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
    /// Instruction block the twitter summarizer wraps its timeline into. The
    /// canonical text lives in `conf/prompts.toml`; this embedded copy is the
    /// fallback a static binary uses with no checkout, and the drift-gate test
    /// below keeps the two equal, so the runtime text is the same either way.
    #[serde(default = "default_twitter_summarize_prompt")]
    pub twitter_summarize_prompt: String,
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
fn default_image_renamer_vlm_model() -> String {
    String::new()
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

/// Embedded fallback of `conf/prompts.toml` `[twitter.summarize].instructions`.
/// Kept byte-identical to that file by `test_twitter_prompt_matches_shared_conf`.
const TWITTER_SUMMARIZE_PROMPT: &str = r#"You are an objective news distillation system. Your task is to extract hard
facts from the provided chronological Twitter/X timeline.

<instructions>
1. First, analyze the timeline in block.
2. Start with an overall ## Executive Summary section capturing the main narrative.
3. Organize into topic sections using ## headers and bullet points.
4. Use connecting phrases ('following up on', 'subsequently announced') and narrative verbs
   ('released', 'responded', 'criticized') to show how events relate.
5. CRITICAL: End EVERY bullet with the author handle and timestamp copied EXACTLY as
   they appear in that tweet's source line. A source line beginning
   `[@TechCrunch | 08:00]:` yields a bullet ending `(@TechCrunch | 08:00)`.
   Never invent or reformat a date, weekday or time that is not in the source line.
</instructions>

<formatting_rules>
- Start with a `## Executive Summary` paragraph
- Use topic headers starting with `##`
- Use bullet points for facts
- Use narrative verbs and connecting phrases showing event relationships
- End every bullet with `(@handle | timestamp-exactly-as-written-in-the-source-line)`
</formatting_rules>"#;

pub(crate) fn default_twitter_summarize_prompt() -> String {
    TWITTER_SUMMARIZE_PROMPT.to_string()
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
            image_renamer_vlm_model: default_image_renamer_vlm_model(),
            llm_timeout_secs: default_llm_timeout_secs(),
            llm_extended_timeout_secs: default_llm_extended_timeout_secs(),
            llm_quick_timeout_secs: default_llm_quick_timeout_secs(),
            twitter_prompt_max_chars: default_twitter_prompt_max_chars(),
            max_image_filename_len: default_max_image_filename_len(),
            twitter_summarize_prompt: default_twitter_summarize_prompt(),
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

    /// Layer shared prompt texts from `conf/prompts.toml` over the embedded
    /// fallbacks. The drift-gate test keeps the fallbacks byte-equal to that
    /// file, so a run behaves identically whether the file is present or not —
    /// the static binary still works standalone, and a checkout still edits
    /// prompts in exactly one place.
    pub fn with_shared_prompts(mut self) -> Self {
        let candidates = [
            dirs::home_dir().map(|h| h.join(".config/ztools/prompts.toml")),
            dirs::home_dir().map(|h| h.join("Projects/ztools/conf/prompts.toml")),
        ];
        for cand in candidates.into_iter().flatten() {
            if cand.is_file() {
                if let Ok(content) = std::fs::read_to_string(&cand) {
                    if let Ok(val) = toml::from_str::<toml::Value>(&content) {
                        if let Some(p) = val
                            .get("twitter")
                            .and_then(|t| t.get("summarize"))
                            .and_then(|s| s.get("instructions"))
                            .and_then(|v| v.as_str())
                        {
                            self.twitter_summarize_prompt = p.to_string();
                        }
                        break;
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

    /// The drift gate for the shared prompt surface: `conf/prompts.toml` is the
    /// canonical home of the twitter summarize prompt, and this embedded copy is
    /// the fallback a static binary runs with when no checkout is present. If
    /// they ever diverge, the two sides answer different prompts — exactly the
    /// parallel-copy drift this phase exists to kill — so the test fails loudly
    /// and tells the author to update both.
    #[test]
    fn test_twitter_prompt_matches_shared_conf() {
        use std::path::Path;
        let manifest = env!("CARGO_MANIFEST_DIR");
        let conf_path = Path::new(manifest).parent().unwrap().join("conf/prompts.toml");
        let content = std::fs::read_to_string(&conf_path)
            .unwrap_or_else(|e| panic!("conf/prompts.toml missing at {}: {e}", conf_path.display()));
        let val: toml::Value = toml::from_str(&content).expect("conf/prompts.toml must parse");
        let shared = val
            .get("twitter")
            .and_then(|t| t.get("summarize"))
            .and_then(|s| s.get("instructions"))
            .and_then(|v| v.as_str())
            .expect("conf/prompts.toml needs [twitter.summarize].instructions");
        assert_eq!(
            default_twitter_summarize_prompt(),
            shared,
            "embedded twitter summarize prompt drifted from conf/prompts.toml — \
             the file is canonical; update the embedded fallback in config.rs to match"
        );
    }
}
