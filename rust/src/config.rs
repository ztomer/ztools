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
    /// The second search engine, consulted only when `DuckDuckGo` walls or
    /// empties a query. Configurable for the same reason as the first.
    #[serde(default = "default_bing_url")]
    pub bing_url: String,
    /// The third engine, consulted only when the first two both fail a query.
    #[serde(default = "default_brave_url")]
    pub brave_url: String,
    /// Where the weekend planner looks for its exclusion list, in order; the
    /// first file that yields entries wins, and an empty list means "use the
    /// built-in defaults". Configurable because a hardcoded `~/…` path makes
    /// the loader read whichever machine happens to be running the tests.
    #[serde(default = "default_weekend_exclusions_paths")]
    pub weekend_exclusions_paths: Vec<String>,
    /// Where the weekend planner looks for its region-evidence lists, in
    /// order: the `[region]` table (`in_region`, `foreign`) of `weekend.toml`.
    /// Same candidates as the exclusions paths — one shared helper builds
    /// both defaults so they cannot drift apart.
    #[serde(default = "default_weekend_region_paths")]
    pub weekend_region_paths: Vec<String>,
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
    /// Where the twitter collector looks for `twitter.toml`, in order; the
    /// `[endpoints]` table (timeline URL markers + the Following marker) is
    /// read from the first file that carries it. Same two-spot shape as the
    /// weekend paths: a user overlay, then the shipped checkout file.
    #[serde(default = "default_twitter_config_paths")]
    pub twitter_config_paths: Vec<String>,
    /// Where the eval looks for its data files (`eval_inputs.toml`,
    /// `eval_vision.toml`), in order: a user overlay, then the shipped
    /// checkout `conf/`. The first directory that exists wins.
    #[serde(default = "default_eval_conf_dirs")]
    pub eval_conf_dirs: Vec<String>,
    /// Where the eval looks for task snapshots (`taxes/*.json`) when
    /// `--tasks-dir` is not given; the first directory that exists wins.
    #[serde(default = "default_eval_tasks_dirs")]
    pub eval_tasks_dirs: Vec<String>,
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
    /// Structured reasoning / fallback model (from [`best_models`].think).
    #[serde(default = "default_think_model")]
    pub think_model: String,
    #[serde(default = "default_llm_timeout_secs")]
    pub llm_timeout_secs: u64,
    #[serde(default = "default_llm_extended_timeout_secs")]
    pub llm_extended_timeout_secs: u64,
    #[serde(default = "default_llm_quick_timeout_secs")]
    pub llm_quick_timeout_secs: u64,
    /// How long ONE warm-up request may wait for a cold model to load before
    /// the pipeline declares the model unavailable. Separate from the call
    /// timeouts above, because those measure GENERATION and this measures
    /// LOADING: a 25GB model measured an 8m38s cold start on 2026-09-19, and
    /// a client that gives up at 300s makes the server cancel the load, so
    /// every subsequent call restarted it and none ever finished.
    #[serde(default = "default_llm_warmup_timeout_secs")]
    pub llm_warmup_timeout_secs: u64,
    /// No token for this long means the server is not generating. The one
    /// timeout that decides a call has failed; the per-call figures above
    /// are CAPS on a call whose tokens keep flowing (see `ztools::llm`).
    #[serde(default = "default_llm_stall_secs")]
    pub llm_stall_secs: u64,
    /// The bound every production answer is generated under. The weekend
    /// JSON for ten events is ~1,500 tokens and a timeline summary ~3,000;
    /// this is room, not a target. Server-side, so a runaway is cut by the
    /// server rather than abandoned by the client (which wedges it).
    #[serde(default = "default_llm_max_tokens")]
    pub llm_max_tokens: u32,
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
fn default_bing_url() -> String {
    "https://www.bing.com/search".to_string()
}
fn default_brave_url() -> String {
    "https://search.brave.com/search".to_string()
}
fn default_twitter_cache_path() -> String {
    "~/.cache/twitter/debug_tweets.json".to_string()
}
fn default_twitter_collector_dir() -> String {
    "~/Projects/ztools".to_string()
}
fn default_twitter_config_paths() -> Vec<String> {
    vec![
        "~/.config/ztools/twitter.toml".to_string(),
        "~/Projects/ztools/conf/twitter.toml".to_string(),
    ]
}
fn default_eval_conf_dirs() -> Vec<String> {
    vec![
        "~/.config/ztools".to_string(),
        "~/Projects/ztools/conf".to_string(),
    ]
}
fn default_eval_tasks_dirs() -> Vec<String> {
    vec!["~/Projects/ztools/eval_tasks/data".to_string()]
}
fn default_weekend_toml_paths() -> Vec<String> {
    vec![
        "~/.config/weekend.toml".to_string(),
        "~/Projects/ztools/conf/weekend.toml".to_string(),
    ]
}
fn default_weekend_exclusions_paths() -> Vec<String> {
    default_weekend_toml_paths()
}
fn default_weekend_region_paths() -> Vec<String> {
    default_weekend_toml_paths()
}
// The embedded slot defaults are what a static binary uses with no checkout.
// `config_tests::embedded_slot_defaults_match_conf_best_models` keeps them
// equal to `conf/config.toml [best_models]`, the derived source of truth --
// they named three uninstalled models for a month before that gate existed.
fn default_twitter_model() -> String {
    "gemma-4-e4b-it-8bit".to_string()
}
fn default_weekend_model() -> String {
    "qwen3.8-27b-jang_6d".to_string()
}
fn default_image_renamer_model() -> String {
    "qwen3.8-27b-jang_6d".to_string()
}
fn default_image_renamer_vlm_model() -> String {
    "qwen3.8-27b-jang_6d".to_string()
}
fn default_think_model() -> String {
    "qwen3.8-27b-jang_6d".to_string()
}
const fn default_llm_timeout_secs() -> u64 {
    120
}
const fn default_llm_extended_timeout_secs() -> u64 {
    // A CAP on a streaming call, not a wait: reached only when tokens keep
    // arriving for this long. 4,096 tokens at the slowest measured decode
    // (14 tok/s) is ~290s, so 300 would cut a full-length answer that was
    // going fine. The stall guard (`llm_stall_secs`) is what catches a dead
    // server.
    900
}
const fn default_llm_quick_timeout_secs() -> u64 {
    10
}
const fn default_llm_warmup_timeout_secs() -> u64 {
    900
}
const fn default_llm_stall_secs() -> u64 {
    120
}
const fn default_llm_max_tokens() -> u32 {
    4096
}
const fn default_twitter_prompt_max_chars() -> usize {
    24000
}
const fn default_max_image_filename_len() -> usize {
    50
}

/// Embedded fallback of `conf/prompts.toml` `[twitter.summarize].instructions`.
/// Kept byte-identical to that file by `test_twitter_prompt_matches_shared_conf`.
const TWITTER_SUMMARIZE_PROMPT: &str = r"You are an objective news distillation system. Your task is to extract hard
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
</formatting_rules>";

pub(crate) fn default_twitter_summarize_prompt() -> String {
    TWITTER_SUMMARIZE_PROMPT.to_string()
}

impl ZtoolsConfig {
    /// The streaming budget for one production call capped at `cap_secs`.
    #[must_use]
    pub const fn chat_budget(&self, cap_secs: u64) -> crate::ztools::llm::ChatBudget {
        crate::ztools::llm::ChatBudget {
            stall_secs: self.llm_stall_secs,
            cap_secs,
            max_tokens: self.llm_max_tokens,
        }
    }
}

impl Default for ZtoolsConfig {
    fn default() -> Self {
        Self {
            osaurus_url: default_osaurus_url(),
            duckduckgo_url: default_duckduckgo_url(),
            bing_url: default_bing_url(),
            brave_url: default_brave_url(),
            weekend_exclusions_paths: default_weekend_exclusions_paths(),
            weekend_region_paths: default_weekend_region_paths(),
            twitter_cache_path: default_twitter_cache_path(),
            twitter_collector_dir: default_twitter_collector_dir(),
            twitter_config_paths: default_twitter_config_paths(),
            eval_conf_dirs: default_eval_conf_dirs(),
            eval_tasks_dirs: default_eval_tasks_dirs(),
            twitter_model: default_twitter_model(),
            weekend_model: default_weekend_model(),
            image_renamer_model: default_image_renamer_model(),
            image_renamer_vlm_model: default_image_renamer_vlm_model(),
            think_model: default_think_model(),
            llm_timeout_secs: default_llm_timeout_secs(),
            llm_extended_timeout_secs: default_llm_extended_timeout_secs(),
            llm_quick_timeout_secs: default_llm_quick_timeout_secs(),
            llm_warmup_timeout_secs: default_llm_warmup_timeout_secs(),
            llm_stall_secs: default_llm_stall_secs(),
            llm_max_tokens: default_llm_max_tokens(),
            twitter_prompt_max_chars: default_twitter_prompt_max_chars(),
            max_image_filename_len: default_max_image_filename_len(),
            twitter_summarize_prompt: default_twitter_summarize_prompt(),
        }
    }
}

impl ZtoolsConfig {
    /// The first existing directory in `dirs` (tilde-expanded), if any.
    fn first_existing_dir(dirs: &[String]) -> Option<std::path::PathBuf> {
        dirs.iter()
            .map(|d| crate::manifest::expand_tilde(d))
            .find(|p| p.is_dir())
    }

    /// One eval data file, from the first `eval_conf_dirs` entry that holds
    /// it — resolved per FILE, so a user overlay dir that carries only some
    /// of them does not hide the shipped copies of the rest.
    ///
    /// # Errors
    ///
    /// When no candidate holds the file: the roster refuses to guess its
    /// inputs, so the absence is reported with the paths that were tried.
    pub fn eval_data_file(&self, name: &str) -> anyhow::Result<std::path::PathBuf> {
        self.eval_conf_dirs
            .iter()
            .map(|d| crate::manifest::expand_tilde(d).join(name))
            .find(|p| p.is_file())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no {name} found under any eval conf dir (tried {}); set eval_conf_dirs in \
                     the ztools config",
                    self.eval_conf_dirs.join(", ")
                )
            })
    }

    /// The roster's data files, each resolved through [`Self::eval_data_file`].
    ///
    /// # Errors
    ///
    /// As [`Self::eval_data_file`].
    pub fn eval_roster_inputs(&self) -> anyhow::Result<crate::ztools::eval::tasks::RosterInputs> {
        Ok(crate::ztools::eval::tasks::RosterInputs {
            inputs: self.eval_data_file("eval_inputs.toml")?,
            vision: self.eval_data_file("eval_vision.toml")?,
        })
    }

    /// The default task-snapshot directory (see `eval_tasks_dirs`), if any exists.
    #[must_use]
    pub fn eval_tasks_dir(&self) -> Option<std::path::PathBuf> {
        Self::first_existing_dir(&self.eval_tasks_dirs)
    }

    /// Attempt to load dynamic `[best_models]` from ztools config if present.
    #[must_use]
    pub fn with_ztools_best_models(mut self) -> Self {
        let candidates = [
            dirs::home_dir().map(|h| h.join(".config/ztools/config.toml")),
            dirs::home_dir().map(|h| h.join("Projects/ztools/conf/config.toml")),
        ];
        for cand in candidates.into_iter().flatten() {
            if cand.is_file() {
                if let Ok(content) = std::fs::read_to_string(cand) {
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
                            if let Some(m) = best.get("vlm").and_then(|v| v.as_str()) {
                                self.image_renamer_vlm_model = m.to_string();
                            }
                            if let Some(m) = best.get("think").and_then(|v| v.as_str()) {
                                self.think_model = m.to_string();
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
    #[must_use]
    pub fn with_shared_prompts(self) -> Self {
        let candidates: Vec<std::path::PathBuf> = [
            dirs::home_dir().map(|h| h.join(".config/ztools/prompts.toml")),
            dirs::home_dir().map(|h| h.join("Projects/ztools/conf/prompts.toml")),
        ]
        .into_iter()
        .flatten()
        .collect();
        self.with_shared_prompts_from(&candidates)
    }

    /// The seam. `with_shared_prompts` anchors its candidates to `$HOME`,
    /// which makes every branch below — file absent, unreadable, malformed,
    /// present but missing the key — untestable without writing into the
    /// developer's own home directory. Taking the list as an argument costs
    /// one line and makes all four provable.
    ///
    /// First readable, parseable candidate wins and the search STOPS, even if
    /// it does not carry the key: a file that exists is the operator's answer,
    /// and falling through to the next one would silently prefer a stale copy
    /// over an intentionally minimal one.
    #[must_use]
    pub fn with_shared_prompts_from(mut self, candidates: &[std::path::PathBuf]) -> Self {
        for cand in candidates {
            if cand.is_file() {
                if let Ok(content) = std::fs::read_to_string(cand) {
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
#[path = "config_tests.rs"]
mod tests;
