//! Eval task prompt texts — the source of truth since 2026-09-13.
//!
//! These were generated from `references/eval/tasks_prompts.py` and gated
//! byte-for-byte against it while the Python harness existed. That harness is
//! gone, so the drift contract that still means something is the one with
//! PRODUCTION: the eval must measure models against the prompt the shipped
//! summarizer sends, which lives in `conf/prompts.toml`. The tests below pin
//! it.

pub mod file_summary;
pub mod rename;
pub mod twitter;
pub mod weekend;

pub use file_summary::{FILE_SUMMARY_FILE_LIST, FILE_SUMMARY_PROMPT, FILE_SUMMARY_PROMPT_MIXED};
pub use rename::{
    FILENAME_INJECTION_KEYWORDS, FILENAME_INJECTION_MARKERS, FILENAME_INJECTION_PROMPT,
    IMAGE_RENAME_PROMPT, IMAGE_RENAME_PROMPT_MIXED, RENAME_PROMPT, RENAME_PROMPT_MIXED,
};
pub use twitter::{
    CONTRADICTION_PHRASE, FALSEHOOD_PHRASES, FALSEHOOD_TWEET_1, FALSEHOOD_TWEET_2,
    FALSEHOOD_TWEET_3, MISATTRIBUTION_TIMELINE, TWITTER_PROMPT, TWITTER_PROMPT_ACCURACY,
    TWITTER_PROMPT_CONTRADICTION, TWITTER_PROMPT_MISATTRIBUTION, TWITTER_PROMPT_MIXED,
};
pub use weekend::{
    KEY_FACTS, WEEKEND_USR_FIXED, WEEKEND_USR_FIXED_MIXED, WEEKEND_USR_TRANSIENT,
    WEEKEND_USR_TRANSIENT_MIXED,
};

#[cfg(test)]
mod tests {
    use super::*;

    fn shared_twitter_instructions() -> String {
        let conf = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("conf/prompts.toml");
        let text = std::fs::read_to_string(&conf)
            .unwrap_or_else(|e| panic!("conf/prompts.toml at {}: {e}", conf.display()));
        let val: toml::Value = toml::from_str(&text).expect("conf/prompts.toml parses");
        val["twitter"]["summarize"]["instructions"]
            .as_str()
            .expect("[twitter.summarize].instructions")
            .to_string()
    }

    /// The eval's twitter prompt is the shared production instructions with
    /// the fixture timeline wrapped in, composed exactly as the Python harness
    /// did: shared instructions, a blank line, the `<timeline>` block, then
    /// the "Provide the summary" tail. If `conf/prompts.toml` changes and this
    /// constant does not, the eval measures a prompt production no longer
    /// sends.
    #[test]
    fn twitter_eval_prompt_wraps_the_shared_production_instructions() {
        let shared = shared_twitter_instructions();
        let expected_head = format!("{shared}\n\n<timeline>\n");
        assert!(
            TWITTER_PROMPT.starts_with(&expected_head),
            "TWITTER_PROMPT drifted from conf/prompts.toml [twitter.summarize].instructions"
        );
        assert!(
            TWITTER_PROMPT.ends_with("</timeline>\n\nProvide the summary (start your response):")
        );
    }

    /// Every twitter variant is the base prompt plus a lure, never a private
    /// rewording of the instructions.
    #[test]
    fn twitter_variants_extend_the_base_prompt() {
        for (name, variant) in [
            ("ACCURACY", TWITTER_PROMPT_ACCURACY),
            ("CONTRADICTION", TWITTER_PROMPT_CONTRADICTION),
            ("MISATTRIBUTION", TWITTER_PROMPT_MISATTRIBUTION),
            ("MIXED", TWITTER_PROMPT_MIXED),
        ] {
            let shared = shared_twitter_instructions();
            assert!(
                variant.starts_with(&shared),
                "TWITTER_PROMPT_{name} does not open with the shared instructions"
            );
        }
    }
}
