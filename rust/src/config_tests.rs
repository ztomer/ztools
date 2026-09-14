use super::*;

#[test]
fn test_default_config_values() {
    let cfg = ZtoolsConfig::default();
    assert_eq!(cfg.twitter_model, "gemma-4-e2b-it-8bit");
    assert_eq!(cfg.weekend_model, "qwen3.8-27b-jang_6d");
    assert_eq!(cfg.image_renamer_model, "gemma-4-e2b-it-8bit");
    assert_eq!(cfg.image_renamer_vlm_model, "qwen3.8-27b-8bit");
    assert_eq!(cfg.think_model, "ornith-1.0-35b-jang_4m");
    assert_eq!(cfg.llm_timeout_secs, 120);
}

#[test]
fn test_with_ztools_best_models_preserves_on_missing() {
    let cfg = ZtoolsConfig::default().with_ztools_best_models();
    assert!(!cfg.twitter_model.is_empty());
    assert!(!cfg.weekend_model.is_empty());
    assert!(!cfg.image_renamer_model.is_empty());
    assert!(!cfg.image_renamer_vlm_model.is_empty());
    assert!(!cfg.think_model.is_empty());
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
    let conf_path = Path::new(manifest)
        .parent()
        .unwrap()
        .join("conf/prompts.toml");
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

// MARK: - Layering shared prompts over the embedded fallbacks
//
// Every branch here used to be unreachable from a test, because the
// candidate paths were anchored to `$HOME`. They are the branches that
// decide whether a run uses the operator's prompt or the compiled-in one,
// which is the difference between two runs that look identical and are not.

fn prompt_file(dir: &std::path::Path, name: &str, body: &str) -> std::path::PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, body).unwrap();
    path
}

#[test]
fn a_shared_prompt_file_overrides_the_embedded_fallback() {
    let tmp = tempfile::tempdir().unwrap();
    let file = prompt_file(
        tmp.path(),
        "prompts.toml",
        "[twitter.summarize]\ninstructions = \"summarize like a telegram\"\n",
    );
    let cfg = ZtoolsConfig::default().with_shared_prompts_from(&[file]);
    assert_eq!(cfg.twitter_summarize_prompt, "summarize like a telegram");
}

#[test]
fn no_candidate_file_leaves_the_embedded_fallback_alone() {
    let tmp = tempfile::tempdir().unwrap();
    let embedded = ZtoolsConfig::default().twitter_summarize_prompt;
    let cfg = ZtoolsConfig::default().with_shared_prompts_from(&[tmp.path().join("absent.toml")]);
    assert_eq!(
        cfg.twitter_summarize_prompt, embedded,
        "a missing file is the normal standalone-binary case, not an error"
    );
}

#[test]
fn an_unparseable_prompt_file_leaves_the_fallback_alone() {
    let tmp = tempfile::tempdir().unwrap();
    let embedded = ZtoolsConfig::default().twitter_summarize_prompt;
    let file = prompt_file(tmp.path(), "prompts.toml", "this is not [[[ toml");
    let cfg = ZtoolsConfig::default().with_shared_prompts_from(&[file]);
    assert_eq!(
        cfg.twitter_summarize_prompt, embedded,
        "a broken file must not blank the prompt -- an empty instruction \
         would change what the model is asked without any error"
    );
}

#[test]
fn a_file_without_the_key_leaves_the_fallback_alone() {
    let tmp = tempfile::tempdir().unwrap();
    let embedded = ZtoolsConfig::default().twitter_summarize_prompt;
    let file = prompt_file(tmp.path(), "prompts.toml", "[weekend]\nsomething = 1\n");
    let cfg = ZtoolsConfig::default().with_shared_prompts_from(&[file]);
    assert_eq!(cfg.twitter_summarize_prompt, embedded);
}

/// The first readable, parseable file wins and the search stops -- even
/// when it does not carry the key. Falling through would silently prefer a
/// stale second copy over an intentionally minimal first one.
#[test]
fn the_first_parseable_candidate_wins_and_stops_the_search() {
    let tmp = tempfile::tempdir().unwrap();
    let first = prompt_file(
        tmp.path(),
        "first.toml",
        "[twitter.summarize]\ninstructions = \"first wins\"\n",
    );
    let second = prompt_file(
        tmp.path(),
        "second.toml",
        "[twitter.summarize]\ninstructions = \"second must not\"\n",
    );
    let cfg = ZtoolsConfig::default().with_shared_prompts_from(&[first, second.clone()]);
    assert_eq!(cfg.twitter_summarize_prompt, "first wins");

    // And a first candidate that is simply absent is skipped, not fatal.
    let cfg =
        ZtoolsConfig::default().with_shared_prompts_from(&[tmp.path().join("absent.toml"), second]);
    assert_eq!(cfg.twitter_summarize_prompt, "second must not");
}

/// A directory at a candidate path is not a file, and must be skipped
/// rather than read.
#[test]
fn a_directory_at_a_candidate_path_is_skipped() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("prompts.toml");
    std::fs::create_dir(&dir).unwrap();
    let good = prompt_file(
        tmp.path(),
        "real.toml",
        "[twitter.summarize]\ninstructions = \"from the real file\"\n",
    );
    let cfg = ZtoolsConfig::default().with_shared_prompts_from(&[dir, good]);
    assert_eq!(cfg.twitter_summarize_prompt, "from the real file");
}
