//! Every test here constructs a [`ZtoolsConfig`], and that type's defaults name
//! `~/…` paths and four third-party hosts. Taking the shared sandbox
//! ([`crate::test_env::TestEnv`]) is what makes the isolation a property of the
//! test rather than a coincidence of code order: nothing here resolves a `~`
//! against the operator's home, and `dirs::home_dir()` reads cannot race another
//! sandboxed test mid-redirect.
use serial_test::serial;

use super::*;
use crate::test_env::TestEnv;

/// The ONE place this file builds a [`ZtoolsConfig`], and the sandbox it is
/// built under.
///
/// Both halves are load-bearing, which is why they are the same call.
/// `ZtoolsConfig::default()` names `~/…` for the twitter cache, the collector,
/// the weekend data files and the eval inputs, and it resolves them through
/// `dirs::home_dir()` — process-global — so a test that reads one while a peer
/// test has `$HOME` pointed elsewhere measures the PEER's home. Folding the
/// guard into the constructor makes "build a config here" and "hold the
/// sandbox" one statement, which is the only way the next test added below
/// cannot get one without the other.
///
/// It returns the defaults UNMODIFIED, on purpose: this file is about what the
/// defaults are, so a helper that rewrote the path fields would quietly stop
/// testing them and start testing the helper.
fn sandboxed() -> (TestEnv, ZtoolsConfig) {
    (TestEnv::new(), ZtoolsConfig::default())
}

/// The slot names themselves are pinned against `conf/config.toml` by
/// `embedded_slot_defaults_match_conf_best_models` below -- a literal here
/// pinned three uninstalled models for a month, which is a golden test
/// encoding a wrong value (house rule #8).
#[test]
#[serial]
fn test_default_config_values() {
    let (env, cfg) = sandboxed();
    assert_eq!(cfg.llm_timeout_secs, 120);
    assert_eq!(cfg.llm_stall_secs, 120);
    assert_eq!(cfg.llm_max_tokens, 4096);
    assert_eq!(cfg.llm_warmup_timeout_secs, 900);
    drop(env);
}

#[test]
#[serial]
fn test_with_ztools_best_models_preserves_on_missing() {
    let (env, base) = sandboxed();
    let cfg = base.with_ztools_best_models();
    assert_nonempty!(cfg.twitter_model);
    assert_nonempty!(cfg.weekend_model);
    assert_nonempty!(cfg.image_renamer_model);
    assert_nonempty!(cfg.image_renamer_vlm_model);
    assert_nonempty!(cfg.think_model);
    drop(env);
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
#[serial]
fn a_shared_prompt_file_overrides_the_embedded_fallback() {
    let (env, base) = sandboxed();
    let tmp = tempfile::tempdir().unwrap();
    let file = prompt_file(
        tmp.path(),
        "prompts.toml",
        "[twitter.summarize]\ninstructions = \"summarize like a telegram\"\n",
    );
    let cfg = base.with_shared_prompts_from(&[file]);
    assert_eq!(cfg.twitter_summarize_prompt, "summarize like a telegram");
    drop(env);
}

#[test]
#[serial]
fn no_candidate_file_leaves_the_embedded_fallback_alone() {
    let (env, base) = sandboxed();
    let tmp = tempfile::tempdir().unwrap();
    let embedded = base.twitter_summarize_prompt.clone();
    let cfg = base.with_shared_prompts_from(&[tmp.path().join("absent.toml")]);
    assert_eq!(
        cfg.twitter_summarize_prompt, embedded,
        "a missing file is the normal standalone-binary case, not an error"
    );
    drop(env);
}

#[test]
#[serial]
fn an_unparseable_prompt_file_leaves_the_fallback_alone() {
    let (env, base) = sandboxed();
    let tmp = tempfile::tempdir().unwrap();
    let embedded = base.twitter_summarize_prompt.clone();
    let file = prompt_file(tmp.path(), "prompts.toml", "this is not [[[ toml");
    let cfg = base.with_shared_prompts_from(&[file]);
    assert_eq!(
        cfg.twitter_summarize_prompt, embedded,
        "a broken file must not blank the prompt -- an empty instruction \
         would change what the model is asked without any error"
    );
    drop(env);
}

#[test]
#[serial]
fn a_file_without_the_key_leaves_the_fallback_alone() {
    let (env, base) = sandboxed();
    let tmp = tempfile::tempdir().unwrap();
    let embedded = base.twitter_summarize_prompt.clone();
    let file = prompt_file(tmp.path(), "prompts.toml", "[weekend]\nsomething = 1\n");
    let cfg = base.with_shared_prompts_from(&[file]);
    assert_eq!(cfg.twitter_summarize_prompt, embedded);
    drop(env);
}

/// The first readable, parseable file wins and the search stops -- even
/// when it does not carry the key. Falling through would silently prefer a
/// stale second copy over an intentionally minimal first one.
#[test]
#[serial]
fn the_first_parseable_candidate_wins_and_stops_the_search() {
    let (env, base) = sandboxed();
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
    let cfg = base
        .clone()
        .with_shared_prompts_from(&[first, second.clone()]);
    assert_eq!(cfg.twitter_summarize_prompt, "first wins");

    // And a first candidate that is simply absent is skipped, not fatal.
    let cfg = base.with_shared_prompts_from(&[tmp.path().join("absent.toml"), second]);
    assert_eq!(cfg.twitter_summarize_prompt, "second must not");
    drop(env);
}

/// A directory at a candidate path is not a file, and must be skipped
/// rather than read.
#[test]
#[serial]
fn a_directory_at_a_candidate_path_is_skipped() {
    let (env, base) = sandboxed();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("prompts.toml");
    std::fs::create_dir(&dir).unwrap();
    let good = prompt_file(
        tmp.path(),
        "real.toml",
        "[twitter.summarize]\ninstructions = \"from the real file\"\n",
    );
    let cfg = base.with_shared_prompts_from(&[dir, good]);
    assert_eq!(cfg.twitter_summarize_prompt, "from the real file");
    drop(env);
}

/// The drift gate for the model slots: the embedded defaults a static binary
/// falls back on must equal `conf/config.toml [best_models]`, the derived
/// source of truth. Before this gate the defaults named three models that
/// were not installed for a month, and a checkout-less run would have fallen
/// through its chain on every call.
#[test]
#[serial]
fn embedded_slot_defaults_match_conf_best_models() {
    let (env, d) = sandboxed();
    let conf_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("conf/config.toml");
    let content = std::fs::read_to_string(&conf_path)
        .unwrap_or_else(|e| panic!("conf/config.toml missing at {}: {e}", conf_path.display()));
    let val: toml::Value = toml::from_str(&content).expect("conf/config.toml must parse");
    let best = &val["best_models"];
    let slot = |k: &str| {
        best[k]
            .as_str()
            .unwrap_or_else(|| panic!("[best_models].{k} missing"))
    };
    assert_eq!(d.twitter_model, slot("summarize"));
    assert_eq!(d.weekend_model, slot("json"));
    assert_eq!(d.image_renamer_model, slot("filename"));
    assert_eq!(d.image_renamer_vlm_model, slot("vlm"));
    assert_eq!(d.think_model, slot("think"));
    drop(env);
}

// MARK: - The forecast endpoint
//
// `weather_url` used to be a literal inside `open_meteo_url`, so there was
// nothing to configure and every run that fetched a forecast went to
// `api.open-meteo.com`. These two tests are the whole of the config-path claim:
// the default is UNCHANGED (so nobody's run silently moved), and a config file
// is honoured (so the endpoint can be moved).

/// The behaviour-preservation claim, as a literal rather than against
/// [`default_weather_url`]: the host that used to be compiled in must be
/// exactly what an operator with no config file still reaches. Comparing the
/// default to its own constructor would keep passing if the constructor were
/// edited — which is the one edit nobody would notice, because a moved origin
/// still returns a plausible forecast.
#[test]
#[serial]
fn the_default_weather_endpoint_is_the_host_that_was_hardcoded() {
    let (env, cfg) = sandboxed();
    assert_eq!(
        cfg.weather_url, "https://api.open-meteo.com",
        "an operator with no config file must reach the same forecast origin \
         this endpoint replaced; changing the default changes every run"
    );
    drop(env);
}

/// The override path, through the SAME parse `--config` performs.
///
/// A field with a serde default is only overridable if the file is actually
/// read for it, and the failure that hides is the quiet kind: an override
/// nothing consumes leaves the default in place, which looks exactly like a
/// working override until you watch the request go somewhere else. The second
/// assertion is the other half — one endpoint being overridden must not blank
/// the three beside it, which is what a missing `#[serde(default)]` would do.
#[test]
#[serial]
fn a_config_file_overrides_the_weather_endpoint_and_leaves_the_others_alone() {
    let (env, defaults) = sandboxed();
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("ztools.toml");
    std::fs::write(&path, "weather_url = \"http://127.0.0.1:9\"\n").unwrap();

    let cfg: ZtoolsConfig = toml::from_str(&std::fs::read_to_string(&path).unwrap())
        .expect("a one-key config file is a valid ZtoolsConfig");
    assert_eq!(
        cfg.weather_url, "http://127.0.0.1:9",
        "the endpoint must come from the file, not survive as the default"
    );
    for (slot, got, want) in [
        ("osaurus_url", &cfg.osaurus_url, &defaults.osaurus_url),
        (
            "duckduckgo_url",
            &cfg.duckduckgo_url,
            &defaults.duckduckgo_url,
        ),
        ("bing_url", &cfg.bing_url, &defaults.bing_url),
        ("brave_url", &cfg.brave_url, &defaults.brave_url),
    ] {
        assert_eq!(got, want, "{slot} must keep its default");
    }
    drop(env);
}

// MARK: - Where the shipped data is looked for
//
// Five defaults used to name the historical home checkout literally, so the
// installed binary found its config only on the machine whose home held one at
// that exact path. They are derived now (`manifest::checkout_roots_from`), and
// the ORDER those candidates are read in is a product decision rather than an
// implementation detail: swap two entries and a different file answers, with no
// error anywhere. So each default's list is pinned as exact equality below, and
// the fixture is a checkout beside a FAKE executable plus a checkout at the
// historical path — two distinct roots, or the test could not tell which one won.

/// The historical candidate, spelled ONCE so the five pins below cannot disagree
/// about it — the same argument as `weekend_exclusions_paths` and
/// `weekend_region_paths` being built by one helper.
///
/// It is a literal on purpose. These assertions are the guard on the *derivation*,
/// so deriving the expected value from `manifest` would be the code compared with
/// itself (see `docs/TESTING.md`, "An assertion whose expected value is the
/// fallback cannot see which branch ran"). The gate's `path-ok:` marker is what
/// lets a literal live in a test at all.
/// path-ok: the pre-derivation candidate these tests pin, not a resolved location.
const HISTORICAL: &str = "~/Projects/ztools";

/// The seams the precedence pins below are written against: each is
/// `manifest::config_paths_from` with the SAME overlay and relative path the
/// production default passes, spelled out because the production default cannot
/// be reached at all with a chosen `current_exe`. One way for the two to drift
/// remains — an overlay edited in `config.rs` only — and it is closed by
/// [`the_defaults_still_name_the_home_checkout_when_there_is_no_checkout_above_the_binary`],
/// which pins the LIVE defaults' literals.
fn twitter_config_paths(exe: &std::path::Path, home: &std::path::Path) -> Vec<String> {
    crate::manifest::config_paths_from(
        Some(exe),
        Some(home),
        &["~/.config/ztools/twitter.toml"],
        "conf/twitter.toml",
    )
}
fn eval_conf_dirs(exe: &std::path::Path, home: &std::path::Path) -> Vec<String> {
    crate::manifest::config_paths_from(Some(exe), Some(home), &["~/.config/ztools"], "conf")
}
fn eval_tasks_dirs(exe: &std::path::Path, home: &std::path::Path) -> Vec<String> {
    crate::manifest::config_paths_from(Some(exe), Some(home), &[], "eval_tasks/data")
}
fn weekend_toml_paths(exe: &std::path::Path, home: &std::path::Path) -> Vec<String> {
    crate::manifest::config_paths_from(
        Some(exe),
        Some(home),
        &["~/.config/weekend.toml"],
        "conf/weekend.toml",
    )
}
fn twitter_collector_dir(exe: &std::path::Path, home: &std::path::Path) -> String {
    crate::manifest::first_checkout_root_from(Some(exe), Some(home))
}

/// A fake executable inside a checkout, and a home holding a second one.
///
/// Returned as `(exe, home, installed, historical)` because every assertion
/// below is about the RELATIONSHIP between them, and spelling that relationship
/// out per test is how four copies of the same fixture drifted once already.
fn two_checkouts() -> (
    tempfile::TempDir,
    std::path::PathBuf,
    std::path::PathBuf,
    std::path::PathBuf,
    std::path::PathBuf,
) {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let historical = home.join("Projects/ztools");
    let installed = tmp.path().join("opt/ztools");
    // The two markers `is_checkout_root` accepts, so a root is really found.
    for root in [&historical, &installed] {
        std::fs::create_dir_all(root.join("conf")).unwrap();
        std::fs::create_dir_all(root.join("rust")).unwrap();
        std::fs::write(root.join("conf/config.toml"), "[best_models]\n").unwrap();
        std::fs::write(root.join("rust/Cargo.toml"), "[package]\n").unwrap();
    }
    let exe = installed.join("bin/ztools");
    std::fs::create_dir_all(exe.parent().unwrap()).unwrap();
    (tmp, exe, home, installed, historical)
}

/// The rule every one of these lists shares: the operator's own overlay first,
/// then the checkout the binary was installed from, then the historical home
/// path. One test for the shape, then one per default for its OWN file, because
/// a shared rule and a shared list are two different things and a regression
/// could move either.
#[test]
#[serial]
fn every_shipped_path_default_reads_the_installed_checkout_before_the_home_one() {
    let (_tmp, exe, home, installed, _historical) = two_checkouts();
    let exe = exe.as_path();
    let home = home.as_path();
    let under = |root: &std::path::Path, rel: &str| format!("{}/{rel}", root.display());

    for (name, got, overlay, rel) in [
        (
            "twitter_config_paths",
            twitter_config_paths(exe, home),
            "~/.config/ztools/twitter.toml",
            "conf/twitter.toml",
        ),
        (
            "eval_conf_dirs",
            eval_conf_dirs(exe, home),
            "~/.config/ztools",
            "conf",
        ),
        (
            "eval_tasks_dirs",
            eval_tasks_dirs(exe, home),
            "",
            "eval_tasks/data",
        ),
        (
            "weekend_exclusions_paths",
            weekend_toml_paths(exe, home),
            "~/.config/weekend.toml",
            "conf/weekend.toml",
        ),
        (
            "weekend_region_paths",
            weekend_toml_paths(exe, home),
            "~/.config/weekend.toml",
            "conf/weekend.toml",
        ),
    ] {
        let mut want: Vec<String> = if overlay.is_empty() {
            Vec::new()
        } else {
            vec![overlay.to_string()]
        };
        want.push(under(&installed, rel));
        want.push(format!("{HISTORICAL}/{rel}"));
        assert_eq!(got, want, "{name}");
    }

    assert_eq!(
        twitter_collector_dir(exe, home),
        installed.to_string_lossy().into_owned(),
        "the single-directory default is the FIRST root, so the Playwright \\
         collector is the checkout this binary came from"
    );
}

/// The list as it comes out of `ZtoolsConfig::default()` on THIS machine, which
/// is what an operator reads in their config file. Asserted rather than left
/// implicit because "the derived candidate is silently absent when the binary
/// runs from the build dir" is a behaviour with two possible readings — working
/// as intended, or broken — and only the literal settles it.
#[test]
#[serial]
fn the_defaults_still_name_the_home_checkout_when_there_is_no_checkout_above_the_binary() {
    let (_env, cfg) = sandboxed();
    // The sandbox puts no checkout above the test binary, so this is the
    // fallback-only shape: exactly what an operator on this machine had before
    // the derivation, which is what makes "keep every candidate working" true.
    assert_eq!(
        cfg.eval_conf_dirs,
        vec!["~/.config/ztools".to_string(), format!("{HISTORICAL}/conf")]
    );
    assert_eq!(
        cfg.twitter_config_paths,
        vec![
            "~/.config/ztools/twitter.toml".to_string(),
            format!("{HISTORICAL}/conf/twitter.toml")
        ]
    );
    assert_eq!(
        cfg.eval_tasks_dirs,
        vec![format!("{HISTORICAL}/eval_tasks/data")]
    );
    assert_eq!(
        cfg.weekend_exclusions_paths, cfg.weekend_region_paths,
        "the two weekend lists are built by ONE helper, so they cannot drift"
    );
    assert_eq!(
        cfg.twitter_collector_dir, HISTORICAL,
        "the collector directory is a real path, not an empty string"
    );
    // Every one of them resolves under the sandbox home, so no default can name
    // the operator's disk even when the fallback is the only candidate.
    for slot in [
        &cfg.twitter_collector_dir,
        &cfg.eval_tasks_dirs[0],
        &cfg.eval_conf_dirs[1],
    ] {
        let expanded = crate::manifest::expand_tilde(slot);
        assert!(
            expanded.starts_with(std::env::var_os("HOME").unwrap()),
            "{slot:?} resolved to {expanded:?}, outside the sandbox home"
        );
    }
}
