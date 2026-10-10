//! No test process ever sees the operator's real home.
//!
//! Test runs left fixtures in the operator's real store more than once --
//! `@cached_user` and `@ai_researcher` summaries in
//! `~/Documents/twitter_summaries`, `{"screen_name":"u"}` in
//! `~/.twitter_summary_debug_cache.json` -- each time because one test
//! resolved `~` without `TestEnv`. The audit catches the spellings it knows;
//! this removes the cause: `.cargo/config.toml` points `HOME` at
//! `rust/target/test-home` for every process cargo starts, so a test that
//! forgets the guard writes into the build directory, never into `~`.

use std::process::Command;

/// The operator's home from the password database, which `HOME` cannot move.
fn real_home() -> String {
    let out = Command::new("sh")
        .args(["-c", "eval echo ~$(id -un)"])
        .output()
        .expect("sh runs");
    String::from_utf8(out.stdout)
        .expect("utf-8")
        .trim()
        .to_string()
}

#[test]
fn a_test_process_home_is_the_sandbox_not_the_operators() {
    let home = std::env::var("HOME").expect("HOME is set");
    let real = real_home();
    assert_ne!(real.len(), 0, "could not read the real home");
    assert_ne!(home, real, "a test process sees the operator's real home");
    assert!(
        home.ends_with("target/test-home"),
        "HOME is {home}, expected the cargo [env] sandbox"
    );
    // A spawned binary inherits it too: what an integration test runs.
    let child = Command::new("sh")
        .args(["-c", "echo $HOME"])
        .output()
        .expect("sh runs");
    assert_eq!(String::from_utf8_lossy(&child.stdout).trim(), home);
}
