//! The off-machine tier end to end: the Osaurus server is down and
//! token-pool's server answers (a fake speaking MCP over stdio).

use super::super::*;
use super::support::sandboxed_config;
use crate::test_env::TestEnv;
use serial_test::serial;

/// A port nothing listens on: the server this Mac usually has, down.
const DOWN: &str = "http://127.0.0.1:59999";

fn tweet() -> Tweet {
    Tweet {
        id: String::new(),
        screen_name: "routine_user".to_string(),
        text: "the server is down".to_string(),
        created_at: "14:00".to_string(),
        favorite_count: 1,
        retweet_count: 0,
        reply_to: None,
    }
}

fn config_with(env: &TestEnv, fallback: &str) -> crate::config::ZtoolsConfig {
    let cfg = sandboxed_config(&env.root().join("cfg"));
    std::fs::write(&cfg.twitter_config_paths[0], fallback).unwrap();
    cfg
}

#[test]
#[serial]
fn the_pool_answers_when_the_server_is_down() {
    let env = TestEnv::new();
    let reply = serde_json::json!({"status": "unverified", "tier": "opencode/fake-free",
        "text": "## Section\n- Item 1 (@routine_user | 14:00)"});
    let cmd = crate::ztools::pool::tests::fake_command(env.root(), &reply);
    let cmd_toml = serde_json::to_string(&cmd).unwrap();
    let cfg = config_with(
        &env,
        &format!("[fallback]\nmodels = [\"pool\"]\npreferred = []\npool = {cmd_toml}\n"),
    );
    let output = tempfile::tempdir().unwrap();
    let path = run_summary(&[tweet()], output.path(), Some(DOWN), Some("m1"), &cfg)
        .expect("the pool answered, so a summary is written");
    let content = std::fs::read_to_string(path).unwrap();
    assert!(content.contains("pool (token-pool, fallback)"), "{content}");
    assert!(content.contains("DEGRADED"), "{content}");
    assert!(content.contains("## Section"), "{content}");
    drop(env);
}

#[test]
#[serial]
fn a_pool_named_without_its_command_is_refused_not_sent_to_the_server() {
    let env = TestEnv::new();
    let cfg = config_with(&env, "[fallback]\nmodels = [\"pool\"]\npreferred = []\n");
    let output = tempfile::tempdir().unwrap();
    let why = format!(
        "{:#}",
        run_summary(&[tweet()], output.path(), Some(DOWN), Some("m1"), &cfg).unwrap_err()
    );
    assert!(why.contains("[fallback] pool is unset"), "{why}");
    drop(env);
}
