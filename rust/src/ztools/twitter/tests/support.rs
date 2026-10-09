//! The sandbox config and the loopback stub the `run_summary` tests share.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::thread;

use crate::config::ZtoolsConfig;

/// A `ZtoolsConfig` whose EVERY filesystem default points inside `root`.
///
/// Not decoration. `ZtoolsConfig::default()` reads a cache file, the Playwright
/// collector's checkout and that checkout's `conf/twitter.toml`, so a default
/// config in a unit test is three ways to touch the operator's disk — and the
/// `[fallback]` table it needs came from this checkout, so the tests passed
/// only while a developer had the right repo in the right place. Every path
/// field is replaced, not just the two the current call happens to reach, so
/// the next caller added here cannot pick up a `~` by accident.
pub(super) fn sandboxed_config(root: &Path) -> ZtoolsConfig {
    let conf = root.join("conf");
    std::fs::create_dir_all(&conf).unwrap();
    // The `[fallback]` table `chain::load_fallback_policy` refuses to run
    // without: a substitute is the whole point of the chain.
    std::fs::write(
        conf.join("twitter.toml"),
        "[fallback]\nmodels = [\"fixture-model\"]\npreferred = [\"fixture\"]\n",
    )
    .unwrap();
    let path = |p: PathBuf| p.to_string_lossy().into_owned();
    ZtoolsConfig {
        twitter_cache_path: path(root.join("cache/debug_tweets.json")),
        twitter_collector_dir: path(root.join("collector")),
        twitter_config_paths: vec![path(conf.join("twitter.toml"))],
        search_record_path: path(root.join("search_health.json")),
        weekend_exclusions_paths: vec![path(root.join("weekend.toml"))],
        weekend_region_paths: vec![path(root.join("weekend.toml"))],
        twitter_prompt_max_chars: 24_000,
        ..ZtoolsConfig::default()
    }
}

/// A one-shot loopback HTTP stub on `127.0.0.1`, answering every request with
/// `body` until dropped. Loopback only: no test may touch the real network.
pub(super) fn stub_server(body: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        for mut stream in listener.incoming().flatten() {
            let mut buf = vec![0u8; 4096];
            let _ = stream.read(&mut buf);
            let request = String::from_utf8_lossy(&buf);
            let reply = if request.contains("/v1/embeddings") {
                r#"{"data": [{"embedding": [0.1, 0.2]}]}"#.to_string()
            } else {
                body.to_string()
            };
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{reply}",
                reply.len()
            );
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
        }
    });
    format!("http://127.0.0.1:{port}")
}

/// The stub's chat answer for a summariser test: a body that already opens
/// with its own heading, so the writer's empty-`## Summary` guard is exercised.
pub(super) const CHAT_BODY: &str =
    r###"{"choices": [{"message": {"content": "## Section\n- Item 1\n- Item 2"}}]}"###;
