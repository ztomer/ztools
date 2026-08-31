//! Pure-Rust Camoufox browser collector for the Weekend Planner.
//!
//! Spawns and drives Camoufox over native Unix pipe transport via the
//! vendored `camoufox` crate. Zero Python dependencies.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use camoufox::api::{Browser, BrowserOptions, ContextOptions};
use camoufox::config::LaunchConfig;
use camoufox::process;
use camoufox::protocol::client::Connection;
use camoufox::transport::pipe::PipeTransport;

const PAGE_LOAD_TIMEOUT: Duration = Duration::from_secs(15);
const EVALUATE_TIMEOUT: Duration = Duration::from_secs(5);

/// Locate the Camoufox browser executable on the system.
pub fn find_camoufox_binary() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("CAMOUFOX_BIN") {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    for candidate in [
        "/opt/homebrew/bin/camoufox",
        "/usr/local/bin/camoufox",
        "/usr/bin/camoufox",
    ] {
        let p = PathBuf::from(candidate);
        if p.exists() {
            return Some(p);
        }
    }
    if let Some(home) = dirs::home_dir() {
        for sub in [
            ".cache/camoufox/camoufox",
            ".cache/camoufox/browser/camoufox",
            ".local/bin/camoufox",
        ] {
            let p = home.join(sub);
            if p.exists() {
                return Some(p);
            }
        }
    }
    None
}

/// Scrape search result snippets using native headless Camoufox.
pub fn collect_snippets_camoufox(query: &str) -> Vec<String> {
    let Some(binary) = find_camoufox_binary() else {
        return Vec::new();
    };

    let profile_dir = match tempfile::tempdir() {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };

    let config = LaunchConfig {
        executable: binary,
        profile_dir: Some(profile_dir.path().to_owned()),
        headless: true,
        timeout: Duration::from_secs(10),
        close_timeout: Duration::from_secs(3),
        ..Default::default()
    };

    let mut launched = match process::unix::spawn(&config) {
        Ok(l) => l,
        Err(_) => return Vec::new(),
    };

    if process::readiness::wait_for_ready(&mut launched.child, config.timeout).is_err() {
        let _ = launched.child.kill();
        let _ = launched.child.wait();
        return Vec::new();
    }

    let transport = PipeTransport::new(launched.command_pipe, launched.response_pipe);
    let conn = Connection::new(Box::new(transport));
    let root = conn.root_session();
    let browser = match Browser::connect(conn, root, BrowserOptions::default()) {
        Ok(b) => b,
        Err(_) => {
            let _ = launched.child.kill();
            let _ = launched.child.wait();
            return Vec::new();
        }
    };

    let snippets = fetch_snippets_from_browser(&browser, query);

    // Clean shutdown
    let _ = browser.close();
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline {
        match launched.child.try_wait() {
            Ok(Some(_)) | Err(_) => break,
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
        }
    }
    if launched.child.try_wait().map(|s| s.is_none()).unwrap_or(false) {
        let _ = launched.child.kill();
        let _ = launched.child.wait();
    }

    snippets
}

fn fetch_snippets_from_browser(browser: &Browser, query: &str) -> Vec<String> {
    let context = match browser.new_context(ContextOptions::default()) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let frame = match context.new_main_frame() {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };

    let encoded_query = url_encode(query);
    let url = format!("https://html.duckduckgo.com/html/?q={encoded_query}");

    if frame.navigate(&url, Default::default(), PAGE_LOAD_TIMEOUT).is_err() {
        return Vec::new();
    }

    // Wait for DOM to load
    let deadline = Instant::now() + Duration::from_secs(8);
    while Instant::now() < deadline {
        let state = frame
            .evaluate("document.readyState", EVALUATE_TIMEOUT)
            .ok()
            .and_then(|v| {
                v.get("result")
                    .and_then(|r| r.get("value"))
                    .or_else(|| v.get("value"))
                    .and_then(|val| val.as_str())
                    .map(String::from)
            })
            .unwrap_or_default();

        if state == "complete" || state == "interactive" {
            break;
        }
        std::thread::sleep(Duration::from_millis(150));
    }

    let extract_js = r#"
        JSON.stringify(
            Array.from(document.querySelectorAll('.result__snippet, .result-snippet, [data-testid="result-snippet"]'))
                .map(el => (el.innerText || el.textContent || '').trim())
                .filter(Boolean)
        )
    "#;

    let res = frame.evaluate(extract_js, EVALUATE_TIMEOUT).ok();
    let json_str = res
        .and_then(|v| {
            v.get("result")
                .and_then(|r| r.get("value"))
                .or_else(|| v.get("value"))
                .and_then(|val| val.as_str())
                .map(String::from)
        })
        .unwrap_or_default();

    serde_json::from_str::<Vec<String>>(&json_str).unwrap_or_default()
}

fn url_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            b' ' => out.push('+'),
            _ => {
                out.push_str(&format!("%{:02X}", b));
            }
        }
    }
    out
}
