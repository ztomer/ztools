//! Native `camoufox-rs` collect + login driver (Phase 2a-ii).
//!
//! Behavioral port of `collect_tweets_via_browser` and `run_login` in
//! `twitter/browser.py` + `twitter/session.py`. Additive: the Python subprocess
//! path stays the default until Phase 3 A/B passes; `TWITTER_COLLECTOR=native`
//! selects this driver. Every pure decision (scroll stops, dedup, login check)
//! lives in [`collect`]; this module only drives the browser and feeds it
//! observations.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use chrono::{DateTime, FixedOffset, Utc};

use super::browser_bin;
use super::capture::{
    collected_wrong_feed, drain_responses, record_request, record_response, CaptureState,
};
use super::collect::{
    collect_dedup, is_logged_out, resolve_since, ScrollLimits, ScrollObservation, ScrollState,
};
use super::cookies::{self, Cookie};
use super::endpoints::load_endpoint_markers;
use super::session;
use super::Tweet;

/// Timeline page, wait bounds, and scroll pacing from `browser.py`.
pub const TWITTER_HOME_URL: &str = "https://x.com/home";
pub const PAGE_LOAD_TIMEOUT: Duration = Duration::from_secs(30);
pub const INITIAL_PAGE_WAIT: Duration = Duration::from_secs(3);
pub const TAB_SWITCH_WAIT: Duration = Duration::from_secs(2);
pub const SCROLL_PAUSE: Duration = Duration::from_millis(1800);
/// Localized "Following" labels, in `browser.py` order, overridable via
/// `TWITTER_FOLLOWING_TAB` (comma-separated).
pub const FOLLOWING_TERMS: &[&str] = &[
    "Following",
    "Abonnements",
    "Siguiendo",
    "Sigo",
    "Gefolgt",
    "关注",
    "フォロー中",
];

fn following_terms() -> Vec<String> {
    if let Ok(raw) = std::env::var("TWITTER_FOLLOWING_TAB") {
        let terms: Vec<String> = raw
            .split(',')
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(str::to_owned)
            .collect();
        if !terms.is_empty() {
            return terms;
        }
    }
    FOLLOWING_TERMS.iter().map(ToString::to_string).collect()
}

fn extract(value: &serde_json::Value) -> serde_json::Value {
    value
        .get("result")
        .and_then(|r| r.get("value"))
        .cloned()
        .unwrap_or_else(|| value.clone())
}

fn eval_value(
    frame: &camoufox::api::MainFrame,
    expr: &str,
    what: &str,
) -> Result<serde_json::Value> {
    let v = frame
        .evaluate(expr, Duration::from_secs(15))
        .with_context(|| format!("evaluate failed: {what}"))?;
    Ok(extract(&v))
}

fn eval_text(frame: &camoufox::api::MainFrame, expr: &str, what: &str) -> Result<String> {
    let v = eval_value(frame, expr, what)?;
    // Numbers and booleans render as JSON text; strings come through bare.
    // (`window.scrollY` is a number — reading it as text yields "".)
    match &v {
        serde_json::Value::String(s) => Ok(s.clone()),
        serde_json::Value::Null => Ok(String::new()),
        other => Ok(other.to_string()),
    }
}

fn wait_ready(frame: &camoufox::api::MainFrame, timeout: Duration) -> Result<()> {
    let deadline = Instant::now() + timeout;
    loop {
        let state = eval_text(frame, "document.readyState", "readyState")?;
        if state == "complete" || Instant::now() >= deadline {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// A launched browser and its profile dir guard (temp profiles die with it).
struct Launched {
    child: std::process::Child,
    browser: camoufox::api::Browser,
    _profile_guard: Option<tempfile::TempDir>,
}

fn launch(
    headless: bool,
    persistent_profile: Option<PathBuf>,
    subscribe: impl FnOnce(&camoufox::protocol::client::Connection),
) -> Result<Launched> {
    use camoufox::config::LaunchConfig;
    let binary = browser_bin::resolve()?;
    let (dir, guard) = if let Some(d) = persistent_profile {
        std::fs::create_dir_all(&d)?;
        (d, None)
    } else {
        let t = tempfile::TempDir::new().context("temp profile dir")?;
        let d = t.path().to_path_buf();
        (d, Some(t))
    };
    let config = LaunchConfig {
        executable: binary,
        profile_dir: Some(dir),
        headless,
        ..Default::default()
    };
    let mut launched = camoufox::process::unix::spawn(&config).context("spawn camoufox")?;
    camoufox::process::readiness::wait_for_ready(&mut launched.child, config.timeout)
        .context("camoufox readiness")?;
    let transport = camoufox::transport::pipe::PipeTransport::new(
        launched.command_pipe,
        launched.response_pipe,
    );
    let conn = camoufox::protocol::client::Connection::new(Box::new(transport));
    // Subscribe BEFORE connect: `Connection` moves into `Browser`.
    subscribe(&conn);
    let root = conn.root_session();
    let browser =
        camoufox::api::Browser::connect(conn, root, camoufox::api::BrowserOptions::default())
            .context("juggler connect")?;
    Ok(Launched {
        child: launched.child,
        browser,
        _profile_guard: guard,
    })
}

fn shutdown(sess: Launched) {
    let Launched {
        mut child,
        browser,
        _profile_guard,
    } = sess;
    let _ = browser.close();
    let _ = child.kill();
    let _ = child.wait();
}

/// Inject cookies one at a time so a single malformed cookie cannot drop the
/// batch — then verify the session cookie survived, exactly like
/// `_inject_cookies` (a rejected `expires` once discarded all 17 cookies and
/// the only symptom was the logged-out page).
///
/// # Errors
///
/// When the session cookie is absent after injection: x.com would serve a
/// logged-out page, so collecting would scroll marketing copy.
fn inject_cookies(ctx: &camoufox::api::BrowserContext, cookies: &[Cookie]) -> Result<()> {
    let mut rejected = 0;
    for c in cookies {
        // No `expires`: the collect browser runs on a temp profile that dies
        // with the run, so cookie persistence is meaningless — and dropping
        // the field dodges an `i64 → f64` precision question that has no good
        // answer at this call site.
        let opt = camoufox::api::CookieOptions {
            name: c.name.clone(),
            value: c.value.clone(),
            url: None,
            domain: Some(c.domain.clone()),
            path: Some(c.path.clone()),
            secure: Some(c.secure),
            http_only: Some(c.http_only),
            same_site: None,
            expires: None,
        };
        if ctx.set_cookies(std::slice::from_ref(&opt)).is_err() {
            rejected += 1;
        }
    }
    if rejected > 0 {
        eprintln!(
            "· {rejected}/{} cookies rejected by the browser",
            cookies.len()
        );
    }
    let live = ctx.get_cookies().unwrap_or_default();
    if !live
        .iter()
        .any(|c| c.name == cookies::SESSION_COOKIE_NAME && !c.value.is_empty())
    {
        anyhow::bail!(
            "the session cookie ({}) did not survive injection — x.com would serve a logged-out page",
            cookies::SESSION_COOKIE_NAME
        );
    }
    Ok(())
}

/// Click the Following tab via injected JS (there is no locator API in
/// `camoufox-rs`; with humanize off a synthetic `.click()` lands — the
/// animated-cursor timeouts that forced `force=True` in Python do not apply).
/// Tolerant: a missing tab keeps the current view, exactly like the Python
/// fallback chain (localized terms → second tab → warn and continue).
fn switch_to_following(frame: &camoufox::api::MainFrame) {
    let terms_json = serde_json::to_string(&following_terms()).unwrap_or_else(|_| "[]".to_string());
    let script = format!(
        "(function() {{ \
           const terms = {terms_json}; \
           const tabs = Array.from(document.querySelectorAll('[role=\"tab\"]')); \
           for (const term of terms) {{ \
             const hit = tabs.find(t => (t.innerText || '').includes(term)); \
             if (hit) {{ hit.click(); return 'term:' + term; }} \
           }} \
           if (tabs.length > 1) {{ tabs[1].click(); return 'fallback-index-1'; }} \
           return 'none'; \
         }})()"
    );
    let clicked = eval_text(frame, &script, "following tab click").unwrap_or_default();
    if clicked == "none" {
        eprintln!("· could not locate 'Following' tab (defaulting to current view)");
        return;
    }
    let deadline = Instant::now() + TAB_SWITCH_WAIT;
    while Instant::now() < deadline {
        let selected = eval_text(
            frame,
            "document.querySelector('[role=\"tab\"][aria-selected=\"true\"]') !== null",
            "tab state",
        )
        .unwrap_or_default();
        if selected == "true" {
            break;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// Open the timeline: navigate, settle, refuse logged-out pages, switch to the
/// Following tab. Returns the ready frame.
///
/// # Errors
///
/// When navigation or settling fails, or x.com serves a logged-out page (the
/// session went stale — collecting would scroll marketing copy).
///
/// # Panics
///
/// Never by itself; the scroll that follows panics only if the capture mutex
/// is poisoned, which requires a panicking reader-thread handler.
fn open_timeline(
    ctx: &camoufox::api::BrowserContext,
    source: &str,
) -> Result<camoufox::api::MainFrame> {
    let frame = ctx.new_main_frame().context("main frame")?;
    frame
        .navigate(
            TWITTER_HOME_URL,
            camoufox::api::NavigateOptions::default(),
            PAGE_LOAD_TIMEOUT,
        )
        .context("navigate to timeline")?;
    wait_ready(&frame, PAGE_LOAD_TIMEOUT)?;

    // Tolerant initial selector wait (mirrors the try/except pass in Python).
    let selector_deadline = Instant::now() + INITIAL_PAGE_WAIT;
    while Instant::now() < selector_deadline {
        let present = eval_text(
            &frame,
            "document.querySelector('[data-testid=\"primaryColumn\"], [role=\"tablist\"]') !== null",
            "initial selector",
        )
        .unwrap_or_default();
        if present == "true" {
            break;
        }
        std::thread::sleep(Duration::from_millis(200));
    }

    let url = eval_text(&frame, "window.location.href", "page url")?;
    let title = eval_text(&frame, "document.title", "page title")?;
    if is_logged_out(&url, &title) {
        anyhow::bail!(
            "x.com served a logged-out page — the {source} session is stale; \
             re-open x.com in that browser, or run `twitter --login`"
        );
    }

    switch_to_following(&frame);
    Ok(frame)
}

/// Scroll the timeline to the stop conditions, draining GraphQL responses as
/// they arrive. Returns the raw tweets plus the human-readable stop reason.
///
/// # Panics
///
/// If the capture mutex is poisoned — only possible when a reader-thread
/// event handler panicked, which the recording handler cannot do.
fn scroll_collect(
    frame: &camoufox::api::MainFrame,
    state: &Arc<Mutex<CaptureState>>,
    since_dt: &DateTime<FixedOffset>,
) -> (Vec<Tweet>, String) {
    let limits = ScrollLimits::from_env_or_default();
    let mut scroll = ScrollState::default();
    let mut tweets: Vec<Tweet> = Vec::new();
    let mut oldest: Option<DateTime<FixedOffset>> = None;
    let mut seen: HashSet<String> = HashSet::new();
    let start = Instant::now();
    let reason = loop {
        let before = tweets.len();
        // Bare expressions only: `Runtime.evaluate` takes an expression, and the
        // spike proved this split form (`scrollBy(...)` then `scrollY`)
        // while semicolon programs fail to parse a result.
        let scrolled = eval_value(
            frame,
            "window.scrollBy(0, window.innerHeight * 2)",
            "scroll",
        );
        let offset = scrolled.ok().and_then(|_| {
            eval_value(frame, "window.scrollY", "scroll offset")
                .ok()
                .and_then(|v| v.as_f64())
        });
        if offset.is_none() {
            break String::from("the page stopped responding");
        }
        std::thread::sleep(SCROLL_PAUSE);
        drain_responses(frame, state, &mut seen, &mut tweets, &mut oldest);
        let obs = ScrollObservation {
            grew: tweets.len() > before,
            offset,
            oldest_seen: oldest,
        };
        if let Some(reason) = scroll.step(&obs, since_dt, start.elapsed().as_secs_f64(), &limits) {
            break reason;
        }
    };
    (tweets, reason)
}
/// Collect timeline tweets through a native `camoufox-rs` browser.
///
/// Additive Phase 2a-ii path behind `TWITTER_COLLECTOR=native`; behavior
/// mirrors `collect_tweets_via_browser`.
///
/// # Errors
///
/// When no session exists anywhere, no twitter.toml `[endpoints]` table can
/// be read, the browser cannot start, x.com serves a logged-out page, the
/// feed never switches to Following, or nothing was collected (summarizing
/// nothing produces a confident summary of no data).
///
/// # Panics
///
/// If the capture mutex is poisoned — only possible when a reader-thread
/// event handler panicked, which the recording handler cannot do.
pub fn collect_tweets_native(
    since: Option<&str>,
    debug: bool,
    config: &crate::config::ZtoolsConfig,
) -> Result<Vec<Tweet>> {
    let support = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/root"))
        .join("Library/Application Support");
    let (cookies, source) = cookies::find_session_cookies(&support, cookies::DEFAULT_DOMAINS);
    if !cookies::has_session_cookie(&cookies) {
        if cookies.is_empty() {
            anyhow::bail!(
                "no x.com session in any Firefox-family browser — sign in, or run `twitter --login`"
            );
        }
        anyhow::bail!(
            "{source} has {} x.com cookies but no session token ({}) — that browser is not signed in",
            cookies.len(),
            cookies::SESSION_COOKIE_NAME
        );
    }

    let state = Arc::new(Mutex::new(CaptureState::new(load_endpoint_markers(
        &config.twitter_config_paths,
    )?)));
    let feed = state.clone();
    let sess = launch(!debug, None, |conn| {
        conn.on_event_global(Box::new(move |ev| {
            if ev.method == "Network.requestWillBeSent" {
                let rid = ev
                    .params
                    .get("requestId")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let url = ev.params.get("url").and_then(|v| v.as_str()).unwrap_or("");
                record_request(&feed, rid, url);
            } else if ev.method == "Network.responseReceived" {
                record_response(&feed, &ev.params);
            }
        }));
    })?;

    let ctx = sess
        .browser
        .new_context(camoufox::api::ContextOptions::default())
        .context("new browser context")?;
    inject_cookies(&ctx, &cookies)?;
    let frame = match open_timeline(&ctx, &source) {
        Ok(frame) => frame,
        Err(e) => {
            shutdown(sess);
            return Err(e);
        }
    };
    let since_dt = resolve_since(since, Utc::now());
    let (tweets, stop_reason) = scroll_collect(&frame, &state, &since_dt);
    shutdown(sess);

    // A run with timeline traffic but no Following-endpoint traffic watched
    // the wrong feed: the tab switch silently fell back to For You, and its
    // ID set is not comparable to a Following run. Fail here rather than
    // produce a plausible-but-wrong set.
    if collected_wrong_feed(&state) {
        anyhow::bail!(
            "no Following-timeline responses captured — the feed never switched from For You"
        );
    }

    let unique = collect_dedup(&tweets, &since_dt);
    if unique.is_empty() {
        anyhow::bail!("No tweets collected from browser session ({stop_reason}).");
    }
    Ok(unique)
}

/// Run the headed login flow natively: persistent profile, wait for the user
/// to sign in.
///
/// # Errors
///
/// When the browser cannot start, the window closes first, or no session
/// appears before the timeout.
pub fn login_native() -> Result<()> {
    let dir = session::profile_dir();
    let mut sess = launch(false, Some(dir), |_| {})?;
    let ctx = sess
        .browser
        .new_context(camoufox::api::ContextOptions::default())
        .context("new browser context")?;
    let frame = ctx.new_main_frame().context("main frame")?;
    frame
        .navigate(
            session::LOGIN_URL,
            camoufox::api::NavigateOptions::default(),
            PAGE_LOAD_TIMEOUT,
        )
        .context("navigate to login")?;
    println!("· sign in to x.com in the opened window...");
    let outcome = session::wait_for_session(
        || {
            ctx.get_cookies()
                .unwrap_or_default()
                .iter()
                .any(|c| c.name == cookies::SESSION_COOKIE_NAME && !c.value.is_empty())
        },
        || sess.child.try_wait().ok().flatten().is_some(),
        session::LOGIN_TIMEOUT,
        session::LOGIN_POLL,
        Instant::now,
        std::thread::sleep,
    );
    shutdown(sess);
    outcome.map_err(anyhow::Error::from)?;
    println!("· signed in — future runs reuse this profile headlessly");
    Ok(())
}

/// Whether `TWITTER_COLLECTOR=native` selects this driver.
#[must_use]
pub fn native_selected() -> bool {
    std::env::var("TWITTER_COLLECTOR").is_ok_and(|v| v.trim().eq_ignore_ascii_case("native"))
}

#[path = "native_tests.rs"]
#[cfg(test)]
mod tests;
