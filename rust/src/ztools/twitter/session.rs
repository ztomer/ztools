//! Persistent camoufox profile holding the x.com login.
//!
//! Port of `twitter/session.py`: the user signs in once through a headed
//! `--login` run, x.com issues its cookies to the profile directory, and every
//! later headless run reuses them. No password is ever seen or stored — only
//! the cookies x.com hands out, on disk, in the profile directory.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Profile location. `TWITTER_PROFILE_DIR` overrides the default, exactly like
/// the Python `PROFILE_DIR`.
#[must_use]
pub fn profile_dir() -> PathBuf {
    if let Ok(raw) = std::env::var("TWITTER_PROFILE_DIR") {
        return PathBuf::from(raw);
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/root"))
        .join(".twitter-camoufox-profile")
}

/// Login page and polling bounds from `session.py`.
pub const LOGIN_URL: &str = "https://x.com/i/flow/login";
pub const LOGIN_TIMEOUT: Duration = Duration::from_secs(300);
pub const LOGIN_POLL: Duration = Duration::from_secs(2);
/// A populated Firefox profile always contains this file; an empty dir means
/// `--login` was never completed.
pub const PROFILE_MARKER: &str = "cookies.sqlite";

/// True when the profile directory was actually populated by a login run.
#[must_use]
pub fn profile_exists(dir: &Path) -> bool {
    dir.join(PROFILE_MARKER).is_file()
}

/// True when the on-disk profile holds an x.com session token.
///
/// Reads the profile's own `cookies.sqlite` through the same Firefox-family
/// reader the installed-browser path uses — no browser launch, so this is
/// cheap enough to check before every headless run.
///
/// # Errors
///
/// Only when the marker file is missing (login never completed). An unreadable
/// but present DB reads as no-session, never as an error: a corrupt profile
/// and a logged-out profile demand the same remedy (`--login`).
pub fn has_saved_session(dir: &Path) -> Result<bool, SavedSessionError> {
    if !profile_exists(dir) {
        return Err(SavedSessionError::NoProfile);
    }
    let db = dir.join(PROFILE_MARKER);
    let cookies = super::cookies::read_firefox_cookies(&db, super::cookies::DEFAULT_DOMAINS)
        .unwrap_or_default();
    Ok(super::cookies::has_session_cookie(&cookies))
}

/// Why a saved session cannot be used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SavedSessionError {
    /// The profile directory was never populated — run `--login` first.
    NoProfile,
}

impl std::fmt::Display for SavedSessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoProfile => write!(
                f,
                "no saved x.com session — run `twitter --login` once, then retry"
            ),
        }
    }
}

impl std::error::Error for SavedSessionError {}

/// Poll `has_session` until it reports true or `timeout` elapses.
///
/// Pure over injected time + probe so the login wait is testable without a
/// browser: the deadline, the poll interval, and the "user closed the window"
/// abort all surface here, not in the driver.
///
/// # Errors
///
/// [`WaitError::Aborted`] when the login window goes away first;
/// [`WaitError::TimedOut`] when no session appears before `timeout`.
pub fn wait_for_session(
    mut has_session: impl FnMut() -> bool,
    mut aborted: impl FnMut() -> bool,
    timeout: Duration,
    poll: Duration,
    mut now: impl FnMut() -> Instant,
    mut sleep: impl FnMut(Duration),
) -> Result<(), WaitError> {
    let start = now();
    loop {
        if has_session() {
            return Ok(());
        }
        if aborted() {
            return Err(WaitError::Aborted);
        }
        if now().duration_since(start) >= timeout {
            return Err(WaitError::TimedOut);
        }
        sleep(poll);
    }
}

/// Why a headed login wait ended without a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitError {
    /// The login window went away before any session appeared.
    Aborted,
    /// `LOGIN_TIMEOUT` elapsed with no session cookie.
    TimedOut,
}

impl std::fmt::Display for WaitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Aborted => write!(f, "login window closed before sign-in completed"),
            Self::TimedOut => write!(f, "no x.com session appeared before the login timeout"),
        }
    }
}

impl std::error::Error for WaitError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    #[test]
    fn empty_dir_is_no_profile() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!profile_exists(dir.path()));
        assert_eq!(
            has_saved_session(dir.path()).unwrap_err(),
            SavedSessionError::NoProfile
        );
    }

    #[test]
    fn marker_without_session_reads_false() {
        let dir = tempfile::tempdir().unwrap();
        // A valid but sessionless cookies.sqlite: present profile, no login.
        let conn = rusqlite::Connection::open(dir.path().join(PROFILE_MARKER)).unwrap();
        conn.execute_batch(
            "CREATE TABLE moz_cookies (name TEXT, value TEXT, host TEXT, path TEXT, \
             expiry INTEGER, isSecure INTEGER, isHttpOnly INTEGER, originAttributes TEXT)",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO moz_cookies VALUES ('ct0', 'g', '.x.com', '/', 1786000000, 0, 0, '')",
            [],
        )
        .unwrap();
        conn.close().unwrap();
        assert!(profile_exists(dir.path()));
        assert!(!has_saved_session(dir.path()).unwrap());
    }

    #[test]
    fn wait_returns_on_first_session_sight() {
        let calls = Rc::new(Cell::new(0));
        let c2 = calls.clone();
        let r = wait_for_session(
            move || {
                c2.set(c2.get() + 1);
                c2.get() >= 3
            },
            || false,
            Duration::from_secs(300),
            Duration::from_secs(2),
            Instant::now,
            |_| {},
        );
        assert!(r.is_ok());
        assert_eq!(calls.get(), 3);
    }

    #[test]
    fn wait_reports_abort_and_timeout() {
        assert_eq!(
            wait_for_session(
                || false,
                || true,
                Duration::from_secs(300),
                Duration::from_secs(2),
                Instant::now,
                |_| {}
            ),
            Err(WaitError::Aborted)
        );
        let t0 = Instant::now();
        let now = Cell::new(t0);
        assert_eq!(
            wait_for_session(
                || false,
                || false,
                Duration::from_secs(10),
                Duration::from_secs(2),
                || now.get(),
                |d| now.set(now.get() + d),
            ),
            Err(WaitError::TimedOut)
        );
    }
}
