//! Cookie extraction and representation for Twitter/X authentication.
//!
//! Port of `twitter/cookies.py` and `twitter/cookies_firefox.py`.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const DEFAULT_DOMAINS: &[&str] = &[".twitter.com", ".x.com", "twitter.com", "x.com"];
pub const SESSION_COOKIE_NAME: &str = "auth_token";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Cookie {
    pub name: String,
    pub value: String,
    pub domain: String,
    pub path: String,
    pub secure: bool,
    pub http_only: bool,
    pub expires: Option<i64>,
}

impl Cookie {
    pub fn new(
        name: impl Into<String>,
        value: impl Into<String>,
        domain: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
            domain: domain.into(),
            path: "/".to_string(),
            secure: true,
            http_only: false,
            expires: None,
        }
    }
}

/// Find potential Firefox profile cookie databases.
#[must_use]
pub fn find_firefox_profile_dbs() -> Vec<PathBuf> {
    dirs::home_dir()
        .map(|home| find_profile_dbs_under(&home))
        .unwrap_or_default()
}

/// Scan `home` for Firefox profile cookie databases.
///
/// Narrow seam over [`find_firefox_profile_dbs`] so the discovery logic is
/// testable against a fixture directory instead of the real user home.
#[must_use]
pub fn find_profile_dbs_under(home: &Path) -> Vec<PathBuf> {
    let mut dbs = Vec::new();
    let profiles_dir = home
        .join("Library")
        .join("Application Support")
        .join("Firefox")
        .join("Profiles");

    if profiles_dir.is_dir() {
        if let Ok(entries) = std::fs::read_dir(profiles_dir) {
            for entry in entries.filter_map(std::result::Result::ok) {
                let cookie_path = entry.path().join("cookies.sqlite");
                if cookie_path.is_file() {
                    dbs.push(cookie_path);
                }
            }
        }
    }
    dbs
}

/// Check if a cookie collection contains a valid logged-in session token.
#[must_use]
pub fn has_session_cookie(cookies: &[Cookie]) -> bool {
    cookies
        .iter()
        .any(|c| c.name == SESSION_COOKIE_NAME && !c.value.is_empty())
}

/// Profile roots under `~/Library/Application Support`, in preference order.
///
/// Port of `FIREFOX_FAMILY_ROOTS` — both capitalizations are listed because
/// the on-disk spelling is `Zen` while the Python code relies on a
/// case-insensitive filesystem to resolve `zen`. Rust spells both.
pub const FIREFOX_FAMILY_ROOTS: &[&str] = &[
    "zen/Profiles",
    "Zen/Profiles",
    "Firefox/Profiles",
    "LibreWolf/Profiles",
    "librewolf/Profiles",
    "Waterfox/Profiles",
    "waterfox/Profiles",
];

/// Anything above this cannot be epoch seconds (year 5138) — it is
/// milliseconds, which newer Firefox-family builds (Zen included) store.
/// Port of `_MAX_PLAUSIBLE_EPOCH_SECONDS` + `normalize_expiry`.
#[must_use]
pub const fn normalize_expiry(expiry: i64) -> Option<i64> {
    if expiry <= 0 {
        return None;
    }
    if expiry > 100_000_000_000 {
        return Some(expiry / 1000);
    }
    Some(expiry)
}

/// Every Firefox-family cookie DB under `app_support`, in preference order.
///
/// Narrow seam: production passes `~/Library/Application Support`, tests pass
/// a fixture dir. Duplicate DBs (same file reached through two spellings on a
/// case-insensitive filesystem) are reported once.
#[must_use]
pub fn firefox_family_dbs(app_support: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    // Canonicalized paths for identity: on a case-insensitive filesystem the
    // `zen` and `Zen` roots resolve to the same directory, but the spelled
    // paths differ as strings — dedupe by identity, keep first (preference
    // order) spelling for display.
    let mut seen: Vec<PathBuf> = Vec::new();
    for root in FIREFOX_FAMILY_ROOTS {
        let dir = app_support.join(root);
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut dbs: Vec<PathBuf> = entries
            .filter_map(Result::ok)
            .map(|e| e.path().join("cookies.sqlite"))
            .filter(|p| p.is_file())
            .collect();
        dbs.sort();
        for db in dbs {
            let identity = std::fs::canonicalize(&db).unwrap_or_else(|_| db.clone());
            if !seen.contains(&identity) {
                seen.push(identity);
                out.push(db);
            }
        }
    }
    out
}

fn bare_domain(d: &str) -> &str {
    d.strip_prefix('.').unwrap_or(d)
}

/// One `moz_cookies` row, decoded once so the loop below reads fields by name.
struct FirefoxRow {
    name: String,
    value: String,
    host: String,
    path: Option<String>,
    expiry: i64,
    is_secure: bool,
    is_http_only: bool,
    origin_attrs: Option<String>,
}

/// Read matching cookies out of one Firefox profile DB.
///
/// Mirrors `read_firefox_cookies`: the DB is COPIED to a tempfile first (it
/// may be WAL-locked by a running browser), hosts match by exact or dotted
/// suffix (never a naive `%x.com` LIKE — that also matches netflix.com),
/// non-default containers are skipped, empty values are skipped, and
/// (name, host, path) triples dedupe.
///
/// # Errors
///
/// Only I/O failures (copy/read) surface; a corrupt or unreadable DB yields an
/// empty vec, exactly like the Python `except Exception: return []`.
pub fn read_firefox_cookies(
    db_path: &Path,
    domains: &[&str],
) -> Result<Vec<Cookie>, std::io::Error> {
    let tmp = tempfile::NamedTempFile::new()?;
    std::fs::copy(db_path, tmp.path())?;
    let Ok(conn) = rusqlite::Connection::open_with_flags(
        tmp.path(),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    ) else {
        return Ok(Vec::new());
    };

    let mut clauses = Vec::new();
    let mut params: Vec<String> = Vec::new();
    for d in domains {
        let bare = bare_domain(d);
        clauses.push("(host = ? OR host = ? OR host LIKE ?)".to_owned());
        params.push(bare.to_owned());
        params.push(format!(".{bare}"));
        params.push(format!("%.{bare}"));
    }
    let sql = format!(
        "SELECT name, value, host, path, expiry, isSecure, isHttpOnly, originAttributes \
         FROM moz_cookies WHERE {}",
        clauses.join(" OR ")
    );
    let param_refs: Vec<&dyn rusqlite::ToSql> =
        params.iter().map(|p| p as &dyn rusqlite::ToSql).collect();

    let Ok(mut stmt) = conn.prepare(&sql) else {
        return Ok(Vec::new());
    };
    let rows: Vec<FirefoxRow> = match stmt.query_map(&param_refs[..], |row| {
        Ok(FirefoxRow {
            name: row.get(0)?,
            value: row.get(1)?,
            host: row.get(2)?,
            path: row.get(3)?,
            expiry: row.get(4)?,
            is_secure: row.get::<_, i64>(5)? != 0,
            is_http_only: row.get::<_, i64>(6)? != 0,
            origin_attrs: row.get(7)?,
        })
    }) {
        Ok(mapped) => mapped.filter_map(Result::ok).collect(),
        Err(_) => return Ok(Vec::new()),
    };

    let mut seen = std::collections::HashSet::new();
    let mut cookies = Vec::new();
    for row in rows {
        if row.origin_attrs.as_deref().is_some_and(|s| !s.is_empty()) {
            continue;
        }
        if row.value.is_empty() {
            continue;
        }
        let path = row.path.unwrap_or_else(|| "/".to_owned());
        if !seen.insert((row.name.clone(), row.host.clone(), path.clone())) {
            continue;
        }
        cookies.push(Cookie {
            name: row.name,
            value: row.value,
            domain: row.host,
            path,
            secure: row.is_secure,
            http_only: row.is_http_only,
            expires: normalize_expiry(row.expiry),
        });
    }
    Ok(cookies)
}

/// Find an x.com session in any Firefox-family browser.
///
/// Port of `get_browser_cookies` minus the Chrome fallback (explicitly
/// deferred: the fleet session lives in Zen; Chrome-only users get a refusal
/// naming `--login`, never a silent downgrade). Returns `(cookies, source)`;
/// the source label is `{browser-dir}:{profile-dir}` like the Python one.
/// First session hit wins; otherwise the first guest-only cookies; otherwise
/// empty.
#[must_use]
pub fn find_session_cookies(app_support: &Path, domains: &[&str]) -> (Vec<Cookie>, String) {
    let mut guest_only: Vec<Cookie> = Vec::new();
    let mut guest_source = String::new();
    for db in firefox_family_dbs(app_support) {
        let cookies = read_firefox_cookies(&db, domains).unwrap_or_default();
        let source = format!(
            "{}:{}",
            db.parent()
                .and_then(|p| p.parent())
                .and_then(|p| p.parent())
                .and_then(|p| p.file_name())
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default(),
            db.parent()
                .and_then(|p| p.file_name())
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default()
        );
        if has_session_cookie(&cookies) {
            return (cookies, source);
        }
        if !cookies.is_empty() && guest_only.is_empty() {
            guest_only = cookies;
            guest_source = source;
        }
    }
    (guest_only, guest_source)
}

#[cfg(test)]
#[path = "cookies_tests.rs"]
mod tests;
