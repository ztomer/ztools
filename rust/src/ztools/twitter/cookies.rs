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
            expires: None,
        }
    }
}

/// Find potential Firefox profile cookie databases.
pub fn find_firefox_profile_dbs() -> Vec<PathBuf> {
    dirs::home_dir()
        .map(|home| find_profile_dbs_under(&home))
        .unwrap_or_default()
}

/// Scan `home` for Firefox profile cookie databases.
///
/// Narrow seam over [`find_firefox_profile_dbs`] so the discovery logic is
/// testable against a fixture directory instead of the real user home.
pub fn find_profile_dbs_under(home: &Path) -> Vec<PathBuf> {
    let mut dbs = Vec::new();
    let profiles_dir = home
        .join("Library")
        .join("Application Support")
        .join("Firefox")
        .join("Profiles");

    if profiles_dir.is_dir() {
        if let Ok(entries) = std::fs::read_dir(profiles_dir) {
            for entry in entries.filter_map(|e| e.ok()) {
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
pub fn has_session_cookie(cookies: &[Cookie]) -> bool {
    cookies
        .iter()
        .any(|c| c.name == SESSION_COOKIE_NAME && !c.value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_has_session_cookie() {
        let empty_cookies: Vec<Cookie> = vec![];
        assert!(!has_session_cookie(&empty_cookies));

        let guest_cookies = vec![
            Cookie::new("guest_id", "v1%3A123", ".x.com"),
            Cookie::new("ct0", "abcdef", ".x.com"),
        ];
        assert!(!has_session_cookie(&guest_cookies));

        let authed_cookies = vec![
            Cookie::new("guest_id", "v1%3A123", ".x.com"),
            Cookie::new("auth_token", "secret_session_token_12345", ".x.com"),
        ];
        assert!(has_session_cookie(&authed_cookies));
    }

    #[test]
    fn test_find_profile_dbs_under_collects_only_cookie_files() {
        let home = tempfile::tempdir().unwrap();
        let profiles = home
            .path()
            .join("Library/Application Support/Firefox/Profiles");
        std::fs::create_dir_all(profiles.join("abc123.default")).unwrap();
        std::fs::create_dir_all(profiles.join("def456.nightly")).unwrap();
        std::fs::create_dir_all(profiles.join("no_cookies_here")).unwrap();
        std::fs::write(profiles.join("abc123.default/cookies.sqlite"), b"sqlite").unwrap();
        // A DIRECTORY named cookies.sqlite must not count (is_file, not exists).
        std::fs::create_dir_all(profiles.join("def456.nightly/cookies.sqlite")).unwrap();

        let dbs = find_profile_dbs_under(home.path());

        assert_eq!(dbs.len(), 1);
        assert!(dbs[0].ends_with("abc123.default/cookies.sqlite"));
    }

    #[test]
    fn test_find_profile_dbs_under_missing_profiles_dir_is_empty() {
        let home = tempfile::tempdir().unwrap();
        // Home exists but has no Firefox profiles at all.
        assert!(find_profile_dbs_under(home.path()).is_empty());
    }

    #[test]
    fn test_find_firefox_profile_dbs_reports_only_cookie_dbs() {
        // Contract over the real home dir (no env mutation): whatever it finds
        // must be a cookies.sqlite inside a Firefox Profiles subtree. Whether
        // the user has Firefox at all is not part of this contract.
        for db in find_firefox_profile_dbs() {
            assert_eq!(db.file_name().unwrap().to_string_lossy(), "cookies.sqlite");
            let ancestors: Vec<_> = db
                .ancestors()
                .map(|p| p.to_string_lossy().to_string())
                .collect();
            assert!(
                ancestors.iter().any(|a| a.ends_with("Firefox/Profiles")),
                "{db:?} is not under a Firefox Profiles dir"
            );
            assert!(db.is_file(), "reported db {db:?} is not a file");
        }
    }
}
