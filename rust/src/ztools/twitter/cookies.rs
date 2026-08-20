//! Cookie extraction and representation for Twitter/X authentication.
//!
//! Port of `twitter/cookies.py` and `twitter/cookies_firefox.py`.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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
    let mut dbs = Vec::new();
    if let Some(home) = dirs::home_dir() {
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
}
