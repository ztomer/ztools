use super::*;
use std::path::Path;

#[test]
fn normalize_expiry_bounds() {
    assert_eq!(normalize_expiry(0), None);
    assert_eq!(normalize_expiry(-1), None);
    assert_eq!(normalize_expiry(1_786_000_000), Some(1_786_000_000));
    // Zen-style milliseconds collapse to seconds.
    assert_eq!(normalize_expiry(1_786_000_000_000), Some(1_786_000_000));
}

/// One fixture row: name, value, host, path, expiry, secure, http-only,
/// origin attributes.
type FixtureCookieRow<'a> = (&'a str, &'a str, &'a str, &'a str, i64, i64, i64, &'a str);

fn fixture_cookie_db(dir: &Path, rows: &[FixtureCookieRow<'_>]) -> PathBuf {
    let db = dir.join("cookies.sqlite");
    let conn = rusqlite::Connection::open(&db).unwrap();
    conn.execute_batch(
        "CREATE TABLE moz_cookies (name TEXT, value TEXT, host TEXT, path TEXT, \
         expiry INTEGER, isSecure INTEGER, isHttpOnly INTEGER, originAttributes TEXT)",
    )
    .unwrap();
    for (name, value, host, path, expiry, sec, httponly, attrs) in rows {
        conn.execute(
            "INSERT INTO moz_cookies VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rusqlite::params![name, value, host, path, expiry, sec, httponly, attrs],
        )
        .unwrap();
    }
    conn.close().unwrap();
    db
}

#[test]
fn read_firefox_cookies_filters_like_python() {
    let dir = tempfile::tempdir().unwrap();
    let prof = dir.path().join("prof");
    std::fs::create_dir_all(&prof).unwrap();
    let db = fixture_cookie_db(
        &prof,
        &[
            (
                "auth_token",
                "sess123",
                ".x.com",
                "/",
                1_786_000_000,
                1,
                1,
                "",
            ),
            // Container-tab duplicate: must not surface.
            (
                "auth_token",
                "other",
                ".x.com",
                "/",
                1_786_000_000,
                1,
                1,
                "userContextId=1",
            ),
            // Empty value: skipped.
            ("ct0", "", ".x.com", "/", 1_786_000_000, 0, 0, ""),
            // Naive LIKE would match; dotted-suffix must not.
            ("junk", "v", "fax.com", "/", 1_786_000_000, 0, 0, ""),
            ("ad", "v", "notx.com", "/", 1_786_000_000, 0, 0, ""),
            // Subdomain does match.
            ("ct0", "abc", "api.x.com", "/", 1_786_000_000_000, 0, 0, ""),
        ],
    );
    let cookies = read_firefox_cookies(&db, DEFAULT_DOMAINS).unwrap();
    let names: Vec<_> = cookies
        .iter()
        .map(|c| (c.name.as_str(), c.domain.as_str()))
        .collect();
    assert!(names.contains(&("auth_token", ".x.com")), "{names:?}");
    assert!(names.contains(&("ct0", "api.x.com")), "{names:?}");
    assert_eq!(names.len(), 2, "{names:?}");
    let ct0 = cookies.iter().find(|c| c.name == "ct0").unwrap();
    // Milliseconds normalized; httpOnly carried.
    assert_eq!(ct0.expires, Some(1_786_000_000));
    assert!(!ct0.http_only);
    let sess = cookies.iter().find(|c| c.name == "auth_token").unwrap();
    assert!(sess.secure && sess.http_only);
}

#[test]
fn unreadable_db_yields_empty_not_error() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("cookies.sqlite");
    std::fs::write(&db, b"not a database").unwrap();
    let cookies = read_firefox_cookies(&db, DEFAULT_DOMAINS).unwrap();
    assert!(cookies.is_empty());
}

#[test]
fn firefox_family_dbs_prefers_zen_and_dedupes_case_spellings() {
    let home = tempfile::tempdir().unwrap();
    let support = home.path().join("Library/Application Support");
    for root in ["Zen/Profiles/p1", "Firefox/Profiles/p2"] {
        let d = support.join(root);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("cookies.sqlite"), b"x").unwrap();
    }
    let dbs = firefox_family_dbs(&support);
    assert_eq!(dbs.len(), 2, "{dbs:?}");
    // First spelling wins for display; the filesystem may fold case.
    assert!(
        dbs[0].to_string_lossy().to_lowercase().contains("zen"),
        "{dbs:?}"
    );
}

#[test]
fn find_session_cookies_first_session_wins() {
    let home = tempfile::tempdir().unwrap();
    let support = home.path().join("Library/Application Support");
    // Zen profile sorts first but holds guest-only cookies; Firefox holds
    // the session — the session must win over order.
    let zen = support.join("Zen/Profiles/guest");
    std::fs::create_dir_all(&zen).unwrap();
    fixture_cookie_db(
        &zen,
        &[("ct0", "g", ".x.com", "/", 1_786_000_000, 0, 0, "")],
    );
    let ff = support.join("Firefox/Profiles/authed");
    std::fs::create_dir_all(&ff).unwrap();
    fixture_cookie_db(
        &ff,
        &[("auth_token", "s", ".x.com", "/", 1_786_000_000, 1, 1, "")],
    );
    let (cookies, source) = find_session_cookies(&support, DEFAULT_DOMAINS);
    assert!(has_session_cookie(&cookies));
    assert!(source.contains("Firefox"), "{source}");
}

#[test]
fn find_session_cookies_falls_back_to_guest_only() {
    let home = tempfile::tempdir().unwrap();
    let support = home.path().join("Library/Application Support");
    let zen = support.join("Zen/Profiles/guest");
    std::fs::create_dir_all(&zen).unwrap();
    fixture_cookie_db(
        &zen,
        &[("ct0", "g", ".x.com", "/", 1_786_000_000, 0, 0, "")],
    );
    let (cookies, source) = find_session_cookies(&support, DEFAULT_DOMAINS);
    assert!(!has_session_cookie(&cookies));
    assert_eq!(cookies.len(), 1);
    assert!(source.to_lowercase().contains("zen"), "{source}");
}

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
