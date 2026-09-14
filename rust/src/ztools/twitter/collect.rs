//! Pure timeline-collection logic: scroll stop conditions, collect dedup, login check.
//!
//! Port of the decision logic in `twitter/browser.py` (`_scroll_timeline`,
//! the collect post-pass, `_is_logged_out`). Everything here is browser-free:
//! the clock (`now`) and the page observations (`grew`, `moved`,
//! `oldest_seen`) are parameters, so every stop condition is unit-testable.
//! The live driver (`browser.rs` / Phase 2a-ii) only feeds observations in.

use chrono::{DateTime, FixedOffset, Utc};

use super::browser_parse::TWITTER_DATE_FORMAT;
use super::Tweet;

/// Preview widths from `browser.py` — CHARACTER counts, never byte counts.
/// Slicing a `text` with multi-byte characters by bytes panics; [`take_chars`]
/// does what Python's `text[:80]` does.
pub const EXACT_MATCH_PREVIEW_LIMIT: usize = 80;
pub const CONTENT_MATCH_PREVIEW_LIMIT: usize = 100;

/// Stop conditions bounding the scroll loop independently of the scroll cap.
/// Mirrors the module constants in `browser.py`.
#[derive(Debug, Clone)]
pub struct ScrollLimits {
    /// Hard cap on scroll iterations (`MAX_SCROLLS`, default 1200).
    pub max_scrolls: u32,
    /// Wall-clock budget in seconds (`MAX_RUNTIME_S`, default 300).
    pub max_runtime_secs: f64,
    /// Consecutive no-growth-no-movement scrolls before stopping
    /// (`STAGNANT_SCROLL_LIMIT`, default 8).
    pub stagnant_limit: u32,
}

impl Default for ScrollLimits {
    fn default() -> Self {
        Self {
            max_scrolls: 1200,
            max_runtime_secs: 300.0,
            stagnant_limit: 8,
        }
    }
}

impl ScrollLimits {
    /// Defaults from `browser.py`, honoring the same env overrides
    /// (`TWITTER_MAX_RUNTIME_S`). Malformed values fall back to defaults —
    /// a typo must not silently uncap a scheduled run.
    #[must_use]
    pub fn from_env_or_default() -> Self {
        let mut limits = Self::default();
        if let Ok(raw) = std::env::var("TWITTER_MAX_RUNTIME_S") {
            if let Ok(secs) = raw.trim().parse::<f64>() {
                if secs > 0.0 {
                    limits.max_runtime_secs = secs;
                }
            }
        }
        limits
    }
}

/// Mutable per-run scroll state. Feed one [`ScrollObservation`] per scroll via
/// [`ScrollState::step`]; it returns `None` to continue or `Some(reason)` to stop.
#[derive(Debug, Default)]
pub struct ScrollState {
    scrolls: u32,
    stagnant: u32,
    last_offset: Option<f64>,
    started_secs: Option<f64>,
}

/// One scroll's observations. `offset` is `window.scrollY` after scrolling;
/// `None` when the page would not evaluate (treated as moved — an unreadable
/// offset is not evidence the page is stuck).
pub struct ScrollObservation {
    /// True when the response handler grew the collection this round.
    pub grew: bool,
    /// `window.scrollY` after the scroll, or `None` when unreadable.
    pub offset: Option<f64>,
    /// Oldest tweet collected so far, or `None` when nothing arrived yet.
    pub oldest_seen: Option<DateTime<FixedOffset>>,
}

impl ScrollState {
    /// Decide whether to keep scrolling.
    ///
    /// `now_secs` / `start_secs` are wall-clock seconds on one clock (the
    /// caller owns the clock; tests use a fake). Returns the stop reason when
    /// the loop must end. Every exit path leaves the collected tweets intact —
    /// partial results survive; stopping is never an error.
    #[must_use]
    pub fn step(
        &mut self,
        obs: &ScrollObservation,
        since: &DateTime<FixedOffset>,
        now_secs: f64,
        limits: &ScrollLimits,
    ) -> Option<String> {
        if self.scrolls >= limits.max_scrolls {
            return Some(format!("reached the {}-scroll limit", limits.max_scrolls));
        }
        let start = *self.started_secs.get_or_insert(now_secs);
        if now_secs - start >= limits.max_runtime_secs {
            return Some(format!(
                "hit the {:.0}s runtime budget",
                limits.max_runtime_secs
            ));
        }
        self.scrolls += 1;

        if let Some(oldest) = obs.oldest_seen {
            if oldest < *since {
                return Some("reached the requested time window".to_owned());
            }
        }

        let moved = match (obs.offset, self.last_offset) {
            (Some(cur), Some(prev)) => (cur - prev).abs() > 0.5,
            // Unreadable offset: assume movement (see [`ScrollObservation`]).
            _ => true,
        };
        if obs.grew || moved {
            self.stagnant = 0;
        } else {
            self.stagnant += 1;
            if self.stagnant >= limits.stagnant_limit {
                return Some(format!(
                    "no new tweets and no page movement for {} consecutive scrolls",
                    limits.stagnant_limit
                ));
            }
        }
        if let Some(o) = obs.offset {
            self.last_offset = Some(o);
        }
        None
    }
}

/// First `n` characters of `s` — the Rust spelling of Python's `s[:n]`.
#[must_use]
pub fn take_chars(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}

/// Strip an `RT @user: ` prefix the way `RT_PREFIX_RE` does.
#[must_use]
pub fn strip_rt_prefix(text: &str) -> &str {
    if let Some(rest) = text.strip_prefix("RT @") {
        if let Some(pos) = rest.find(':') {
            let handle = &rest[..pos];
            if !handle.is_empty() && handle.chars().all(|c| c.is_alphanumeric() || c == '_') {
                return rest[pos + 1..].trim_start();
            }
        }
    }
    text
}

/// Parse a `created_at` in [`TWITTER_DATE_FORMAT`]; `None` when unparseable.
///
/// Unparseable tweets never survive the post-pass (the parser only emits ones
/// with a `created_at`, but the post-pass must not panic on a bad one).
#[must_use]
pub fn parse_created_at(s: &str) -> Option<DateTime<FixedOffset>> {
    DateTime::parse_from_str(s, TWITTER_DATE_FORMAT).ok()
}

/// The collect post-pass from `browser.py`: keep tweets at or after `since`,
/// drop exact duplicates (`screen_name` + first 80 chars) and content
/// duplicates (RT-stripped first 100 chars), oldest first.
#[must_use]
pub fn collect_dedup(tweets: &[Tweet], since: &DateTime<FixedOffset>) -> Vec<Tweet> {
    use std::collections::HashSet;
    let mut seen_exact: HashSet<(String, String)> = HashSet::new();
    let mut seen_content: HashSet<String> = HashSet::new();
    let mut unique: Vec<Tweet> = Vec::new();
    for t in tweets {
        let Some(created) = parse_created_at(&t.created_at) else {
            continue;
        };
        if created < *since {
            continue;
        }
        let exact_key = (
            t.screen_name.clone(),
            take_chars(&t.text, EXACT_MATCH_PREVIEW_LIMIT).to_owned(),
        );
        if !seen_exact.insert(exact_key) {
            continue;
        }
        let content_key =
            take_chars(strip_rt_prefix(&t.text), CONTENT_MATCH_PREVIEW_LIMIT).to_owned();
        if !seen_content.insert(content_key) {
            continue;
        }
        unique.push(t.clone());
    }
    unique.sort_by(|a, b| a.created_at.cmp(&b.created_at));
    unique
}

/// Resolve a `--since` value to an absolute time.
///
/// Mirrors `resolve_since_time` in `twitter/cli.py`: `<N>h` counts back from
/// `now`, ISO datetimes parse (naive ones read as UTC), anything else falls
/// back to 24h ago.
///
/// Accepted divergence: the Python fallback consults the `last_run` state file
/// (runs >4h ago reuse it); the native driver has no state file yet, so it
/// takes the same 24h default the Python path takes without one. Phase 3 A/B
/// always passes explicit `--since`, where both sides agree exactly.
#[must_use]
pub fn resolve_since(since: Option<&str>, now: DateTime<Utc>) -> DateTime<FixedOffset> {
    let fallback = now - chrono::Duration::hours(24);
    let Some(raw) = since.map(str::trim).filter(|s| !s.is_empty()) else {
        return fallback.fixed_offset();
    };
    if let Some(hours) = raw
        .strip_suffix('h')
        .and_then(|n| n.parse::<i64>().ok())
        .filter(|h| *h >= 0)
    {
        return (now - chrono::Duration::hours(hours)).fixed_offset();
    }
    if let Ok(dt) = DateTime::parse_from_rfc3339(raw) {
        return dt;
    }
    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"] {
        if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(raw, fmt) {
            return naive.and_utc().fixed_offset();
        }
    }
    if let Ok(date) = chrono::NaiveDate::parse_from_str(raw, "%Y-%m-%d") {
        if let Some(midnight) = date.and_hms_opt(0, 0, 0) {
            return midnight.and_utc().fixed_offset();
        }
    }
    fallback.fixed_offset()
}
pub const LOGGED_OUT_URL_MARKERS: &[&str] = &["/login", "/signin", "/i/flow/login"];
pub const LOGGED_OUT_ROOT_URLS: &[&str] =
    &["https://x.com", "https://twitter.com", "https://www.x.com"];
pub const LOGIN_KEYWORDS: &[&str] = &["log in", "login", "sign in", "signin"];

fn strip_url(url: &str) -> &str {
    let no_query = url.split('?').next().unwrap_or(url);
    let no_frag = no_query.split('#').next().unwrap_or(no_query);
    no_frag.trim_end_matches('/')
}

/// True when x.com served an anonymous page instead of the timeline.
///
/// Stale cookies do not always land on /login: requesting /home while signed
/// out bounces to the site root, so the URL, the stripped root, AND the title
/// are all checked — missing it looks like a hung scroll.
#[must_use]
pub fn is_logged_out(url: &str, title: &str) -> bool {
    let lower = url.to_lowercase();
    if LOGGED_OUT_URL_MARKERS.iter().any(|m| lower.contains(m)) {
        return true;
    }
    if LOGGED_OUT_ROOT_URLS.contains(&strip_url(&lower)) {
        return true;
    }
    let title_lower = title.to_lowercase();
    LOGIN_KEYWORDS.iter().any(|kw| title_lower.contains(kw))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dt(s: &str) -> DateTime<FixedOffset> {
        parse_created_at(s).unwrap()
    }

    const SINCE: &str = "Thu Aug 20 12:00:00 +0000 2026";

    fn obs(grew: bool, offset: Option<f64>) -> ScrollObservation {
        ScrollObservation {
            grew,
            offset,
            oldest_seen: None,
        }
    }

    #[test]
    fn scroll_stops_at_max_scrolls() {
        let limits = ScrollLimits {
            max_scrolls: 2,
            ..Default::default()
        };
        let since = dt(SINCE);
        let mut st = ScrollState::default();
        assert!(st
            .step(&obs(true, Some(100.0)), &since, 0.0, &limits)
            .is_none());
        assert!(st
            .step(&obs(true, Some(200.0)), &since, 1.0, &limits)
            .is_none());
        let reason = st
            .step(&obs(true, Some(300.0)), &since, 2.0, &limits)
            .unwrap();
        assert!(reason.contains("2-scroll limit"), "{reason}");
    }

    #[test]
    fn scroll_stops_on_runtime_budget() {
        let limits = ScrollLimits::default();
        let since = dt(SINCE);
        let mut st = ScrollState::default();
        assert!(st
            .step(&obs(true, Some(1.0)), &since, 0.0, &limits)
            .is_none());
        let reason = st
            .step(&obs(true, Some(2.0)), &since, 400.0, &limits)
            .unwrap();
        assert!(reason.contains("runtime budget"), "{reason}");
    }

    #[test]
    fn scroll_stops_on_time_window() {
        let limits = ScrollLimits::default();
        let since = dt("Thu Aug 20 13:00:00 +0000 2026");
        let mut st = ScrollState::default();
        let old = obs(true, Some(1.0));
        let old_with_seen = ScrollObservation {
            oldest_seen: Some(dt("Thu Aug 20 12:59:00 +0000 2026")),
            ..old
        };
        let reason = st.step(&old_with_seen, &since, 0.0, &limits).unwrap();
        assert!(reason.contains("time window"), "{reason}");
    }

    #[test]
    fn scroll_stops_on_stagnation_and_resets_on_growth() {
        let limits = ScrollLimits {
            stagnant_limit: 3,
            ..Default::default()
        };
        let since = dt(SINCE);
        let mut st = ScrollState::default();
        // Two stagnant, then growth resets, then three stagnant stop.
        assert!(st
            .step(&obs(false, Some(50.0)), &since, 0.0, &limits)
            .is_none());
        assert!(st
            .step(&obs(false, Some(50.0)), &since, 1.0, &limits)
            .is_none());
        assert!(st
            .step(&obs(true, Some(50.0)), &since, 2.0, &limits)
            .is_none());
        assert!(st
            .step(&obs(false, Some(50.0)), &since, 3.0, &limits)
            .is_none());
        assert!(st
            .step(&obs(false, Some(50.0)), &since, 4.0, &limits)
            .is_none());
        let reason = st
            .step(&obs(false, Some(50.0)), &since, 5.0, &limits)
            .unwrap();
        assert!(reason.contains("consecutive scrolls"), "{reason}");
    }

    #[test]
    fn unreadable_offset_counts_as_movement() {
        let limits = ScrollLimits {
            stagnant_limit: 2,
            ..Default::default()
        };
        let since = dt(SINCE);
        let mut st = ScrollState::default();
        // None offsets never increment stagnation on their own.
        for t in [0.0, 1.0, 2.0, 3.0] {
            assert!(st.step(&obs(false, None), &since, t, &limits).is_none());
        }
    }

    #[test]
    fn take_chars_never_splits_utf8() {
        assert_eq!(take_chars("abcdef", 3), "abc");
        assert_eq!(take_chars("关注中ab", 2), "关注");
        assert_eq!(take_chars("short", 80), "short");
    }

    #[test]
    fn strip_rt_prefix_matches_python_regex() {
        assert_eq!(strip_rt_prefix("RT @user: hello"), "hello");
        assert_eq!(strip_rt_prefix("RT @user_name123: hello"), "hello");
        assert_eq!(strip_rt_prefix("RT @: hello"), "RT @: hello");
        assert_eq!(strip_rt_prefix("RT hello"), "RT hello");
        assert_eq!(strip_rt_prefix("hello"), "hello");
    }

    fn tweet(screen: &str, text: &str, created: &str) -> Tweet {
        Tweet {
            id: String::new(),
            screen_name: screen.to_owned(),
            text: text.to_owned(),
            created_at: created.to_owned(),
            favorite_count: 0,
            retweet_count: 0,
            reply_to: None,
        }
    }

    #[test]
    fn dedup_drops_old_exact_and_content_dupes_and_sorts() {
        let since = dt(SINCE);
        let old = tweet("a", "too old", "Thu Aug 20 11:00:00 +0000 2026");
        let t1 = tweet("a", "hello world", "Thu Aug 20 12:05:00 +0000 2026");
        let t2 = tweet("a", "hello world", "Thu Aug 20 12:06:00 +0000 2026");
        let t3 = tweet("b", "RT @a: hello world", "Thu Aug 20 12:07:00 +0000 2026");
        let t4 = tweet("c", "something else", "Thu Aug 20 12:01:00 +0000 2026");
        let out = collect_dedup(&[t1, t2, t3, t4, old], &since);
        assert_eq!(out.len(), 2, "{out:?}");
        assert_eq!(out[0].screen_name, "c");
        assert_eq!(out[1].screen_name, "a");
    }

    #[test]
    fn dedup_unparseable_created_at_is_dropped() {
        let since = dt(SINCE);
        let bad = tweet("a", "hello", "not a date");
        assert!(collect_dedup(&[bad], &since).is_empty());
    }

    #[test]
    fn logged_out_matrix() {
        assert!(is_logged_out("https://x.com/i/flow/login", "Home"));
        assert!(is_logged_out("https://x.com/", "Welcome"));
        assert!(is_logged_out("https://x.com/home", "Log in to X"));
        assert!(!is_logged_out("https://x.com/home", "Home / X"));
    }

    #[test]
    fn resolve_since_matches_python_branches() {
        use chrono::TimeZone;
        let now = Utc.with_ymd_and_hms(2026, 9, 13, 12, 0, 0).unwrap();
        assert_eq!(
            resolve_since(Some("12h"), now),
            Utc.with_ymd_and_hms(2026, 9, 13, 0, 0, 0)
                .unwrap()
                .fixed_offset()
        );
        assert_eq!(
            resolve_since(Some("2026-09-10T08:30:00+00:00"), now).to_rfc3339(),
            "2026-09-10T08:30:00+00:00"
        );
        // Naive reads as UTC; date-only is midnight.
        assert_eq!(
            resolve_since(Some("2026-09-10"), now).to_rfc3339(),
            "2026-09-10T00:00:00+00:00"
        );
        // Garbage and None fall back to 24h.
        assert_eq!(
            resolve_since(Some("soon"), now),
            Utc.with_ymd_and_hms(2026, 9, 12, 12, 0, 0)
                .unwrap()
                .fixed_offset()
        );
        assert_eq!(resolve_since(None, now), resolve_since(Some("24h"), now));
    }
}
