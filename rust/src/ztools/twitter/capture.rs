//! Timeline network-capture machinery for the native collect driver.
//!
//! Split from [`super::native`]: everything here feeds observations to the
//! scroll loop — response recording on the reader thread, body drain + decode
//! on the driver thread. `getResponseBody` is NEVER called from inside the
//! event handler (reader-thread deadlock risk); the handler only records
//! `(requestId, url, encoding)` triples.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use anyhow::{Context, Result};
use chrono::{DateTime, FixedOffset};

use super::browser_parse;
use super::collect::parse_created_at;
use super::endpoints::EndpointMarkers;
use super::Tweet;

/// Decode a captured response body.
///
/// `Network.getResponseBody` returns raw bytes — unlike Playwright, nothing
/// decodes transparently. The declared `content-encoding` is a HINT, not the
/// truth: observed 2026-09-13, x.com timeline bodies arrive brotli-compressed
/// while `responseReceived` headers claim `gzip`. So this sniffs first (JSON
/// passes through, gzip magic dispatches) and only then trusts the header,
/// with a last-resort brotli attempt before giving up. A wrong guess is safe:
/// only bytes that parse as JSON downstream ever become tweets.
///
/// # Errors
///
/// When no decoding yields bytes (the named encoding, or `undecodable` when
/// even the fallbacks fail).
pub(crate) fn decode_body(bytes: &[u8], encoding: Option<&str>) -> Result<Vec<u8>> {
    use std::io::Read as _;

    fn gunzip(bytes: &[u8]) -> Result<Vec<u8>> {
        let mut dec = flate2::read::GzDecoder::new(bytes);
        let mut out = Vec::new();
        dec.read_to_end(&mut out).context("gzip decode failed")?;
        Ok(out)
    }
    fn inflate(bytes: &[u8]) -> Result<Vec<u8>> {
        let mut dec = flate2::read::DeflateDecoder::new(bytes);
        let mut out = Vec::new();
        dec.read_to_end(&mut out).context("deflate decode failed")?;
        Ok(out)
    }
    fn unbrotli(bytes: &[u8]) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        brotli::BrotliDecompress(&mut &bytes[..], &mut out).context("brotli decode failed")?;
        Ok(out)
    }

    // Already plain (or transparently decoded upstream): JSON parses.
    if serde_json::from_slice::<serde_json::Value>(bytes).is_ok() {
        return Ok(bytes.to_vec());
    }
    // Magic beats metadata.
    if bytes.starts_with(&[0x1f, 0x8b]) {
        return gunzip(bytes);
    }
    let declared = encoding.map(str::to_lowercase);
    match declared.as_deref() {
        Some("br") => return unbrotli(bytes),
        Some("deflate") => return inflate(bytes),
        Some("gzip") => {
            if let Ok(out) = gunzip(bytes) {
                return Ok(out);
            }
        }
        Some("") | None => {}
        Some(other) => anyhow::bail!("unknown content-encoding: {other}"),
    }
    // Last resort: x.com serves `br` under unreliable headers.
    unbrotli(bytes).with_context(|| {
        format!(
            "undecodable body (declared: {})",
            declared.as_deref().unwrap_or("none")
        )
    })
}

/// One captured timeline response: its protocol request id plus encoding.
/// The URL served its purpose at filter time and is not retained.
pub(crate) struct Captured {
    encoding: Option<String>,
    request_id: String,
}

/// Shared network-capture state, fed by the reader-thread handler (which must
/// stay cheap: it only records; bodies are drained from the driver thread —
/// `getResponseBody` from inside the handler risks the reader thread).
pub(crate) struct CaptureState {
    /// `requestId → url`, from `requestWillBeSent` (Juggler's
    /// `responseReceived` carries no URL — see Phase 0 S1 findings).
    urls: HashMap<String, String>,
    /// Timeline responses seen but not yet drained.
    pending: Vec<Captured>,
    /// Whether any timeline response (either feed) was seen at all. Without
    /// it, a run with no timeline traffic is an empty run, not a wrong-feed
    /// run — the two need different messages.
    saw_timeline: bool,
    /// Whether any drained-or-seen response came from the Following endpoint.
    /// A run with timeline traffic but no Following traffic collected the
    /// wrong feed (the tab switch silently fell back to For You) and must
    /// fail loudly rather than produce a plausible-but-wrong set.
    saw_following: bool,
    /// Endpoint markers from data, fixed at construction: the reader thread
    /// must never do file I/O.
    markers: EndpointMarkers,
}

impl CaptureState {
    pub(crate) fn new(markers: EndpointMarkers) -> Self {
        Self {
            urls: HashMap::new(),
            pending: Vec::new(),
            saw_timeline: false,
            saw_following: false,
            markers,
        }
    }
}

/// Whether the run watched the wrong feed: timeline traffic arrived but none
/// of it came from the Following endpoint. Read by the driver after the
/// scroll. A run with no timeline traffic at all returns `false` here — that
/// is an empty run, reported as such, not blamed on the feed.
pub(crate) fn collected_wrong_feed(state: &Mutex<CaptureState>) -> bool {
    let st = state.lock().unwrap();
    st.saw_timeline && !st.saw_following
}

pub(crate) fn header<'a>(params: &'a serde_json::Value, name: &str) -> Option<&'a str> {
    params.get("headers")?.as_array()?.iter().find_map(|h| {
        let n = h.get("name")?.as_str()?;
        if n.eq_ignore_ascii_case(name) {
            h.get("value")?.as_str()
        } else {
            None
        }
    })
}

/// Record a `requestWillBeSent` event: `requestId → url` linkage that
/// `responseReceived` lacks. Cheap by construction (two map inserts at most).
pub(crate) fn record_request(state: &Mutex<CaptureState>, request_id: &str, url: &str) {
    if request_id.is_empty() || url.is_empty() {
        return;
    }
    state
        .lock()
        .unwrap()
        .urls
        .insert(request_id.to_owned(), url.to_owned());
}

/// Record a `responseReceived` event, keeping timeline responses for the
/// driver thread to drain. Non-timeline URLs drop out here so the pending
/// queue never holds scroll noise.
///
/// # Panics
///
/// If the capture mutex is poisoned — only possible when a previous handler
/// invocation panicked, which this one cannot do.
pub(crate) fn record_response(state: &Mutex<CaptureState>, params: &serde_json::Value) {
    let rid = params
        .get("requestId")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if rid.is_empty() {
        return;
    }
    let mut st = state.lock().unwrap();
    if let Some(url) = st.urls.get(rid).cloned() {
        if st.markers.is_timeline_url(&url) {
            st.saw_timeline = true;
            if st.markers.is_following_url(&url) {
                st.saw_following = true;
            }
            let enc = header(params, "content-encoding").map(str::to_owned);
            st.pending.push(Captured {
                encoding: enc,
                request_id: rid.to_owned(),
            });
        }
    }
}

/// Drain newly-seen timeline responses into `tweets`, returning how many
/// tweets arrived. Evicted or undecodable bodies are skipped, never fatal —
/// a lost batch means fewer tweets, not wrong ones.
pub(crate) fn drain_responses(
    frame: &camoufox::api::MainFrame,
    state: &Mutex<CaptureState>,
    seen: &mut HashSet<String>,
    tweets: &mut Vec<Tweet>,
    oldest: &mut Option<DateTime<FixedOffset>>,
) {
    let pending: Vec<Captured> = {
        let mut st = state.lock().unwrap();
        std::mem::take(&mut st.pending)
    };
    for cap in pending {
        if !seen.insert(cap.request_id.clone()) {
            continue;
        }
        let Ok((raw, evicted)) = frame.get_response_body(&cap.request_id) else {
            continue;
        };
        if evicted || raw.is_empty() {
            continue;
        }
        let Ok(body) = decode_body(&raw, cap.encoding.as_deref()) else {
            continue;
        };
        let Ok(json) = serde_json::from_slice::<serde_json::Value>(&body) else {
            continue;
        };
        for t in browser_parse::parse_tweets_from_response(&json) {
            if let Some(created) = parse_created_at(&t.created_at) {
                if oldest.is_none_or(|o| created < o) {
                    *oldest = Some(created);
                }
            }
            tweets.push(t);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ztools::twitter::endpoints::EndpointMarkers;
    use std::sync::Mutex;

    fn test_state() -> Mutex<CaptureState> {
        // Fixture markers, not production data: the shipped file's contents
        // are pinned by `shipped_config_loads` in endpoints.rs.
        Mutex::new(CaptureState::new(EndpointMarkers {
            timeline: vec!["HomeTimeline".to_string(), "HomeLatestTimeline".to_string()],
            following: "HomeLatestTimeline".to_string(),
        }))
    }

    fn params(rid: &str) -> serde_json::Value {
        serde_json::json!({"requestId": rid, "headers": []})
    }

    fn feed_url(state: &Mutex<CaptureState>, rid: &str, url: &str) {
        record_request(state, rid, url);
        record_response(state, &params(rid));
    }

    #[test]
    fn following_endpoint_queues_and_is_the_right_feed() {
        let state = test_state();
        feed_url(
            &state,
            "r1",
            "https://x.com/api/graphql/HomeLatestTimeline?a=1",
        );
        assert!(!collected_wrong_feed(&state));
        assert_eq!(state.lock().unwrap().pending.len(), 1);
    }

    #[test]
    fn for_you_only_traffic_is_the_wrong_feed() {
        let state = test_state();
        feed_url(&state, "r1", "https://x.com/api/graphql/HomeTimeline?a=1");
        assert!(collected_wrong_feed(&state));
        assert_eq!(state.lock().unwrap().pending.len(), 1);
    }

    #[test]
    fn mixed_traffic_with_any_following_is_the_right_feed() {
        let state = test_state();
        feed_url(&state, "r1", "https://x.com/api/graphql/HomeTimeline?a=1");
        feed_url(
            &state,
            "r2",
            "https://x.com/api/graphql/HomeLatestTimeline?a=1",
        );
        assert!(!collected_wrong_feed(&state));
    }

    #[test]
    fn no_timeline_traffic_is_empty_not_wrong_feed() {
        // Zero traffic must not be blamed on the feed: the driver reports it
        // as "no tweets collected", which is a different failure.
        let state = test_state();
        assert!(!collected_wrong_feed(&state));
        feed_url(&state, "r1", "https://x.com/api/graphql/UserByScreenName");
        assert!(!collected_wrong_feed(&state));
        assert!(state.lock().unwrap().pending.is_empty());
    }
}
