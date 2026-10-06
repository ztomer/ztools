//! IPC types for daemon ↔ client communication.
//!
//! Newline-delimited JSON over Unix socket. One request → one response per connection.

use serde::{Deserialize, Serialize};
use serde_json::Value;

const fn default_timeout() -> u64 {
    30
}

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

/// A request from a CLI client to the daemon.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "method", content = "params")]
pub enum DaemonRequest {
    /// Ping the daemon. Returns instance count.
    Ping,

    /// Launch a new browser instance.
    Launch {
        #[serde(default)]
        headless: Option<bool>,
        #[serde(default)]
        executable: Option<String>,
    },

    /// List all running instances.
    List,

    /// Stop a browser instance.
    Stop { instance_id: String },

    /// Create a new page in an instance.
    NewPage { instance_id: String },

    /// Navigate a page to a URL.
    Navigate {
        instance_id: String,
        page_id: String,
        url: String,
        #[serde(default = "default_timeout")]
        timeout_secs: u64,
        /// Optional lifecycle event to wait for after the Page.navigate ack.
        /// Supported values: "load", "domcontentloaded".
        /// Absent (null/missing) means return after ack — existing behavior.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        wait_until: Option<String>,
    },

    /// Evaluate JavaScript on a page.
    Evaluate {
        instance_id: String,
        page_id: String,
        expression: String,
        #[serde(default = "default_timeout")]
        timeout_secs: u64,
    },

    /// Take a screenshot of a page.
    Screenshot {
        instance_id: String,
        page_id: String,
        #[serde(default)]
        format: Option<String>,
        #[serde(default)]
        quality: Option<u32>,
        #[serde(default)]
        path: Option<String>,
        #[serde(default = "default_timeout")]
        timeout_secs: u64,
    },

    /// Shut down the daemon and all instances.
    Shutdown,

    /// Export all cookies for a browser instance (including `HttpOnly`).
    Cookies { instance_id: String },
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

/// A response from the daemon to a CLI client.
#[derive(Debug, Serialize, Deserialize)]
pub struct DaemonResponse {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl DaemonResponse {
    /// Create a success response with data.
    pub const fn ok(data: Value) -> Self {
        Self {
            ok: true,
            error: None,
            data: Some(data),
        }
    }

    /// Create a success response with no data.
    pub const fn ok_empty() -> Self {
        Self {
            ok: true,
            error: None,
            data: None,
        }
    }

    /// Create an error response.
    pub fn err(message: impl Into<String>) -> Self {
        Self {
            ok: false,
            error: Some(message.into()),
            data: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// `DaemonRequest::Cookies` serialises to the expected JSON shape and
    /// deserialises back to the same variant (round-trip).
    #[test]
    fn cookies_request_serde_round_trip() {
        let req = DaemonRequest::Cookies {
            instance_id: "00000001".into(),
        };
        let serialized = serde_json::to_string(&req).expect("serialize");
        let deserialized: DaemonRequest = serde_json::from_str(&serialized).expect("deserialize");
        match deserialized {
            DaemonRequest::Cookies { instance_id } => {
                assert_eq!(instance_id, "00000001");
            }
            other => panic!("expected Cookies, got {other:?}"),
        }
    }

    /// A cookies response carrying an `HttpOnly` cookie round-trips through
    /// `DaemonResponse` without losing the `httpOnly` flag.
    #[test]
    fn cookies_response_preserves_http_only_flag() {
        let cookie_json = json!([{
            "name": "PHPSESSID",
            "value": "secret",
            "domain": "example.com",
            "path": "/",
            "expires": -1.0,
            "size": 13,
            "httpOnly": true,
            "secure": true,
            "session": true,
            "sameSite": "Strict"
        }]);
        let resp = DaemonResponse::ok(json!({ "cookies": cookie_json }));
        let serialized = serde_json::to_string(&resp).expect("serialize");
        let back: DaemonResponse = serde_json::from_str(&serialized).expect("deserialize");

        assert!(back.ok);
        let cookies = back
            .data
            .as_ref()
            .and_then(|d| d.get("cookies"))
            .and_then(|v| v.as_array())
            .expect("cookies array present");
        assert_eq!(cookies.len(), 1);
        assert_eq!(
            cookies[0]["httpOnly"], true,
            "httpOnly flag must survive round-trip"
        );
    }

    /// `DaemonResponse::err` round-trips correctly.
    #[test]
    fn error_response_round_trip() {
        let resp = DaemonResponse::err("instance not found");
        let serialized = serde_json::to_string(&resp).expect("serialize");
        let back: DaemonResponse = serde_json::from_str(&serialized).expect("deserialize");
        assert!(!back.ok);
        assert_eq!(back.error.as_deref(), Some("instance not found"));
        assert!(back.data.is_none());
    }

    // The two tests below cover the `navigate` feature (G3 `wait_until`, G4
    // `status_code`) at the wire level. They live here rather than in
    // `api::main_frame`'s test module because what they assert is the serde
    // shape of `DaemonRequest`/`DaemonResponse` — and because `cli` is behind
    // the `cli` feature, a test in `api` could not name these types under
    // default features at all (`cargo check --all-targets` failed with E0433).

    /// G3 TDD case 3d: IPC serde round-trip — `DaemonRequest::Navigate` with
    /// `wait_until` field serialises and deserialises correctly.
    #[test]
    fn navigate_ipc_wait_until_serde_round_trip() {
        // With wait_until present.
        let req = DaemonRequest::Navigate {
            instance_id: "00000001".into(),
            page_id: "p1".into(),
            url: "https://example.com".into(),
            timeout_secs: 30,
            wait_until: Some("load".into()),
        };
        let serialized = serde_json::to_string(&req).expect("serialize");
        let back: DaemonRequest = serde_json::from_str(&serialized).expect("deserialize");
        match back {
            DaemonRequest::Navigate { wait_until, .. } => {
                assert_eq!(wait_until.as_deref(), Some("load"));
            }
            other => panic!("expected Navigate, got {other:?}"),
        }

        // With wait_until absent — must not appear in serialised JSON.
        let req_no_wait = DaemonRequest::Navigate {
            instance_id: "00000001".into(),
            page_id: "p1".into(),
            url: "https://example.com".into(),
            timeout_secs: 30,
            wait_until: None,
        };
        let serialized_no_wait = serde_json::to_string(&req_no_wait).expect("serialize");
        assert!(
            !serialized_no_wait.contains("wait_until"),
            "wait_until must be absent from serialised JSON when None: {serialized_no_wait}"
        );
        let back_no_wait: DaemonRequest =
            serde_json::from_str(&serialized_no_wait).expect("deserialize");
        match back_no_wait {
            DaemonRequest::Navigate { wait_until, .. } => {
                assert!(wait_until.is_none(), "wait_until must deserialise to None");
            }
            other => panic!("expected Navigate, got {other:?}"),
        }

        // Legacy wire (no wait_until field at all) must deserialise to None.
        let legacy = r#"{"method":"Navigate","params":{"instance_id":"00000001","page_id":"p1","url":"https://example.com","timeout_secs":30}}"#;
        let back_legacy: DaemonRequest = serde_json::from_str(legacy).expect("deserialize legacy");
        match back_legacy {
            DaemonRequest::Navigate { wait_until, .. } => {
                assert!(
                    wait_until.is_none(),
                    "legacy wire (no wait_until) must deserialise to None"
                );
            }
            other => panic!("expected Navigate, got {other:?}"),
        }
    }

    /// G4 TDD case 3: navigate response IPC serde includes `status_code`.
    /// Absence is backward-compatible (legacy callers ignore unknown fields).
    #[test]
    fn navigate_response_serde_includes_status_code() {
        // Response WITH status_code.
        let resp = DaemonResponse::ok(json!({
            "navigation_id": "nav-1",
            "status_code": 200_u16,
        }));
        let serialized = serde_json::to_string(&resp).expect("serialize");
        assert!(
            serialized.contains("status_code"),
            "status_code must appear in JSON: {serialized}"
        );
        let back: DaemonResponse = serde_json::from_str(&serialized).expect("deserialize");
        assert_eq!(
            back.data
                .as_ref()
                .and_then(|d| d.get("status_code"))
                .and_then(serde_json::Value::as_u64),
            Some(200),
            "status_code round-trips"
        );
        assert_eq!(
            back.data
                .as_ref()
                .and_then(|d| d.get("navigation_id"))
                .and_then(|v| v.as_str()),
            Some("nav-1"),
            "navigation_id still present"
        );

        // Response WITHOUT status_code (legacy / null) must deserialise fine.
        let legacy = r#"{"ok":true,"data":{"navigation_id":"nav-2"}}"#;
        let legacy_back: DaemonResponse =
            serde_json::from_str(legacy).expect("deserialize legacy navigate response");
        assert!(legacy_back.ok);
        assert!(
            legacy_back
                .data
                .as_ref()
                .and_then(|d| d.get("status_code"))
                .is_none(),
            "legacy callers without status_code must deserialise fine (field absent is ok)"
        );
    }
}
