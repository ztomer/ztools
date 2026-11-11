//! A loopback fake of the Osaurus server's residency, for tests.
//!
//! It behaves as Osaurus 1.0.1 was measured to (`super`'s header): any
//! completion naming a model makes that model the one resident model (strict
//! one-model eviction); `/health` lists it with `idle_seconds_remaining: 0`
//! once, and the next `/health` after that finds the idle timer fired and the
//! model gone. Every request is recorded, so a test can say what was loaded,
//! in what order, and whether the run ended with nothing resident.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::thread;

/// One request the fake saw.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Call {
    /// A one-token completion: the seam's load.
    Load(String),
    /// Any other completion: the caller's work.
    Chat(String),
    /// `GET /health`, with what it answered as resident.
    Health(Vec<String>),
    /// Anything else, by path.
    Other(String),
}

/// A completion's body as a function of its request body.
pub type Answerer = Arc<dyn Fn(&str) -> String + Send + Sync>;

/// What the fake answers besides residency.
#[derive(Clone, Default)]
pub struct Spec {
    /// Every completion's body.
    pub chat: String,
    /// `/v1/models`'s body.
    pub models: String,
    /// `/v1/embeddings`'s body.
    pub embeddings: String,
    /// The server schedules no idle unload ("Keep Model Loaded").
    pub keep_loaded: bool,
    /// When set, a completion's body is this function of the request body,
    /// for a test whose answer depends on the prompt.
    pub answer_for: Option<Answerer>,
}

impl Spec {
    /// Completions answer `body`; an empty roster; one 2-d embedding.
    #[must_use]
    pub fn chat(body: &str) -> Self {
        Self {
            chat: body.to_string(),
            models: r#"{"data":[]}"#.to_string(),
            embeddings: r#"{"data": [{"embedding": [0.1, 0.2]}]}"#.to_string(),
            keep_loaded: false,
            answer_for: None,
        }
    }
}

#[derive(Default)]
struct State {
    resident: Option<String>,
    shown_since_use: bool,
    calls: Vec<Call>,
}

/// The running fake. Dropping it leaves the thread serving until the test
/// process ends, as every other stub in this crate does.
#[derive(Clone)]
pub struct FakeOsaurus {
    pub url: String,
    state: Arc<Mutex<State>>,
}

impl FakeOsaurus {
    /// Start a fake whose completions answer `chat_body` (an `OpenAI` response).
    #[must_use]
    pub fn start(chat_body: &str) -> Self {
        Self::serve(Spec::chat(chat_body))
    }

    /// As [`Self::start`], but a model whose idle unload is never scheduled
    /// (`idle_seconds_remaining: null`), as with "Keep Model Loaded" on.
    #[must_use]
    pub fn keeping_loaded(chat_body: &str) -> Self {
        Self::serve(Spec {
            keep_loaded: true,
            ..Spec::chat(chat_body)
        })
    }

    /// Start a fake answering as `spec` says.
    ///
    /// # Panics
    ///
    /// When no loopback port can be bound.
    #[must_use]
    pub fn serve(spec: Spec) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let state = Arc::new(Mutex::new(State::default()));
        let shared = Arc::clone(&state);
        thread::spawn(move || {
            for mut stream in listener.incoming().flatten() {
                let request = read_request(&mut stream);
                let reply = answer(&mut shared.lock().unwrap(), &request, &spec);
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
                    reply.len()
                );
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });
        Self {
            url: format!("http://127.0.0.1:{port}"),
            state,
        }
    }

    /// Every request so far, in order.
    ///
    /// # Panics
    ///
    /// When the serving thread panicked holding the state.
    #[must_use]
    pub fn calls(&self) -> Vec<Call> {
        self.state.lock().unwrap().calls.clone()
    }

    /// The models loaded, in order.
    #[must_use]
    pub fn loads(&self) -> Vec<String> {
        self.calls()
            .into_iter()
            .filter_map(|c| match c {
                Call::Load(m) => Some(m),
                _ => None,
            })
            .collect()
    }

    /// What the LAST `/health` answered as resident: `Some(vec![])` is a run
    /// that ended with nothing in memory and SAW it; `None` never asked.
    #[must_use]
    pub fn last_health(&self) -> Option<Vec<String>> {
        self.calls().into_iter().rev().find_map(|c| match c {
            Call::Health(r) => Some(r),
            _ => None,
        })
    }
}

fn read_request(stream: &mut std::net::TcpStream) -> String {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    while let Ok(n) = stream.read(&mut chunk) {
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
        let text = String::from_utf8_lossy(&buf);
        if let Some(head_end) = text.find("\r\n\r\n") {
            let length = text[..head_end]
                .lines()
                .find_map(|l| {
                    let (k, v) = l.split_once(':')?;
                    k.eq_ignore_ascii_case("content-length")
                        .then(|| v.trim().parse::<usize>().ok())?
                })
                .unwrap_or(0);
            if buf.len() >= head_end + 4 + length {
                break;
            }
        }
    }
    String::from_utf8_lossy(&buf).into_owned()
}

fn answer(s: &mut State, request: &str, spec: &Spec) -> String {
    let keep = spec.keep_loaded;
    let first = request.lines().next().unwrap_or_default().to_string();
    if first.starts_with("GET /health") {
        let shown: Vec<String> = s.resident.iter().cloned().collect();
        let rows: Vec<String> = shown
            .iter()
            .map(|m| {
                let remaining = if keep { "null" } else { "0" };
                format!(
                    r#"{{"name":"{m}","is_current":true,"inflight":0,"idle_seconds_remaining":{remaining}}}"#
                )
            })
            .collect();
        s.calls.push(Call::Health(shown));
        if !keep && s.resident.is_some() {
            if s.shown_since_use {
                s.resident = None;
            } else {
                s.shown_since_use = true;
            }
        }
        return format!(
            r#"{{"status":"healthy","batch_diagnostics":{{"accepting_requests":true}},"resident_models":[{}]}}"#,
            rows.join(",")
        );
    }
    let body = request.split("\r\n\r\n").nth(1).unwrap_or_default();
    if first.contains("/v1/chat/completions") {
        let v: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
        let model = v["model"].as_str().unwrap_or_default().to_string();
        let is_load = v["max_tokens"].as_u64() == Some(1);
        s.calls.push(if is_load {
            Call::Load(model.clone())
        } else {
            Call::Chat(model.clone())
        });
        s.resident = Some(model);
        s.shown_since_use = false;
        return spec
            .answer_for
            .as_ref()
            .map_or_else(|| spec.chat.clone(), |f| f(body));
    }
    let path = first
        .split_whitespace()
        .nth(1)
        .unwrap_or_default()
        .to_string();
    s.calls.push(Call::Other(path.clone()));
    if path.starts_with("/v1/models") {
        return spec.models.clone();
    }
    if path.starts_with("/v1/embeddings") {
        return spec.embeddings.clone();
    }
    spec.chat.clone()
}
