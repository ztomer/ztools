//! ONE way to use a model on the Osaurus server: load it, run, release it.
//!
//! THE RULING (owner, 2026-11-11, ZT.OSAURUSLOAD): "osaurus should be told to
//! load models, and it's not done automatically" and "we shouldn't have a
//! model loaded when it's idle  it's a lot of memory". So every ztools caller
//! of `osaurus_url` goes through [`with_model`], which loads the model
//! EXPLICITLY, waits until `/health` shows it resident, current and accepting,
//! runs the caller, and then releases it on every exit path -- an error, a
//! refusal and a panic included -- until `/health` shows it no longer
//! resident.
//!
//! WHAT OSAURUS 1.0.1 OFFERS, measured 2026-11-11 (its route table, the
//! source at tag 1.0.1, and a live run on this Mac):
//!
//! - LOAD has no route of its own. The model is loaded by the first inference
//!   request that names it (`power.jitLoad = true`), so the load here is a
//!   one-token completion, timed: that call IS the cold start, which is why
//!   [`Held::load_seconds`] is what the eval records as `cold_start_seconds`.
//!   (`POST /agents/{id}/warm` loads too, but it is owner-only over the Secure
//!   Channel, so a loopback HTTP client cannot use it.)
//! - UNLOAD has no route at all. What frees the weights is the server's idle
//!   policy (`modelIdleResidencyPolicy` in `~/.osaurus/config/server.json`,
//!   30 s here), which `/health` publishes per resident model as
//!   `idle_seconds_remaining`, plus strict one-model eviction when another
//!   model loads. So release is a WAIT on that timer, bounded by what the
//!   server itself says, and it is an ERROR when the server says no unload is
//!   scheduled (`idle_seconds_remaining: null`, the "Keep Model Loaded"
//!   setting) or when the deadline passes with the model still resident: a
//!   model left in memory is the fault this module exists to catch, so it is
//!   never swallowed.
//! - `memorySafety.keepModelInMemory` (the runtime settings) does NOT keep a
//!   model loaded after use: it pins weights while loaded, and the idle timer
//!   is unchanged (Osaurus's own description of the toggle, and measured).
//!
//! Every load and release is held under the machine-wide GPU lock that
//! `tools/osaurus_one.sh` takes, so a summary cannot land in the middle of a
//! measurement. A caller that already holds it (the eval) inherits it.

use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::ztools::eval::{DEFAULT_MAX_IDLE_SECS, GpuLockGuard};

/// How long the steps may take. Production values are [`Timing::default`];
/// a test shortens them so a fake server answers in milliseconds.
#[derive(Debug, Clone, Copy)]
pub struct Timing {
    /// Waiting for another session's GPU lock.
    pub lock_wait: Duration,
    /// The load request itself: a cold start of a large model is minutes.
    pub load_request: Duration,
    /// After the load request answers, until `/health` shows it current.
    pub settle: Duration,
    /// Added to the server's own `idle_seconds_remaining` before release is
    /// declared failed.
    pub release_grace: Duration,
    /// The longest release will ever wait, whatever the server says.
    pub release_cap: Duration,
    /// First poll interval; it doubles up to [`Timing::poll_max`].
    pub poll_min: Duration,
    /// Largest poll interval.
    pub poll_max: Duration,
}

impl Default for Timing {
    fn default() -> Self {
        Self {
            lock_wait: Duration::from_secs(300),
            load_request: Duration::from_mins(15),
            settle: Duration::from_secs(60),
            release_grace: Duration::from_secs(30),
            release_cap: Duration::from_mins(15),
            poll_min: Duration::from_millis(50),
            poll_max: Duration::from_secs(1),
        }
    }
}

/// One resident model, as `/health` reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resident {
    pub name: String,
    pub is_current: bool,
    /// `None` is the server saying no idle unload is scheduled.
    pub idle_seconds_remaining: Option<u64>,
}

/// The part of `GET /health` this module reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Health {
    pub resident: Vec<Resident>,
    pub accepting: bool,
}

impl Health {
    /// The resident row for `model`, if it is resident.
    #[must_use]
    pub fn find(&self, model: &str) -> Option<&Resident> {
        self.resident.iter().find(|r| same_model(&r.name, model))
    }
}

/// Whether a `/health` name and a request's model id are the same model. The
/// server lists the id the request used; a repo-qualified id
/// (`OsaurusAI/Raptor-...`) is compared by its last component, case-folded.
#[must_use]
pub fn same_model(listed: &str, requested: &str) -> bool {
    let tail = |s: &str| s.rsplit('/').next().unwrap_or(s).to_ascii_lowercase();
    tail(listed) == tail(requested)
}

/// Parse a `/health` body.
///
/// # Errors
///
/// When it is not JSON or carries no `resident_models` array: that is not an
/// Osaurus server, and guessing what it holds would defeat the point.
pub fn parse_health(body: &str) -> Result<Health> {
    let v: Value = serde_json::from_str(body).context("/health is not JSON")?;
    let rows = v
        .get("resident_models")
        .and_then(Value::as_array)
        .context("/health has no resident_models array: not an Osaurus server")?;
    let resident = rows
        .iter()
        .filter_map(|r| {
            Some(Resident {
                name: r.get("name")?.as_str()?.to_string(),
                is_current: r
                    .get("is_current")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                idle_seconds_remaining: r.get("idle_seconds_remaining").and_then(Value::as_u64),
            })
        })
        .collect();
    let accepting = v
        .pointer("/batch_diagnostics/accepting_requests")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    Ok(Health {
        resident,
        accepting,
    })
}

/// Read `GET {base_url}/health`.
///
/// # Errors
///
/// When the server does not answer 2xx or the body is not Osaurus's.
pub fn health(base_url: &str) -> Result<Health> {
    let url = format!("{}/health", base_url.trim_end_matches('/'));
    let resp = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()?
        .get(&url)
        .send()
        .with_context(|| format!("GET {url}"))?;
    if !resp.status().is_success() {
        bail!("GET {url} answered {}", resp.status());
    }
    parse_health(&resp.text()?)
}

/// A model loaded on the server, held under the GPU lock until released.
pub struct Held {
    base_url: String,
    model: String,
    timing: Timing,
    load_seconds: f64,
    released: bool,
    // Dropped LAST, after the release in `Drop::drop` has run.
    _gpu: GpuLockGuard,
}

impl Held {
    /// How long the load request took: the model's cold start.
    #[must_use]
    pub const fn load_seconds(&self) -> f64 {
        self.load_seconds
    }

    /// The model this holds.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Release the model and wait until `/health` shows it gone.
    ///
    /// # Errors
    ///
    /// As [`wait_released`].
    pub fn release(mut self) -> Result<()> {
        self.released = true;
        wait_released(&self.base_url, &self.model, &self.timing)
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        // The exit path nobody called `release` on: an early `?` or a panic.
        // It still waits the model out; a failure can only be reported here.
        if let Err(e) = wait_released(&self.base_url, &self.model, &self.timing) {
            eprintln!("✗ {e:#}");
        }
    }
}

/// Load `model`, and wait until `/health` shows it resident, current and
/// accepting requests.
///
/// # Errors
///
/// When the GPU lock is held by another session past [`Timing::lock_wait`],
/// when the load request fails, and when the model does not show as resident
/// within [`Timing::settle`]. A failed load is still released before the error
/// returns: a half-finished load can leave weights behind.
pub fn load(base_url: &str, model: &str, timing: Timing) -> Result<Held> {
    let gpu = take_gpu_lock(&format!("ztools {model} (load, run, release)"), timing)?;
    let mut held = Held {
        base_url: base_url.to_string(),
        model: model.to_string(),
        timing,
        load_seconds: 0.0,
        released: false,
        _gpu: gpu,
    };
    let started = Instant::now();
    request_load(base_url, model, timing.load_request)?;
    held.load_seconds = started.elapsed().as_secs_f64();
    wait_loaded(base_url, model, &timing)?;
    Ok(held)
}

/// The machine-wide GPU lock, or the reason it is unavailable. The ONE place
/// ztools takes it for the server: [`load`] and [`locked`] both come here.
fn take_gpu_lock(label: &str, timing: Timing) -> Result<GpuLockGuard> {
    GpuLockGuard::acquire(
        label,
        timing.lock_wait,
        Duration::from_secs(DEFAULT_MAX_IDLE_SECS),
    )
    .map_err(|e| anyhow::anyhow!("GPU lock unavailable: {e}"))
}

/// Run `body` -- a request to the model server that loads no model, such as
/// the roster or the embeddings -- while holding the GPU lock (ZT.TWGPULOCK).
///
/// Every request ztools makes to the server is made under the lock, so a
/// request cannot land in the middle of another session's measurement. A
/// caller that already holds it (an eval, a [`Held`] model) inherits it, and
/// the lock stays with that holder.
///
/// # Errors
///
/// When the lock is held by another session past [`Timing::lock_wait`]; `body`
/// has not run.
pub fn locked<T>(label: &str, timing: Timing, body: impl FnOnce() -> T) -> Result<T> {
    let _gpu = take_gpu_lock(label, timing)?;
    Ok(body())
}

/// Load `model`, run `body` with it, release it: the one seam.
///
/// The body's error wins over a release error, which is still printed; a
/// release error after a successful body is the run's error, because the
/// model left in memory is exactly what this seam exists to prevent.
///
/// # Errors
///
/// From [`load`], from `body`, and from the release.
pub fn with_model<T>(
    base_url: &str,
    model: &str,
    timing: Timing,
    body: impl FnOnce(&Held) -> Result<T>,
) -> Result<T> {
    let Ran { out, released } = with_model_kept(base_url, model, timing, body)?;
    match (out, released) {
        (Ok(v), Ok(())) => Ok(v),
        (Ok(_), Err(e)) => Err(e),
        (Err(e), released) => {
            if let Err(r) = released {
                eprintln!("✗ {r:#}");
            }
            Err(e)
        }
    }
}

/// What [`with_model_kept`]'s body produced, and the release after it, apart.
pub struct Ran<T> {
    /// The body's result.
    pub out: Result<T>,
    /// The release: an error is a model still resident after the body.
    pub released: Result<()>,
}

/// [`with_model`] for a caller whose body's result is worth keeping when the
/// release fails.
///
/// ZT.EVALRESIDENT: a finished measurement is still a
/// measurement when the model outstays it, so the caller gets both and says
/// the release error rather than losing the result to it.
///
/// # Errors
///
/// From [`load`] only: the body has not run.
pub fn with_model_kept<T>(
    base_url: &str,
    model: &str,
    timing: Timing,
    body: impl FnOnce(&Held) -> Result<T>,
) -> Result<Ran<T>> {
    let held = load(base_url, model, timing)?;
    let out = body(&held);
    let released = held.release();
    Ok(Ran { out, released })
}

/// The load request: a one-token completion naming the model.
fn request_load(base_url: &str, model: &str, timeout: Duration) -> Result<()> {
    let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));
    let payload = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "ready"}],
        "max_tokens": 1,
        "stream": false,
    });
    let resp = reqwest::blocking::Client::builder()
        .timeout(timeout)
        .build()?
        .post(&url)
        .json(&payload)
        .send()
        .with_context(|| format!("loading {model}: POST {url}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().unwrap_or_default();
        bail!(
            "loading {model}: the server answered {status}: {}",
            text.chars().take(300).collect::<String>()
        );
    }
    Ok(())
}

fn next_poll(current: Duration, timing: &Timing) -> Duration {
    (current * 2).min(timing.poll_max)
}

fn wait_loaded(base_url: &str, model: &str, timing: &Timing) -> Result<()> {
    let deadline = Instant::now() + timing.settle;
    let mut poll = timing.poll_min;
    loop {
        let h = health(base_url)?;
        if let Some(r) = h.find(model)
            && r.is_current
            && h.accepting
        {
            return Ok(());
        }
        if Instant::now() >= deadline {
            bail!(
                "loading {model}: the load request answered but /health does not show it \
                 resident, current and accepting after {:?} (resident: {})",
                timing.settle,
                names(&h)
            );
        }
        std::thread::sleep(poll);
        poll = next_poll(poll, timing);
    }
}

/// Wait until `/health` no longer lists `model` as resident.
///
/// # Errors
///
/// When `/health` cannot be read; when the server says no unload is scheduled
/// for the model (its idle policy keeps it loaded, and Osaurus 1.0.1 has no
/// unload route ztools could call instead); and when the model is still
/// resident past the server's own idle deadline plus [`Timing::release_grace`]
/// (never longer than [`Timing::release_cap`]).
pub fn wait_released(base_url: &str, model: &str, timing: &Timing) -> Result<()> {
    let started = Instant::now();
    let mut poll = timing.poll_min;
    loop {
        let h = health(base_url)?;
        let Some(row) = h.find(model) else {
            return Ok(());
        };
        let Some(remaining) = row.idle_seconds_remaining else {
            // In flight is not "kept": its timer is armed when the work drains.
            if h.resident
                .iter()
                .all(|r| r.idle_seconds_remaining.is_none())
                && started.elapsed() >= timing.release_grace
            {
                bail!(
                    "{model} is still resident on {base_url} and the server schedules no idle \
                     unload for it (idle_seconds_remaining is null: Osaurus's 'Keep Model \
                     Loaded' residency setting is on). Osaurus 1.0.1 has no unload route, so \
                     ztools cannot release it"
                );
            }
            std::thread::sleep(poll);
            poll = next_poll(poll, timing);
            continue;
        };
        let allowed =
            (Duration::from_secs(remaining) + timing.release_grace).min(timing.release_cap);
        if started.elapsed() >= allowed {
            bail!(
                "{model} is still resident on {base_url} {:.0}s after its run ended \
                 (the server reports {remaining}s to its idle unload)",
                started.elapsed().as_secs_f64()
            );
        }
        std::thread::sleep(poll);
        poll = next_poll(poll, timing);
    }
}

fn names(h: &Health) -> String {
    if h.resident.is_empty() {
        "none".to_string()
    } else {
        h.resident
            .iter()
            .map(|r| r.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

// Unconditional, NOT `#[cfg(test)]`, for the reason `test_env` is: the
// integration tests under `rust/tests/` drive the binary against it and see
// only the library's public API.
pub mod fake;
mod session;
pub use session::Session;

#[cfg(test)]
#[path = "residency_tests.rs"]
mod tests;
