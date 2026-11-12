//! How refcheck talks to Osaurus: HTTP reads, and the one door to the CLI.
//!
//! The `osaurus` CLI probes the server for 0.5 s. When the server is busy and
//! misses that, the CLI decides the app is not running and has `LaunchServices`
//! open a SECOND Osaurus (`--launched-by-cli`); the 30-minute `models-refcheck`
//! did exactly that (ZT-4, 2026-11-12). So nothing here runs the CLI to READ:
//! the export and the model list are HTTP, and the verbs with no HTTP form
//! (`config plan`, `config apply`) go through [`run`], which refuses unless
//! `/health` answers within [`HEALTH_TIMEOUT`], ten times the CLI's own probe.
//!
//! Every request in this file is made under the GPU lock like every other ztools
//! request to the server (ZT.TWGPULOCK): each public function takes it.

use super::CmdOut;
use crate::ztools::residency::{self, Timing};
use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::path::Path;
use std::process::Command;
use std::time::Duration;

/// How long `/health` may take before the server counts as not answering.
const HEALTH_TIMEOUT: Duration = Duration::from_secs(5);
const READ_TIMEOUT: Duration = Duration::from_secs(30);

fn get(url: &str, timeout: Duration) -> Result<reqwest::blocking::Response> {
    let resp = reqwest::blocking::Client::builder()
        .timeout(timeout)
        .build()?
        .get(url)
        .send()
        .with_context(|| format!("GET {url}"))?;
    if !resp.status().is_success() {
        bail!("GET {url} answered {}", resp.status());
    }
    Ok(resp)
}

/// Run `body` under the GPU lock; a lock another session holds past the wait is
/// an error, and `body` has not run.
fn under_lock<T>(label: &str, body: impl FnOnce() -> Result<T>) -> Result<T> {
    residency::locked(label, Timing::default(), body)?
}

fn url(base: &str, path: &str) -> String {
    format!("{}{path}", base.trim_end_matches('/'))
}

/// The Osaurus config export: `GET /admin/config/export?format=json` answers
/// `{"json": "<document>"}`, the document as text.
///
/// # Errors
///
/// When the server does not answer or the document is not JSON.
pub fn export(base: &str) -> Result<Value> {
    under_lock("ztools models refcheck: export", || export_now(base))
}

fn export_now(base: &str) -> Result<Value> {
    let u = url(base, "/admin/config/export?format=json");
    let wrapper: Value = get(&u, READ_TIMEOUT)?
        .json()
        .with_context(|| format!("{u} is not JSON"))?;
    let text = wrapper
        .get("json")
        .and_then(Value::as_str)
        .with_context(|| format!("{u} has no \"json\" field"))?;
    serde_json::from_str(text).context("osaurus config export is not JSON")
}

/// The ids of the installed models: `GET /v1/models`.
///
/// # Errors
///
/// When the server does not answer or the list has no `data` array.
pub fn models(base: &str) -> Result<Vec<String>> {
    under_lock("ztools models refcheck: model list", || models_now(base))
}

fn models_now(base: &str) -> Result<Vec<String>> {
    let u = url(base, "/v1/models");
    let body: Value = get(&u, READ_TIMEOUT)?
        .json()
        .with_context(|| format!("{u} is not JSON"))?;
    let data = body
        .get("data")
        .and_then(Value::as_array)
        .with_context(|| format!("{u} has no \"data\" list"))?;
    Ok(data
        .iter()
        .filter_map(|m| m.get("id").and_then(Value::as_str))
        .map(str::to_string)
        .collect())
}

/// Refuse, with the reason, unless `GET /health` answers 2xx in time. Taken only
/// inside [`run`], which holds the lock.
fn require_healthy(base: &str) -> Result<()> {
    get(&url(base, "/health"), HEALTH_TIMEOUT)
        .map(drop)
        .map_err(|e| {
            e.context(format!(
                "not running the osaurus CLI: the server at {base} does not answer /health within \
             {} s, and the CLI would open a second Osaurus app",
                HEALTH_TIMEOUT.as_secs()
            ))
        })
}

/// THE door to the `osaurus` CLI: check `/health`, then run `program args [file]`.
///
/// # Errors
///
/// When `/health` does not answer (the CLI is not run), or the program cannot be run.
pub fn run(base: &str, program: &Path, args: &[&str], file: Option<&Path>) -> Result<CmdOut> {
    under_lock("ztools models refcheck: osaurus CLI", || {
        require_healthy(base)?;
        run_now(program, args, file)
    })
}

fn run_now(program: &Path, args: &[&str], file: Option<&Path>) -> Result<CmdOut> {
    let mut cmd = Command::new(program);
    cmd.args(args);
    if let Some(f) = file {
        cmd.arg(f);
    }
    let out = cmd
        .output()
        .with_context(|| format!("cannot run {}", program.display()))?;
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    Ok(CmdOut {
        code: out.status.code().unwrap_or(-1),
        text,
    })
}
