//! token-pool's free pools, reached through the pool's own MCP server.
//!
//! The summarizer's chain once ended in two services on this Mac: the Osaurus
//! server and the on-device `fm`. On 2026-11-07 both were down at once (the
//! server was not running; `fm`'s terms were not agreed), and every model in
//! the chain failed. A tier that is up when those are down has to live off
//! this machine: token-pool (the owner's checkout of it) spends free remote pools
//! (`OpenCode`'s Zen models, agy Flash) through each pool's own first-party
//! client, and its one entry point is an MCP stdio server, `pool-mcp`. There
//! is no other non-interactive door, so ztools speaks MCP to it: rmcp (OW.60),
//! one current-thread runtime per call, nothing async outside this module.
//!
//! `conf/twitter.toml [fallback] pool` names the command that starts the
//! server; with it set, the chain's [`POOL_MODEL`] is answered here.

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use rmcp::ServiceExt;
use rmcp::model::CallToolRequestParams;
use serde_json::{Map, Value, json};

/// The chain's name for the pool tier.
pub const POOL_MODEL: &str = "pool";

/// The pool's task class for a summary: read-only bulk work.
pub const SUMMARY_CLASS: &str = "bulk_read";

/// What the pool answered, and which of its tiers answered it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolAnswer {
    pub text: String,
    pub tier: String,
}

/// The `ask` tool's arguments. A summary is never `private`: the timeline is
/// public posts, and private work is held to the pool's zero-retention models.
#[must_use]
pub fn ask_arguments(task_class: &str, prompt: &str) -> Map<String, Value> {
    let Value::Object(map) = json!({
        "task_class": task_class,
        "prompt": prompt,
        "private": false,
    }) else {
        unreachable!("a json! object literal is an object")
    };
    map
}

/// Read a `tools/call` result (as JSON) into an answer.
///
/// The pool's answer is `{status, text, tier, detail, attempts}`, carried as
/// structured content or as JSON text. `ok` and `unverified` (no proof was
/// asked for: the summarizer's own quality gate judges the text) with text
/// are answers; anything else is the pool's stated refusal.
///
/// # Errors
///
/// When the call itself failed (`isError`), the result carries no readable
/// answer, or the pool refused or escalated, naming its detail and the tiers
/// it tried.
pub fn read_answer(result: &Value) -> Result<PoolAnswer> {
    let texts: Vec<&str> = result
        .get("content")
        .and_then(Value::as_array)
        .map(|blocks| {
            blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(Value::as_str))
                .collect()
        })
        .unwrap_or_default();
    if result.get("isError").and_then(Value::as_bool) == Some(true) {
        bail!("the pool's ask failed: {}", texts.join(" "));
    }
    let answer = result
        .get("structuredContent")
        .filter(|v| v.is_object())
        .cloned()
        .or_else(|| texts.first().and_then(|t| serde_json::from_str(t).ok()))
        .ok_or_else(|| anyhow!("the pool's answer carried no readable result"))?;
    // A server may wrap a non-object return as `{"result": ...}`.
    let answer = match answer.get("result") {
        Some(inner) if inner.get("status").is_some() => inner.clone(),
        _ => answer,
    };
    let field = |k: &str| answer.get(k).and_then(Value::as_str).unwrap_or_default();
    let status = field("status");
    let text = field("text").trim();
    if matches!(status, "ok" | "unverified") && !text.is_empty() {
        return Ok(PoolAnswer {
            text: text.to_string(),
            tier: field("tier").to_string(),
        });
    }
    let tried: Vec<String> = answer
        .get("attempts")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .map(|t| {
                    let s = |k: &str| t.get(k).and_then(Value::as_str).unwrap_or_default();
                    format!("{} {}", s("tier"), s("status"))
                })
                .collect()
        })
        .unwrap_or_default();
    let status = if status.is_empty() {
        "no status"
    } else {
        status
    };
    bail!(
        "the pool answered {status} with no usable text: {}{}",
        field("detail"),
        if tried.is_empty() {
            String::new()
        } else {
            format!(" (tried: {})", tried.join(", "))
        }
    )
}

/// One answer from the pool: start `command` (the pool's MCP server), call
/// `ask` once, stop the server.
///
/// # Errors
///
/// The command is empty or cannot start, the MCP session cannot open, the
/// call does not finish within `timeout`, or the answer is a refusal
/// ([`read_answer`]). The server's stderr is quoted when it said anything.
pub fn ask(
    command: &[String],
    task_class: &str,
    prompt: &str,
    timeout: Duration,
) -> Result<PoolAnswer> {
    let Some((program, args)) = command.split_first() else {
        bail!("no pool command configured");
    };
    let stderr_file = tempfile::NamedTempFile::new().context("pool stderr file")?;
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("pool runtime")?;
    let outcome = rt.block_on(async {
        let mut child = tokio::process::Command::new(crate::manifest::expand_tilde(program))
            .args(args.iter().map(|a| crate::manifest::expand_tilde(a)))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(stderr_file.reopen().context("pool stderr")?)
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("cannot start the pool server `{}`", command.join(" ")))?;
        let result = call(&mut child, task_class, prompt, timeout).await;
        let _ = child.kill().await;
        result
    });
    outcome.map_err(|e| with_stderr(e, stderr_file.path()))
}

async fn call(
    child: &mut tokio::process::Child,
    task_class: &str,
    prompt: &str,
    timeout: Duration,
) -> Result<PoolAnswer> {
    let (Some(stdout), Some(stdin)) = (child.stdout.take(), child.stdin.take()) else {
        bail!("the pool server's pipes were not opened");
    };
    let work = async {
        let client = ()
            .serve((stdout, stdin))
            .await
            .map_err(|e| anyhow!("the pool's MCP session did not open: {e}"))?;
        let params =
            CallToolRequestParams::new("ask").with_arguments(ask_arguments(task_class, prompt));
        let result = client
            .call_tool(params)
            .await
            .map_err(|e| anyhow!("the pool's ask call failed: {e}"));
        let _ = client.cancel().await;
        serde_json::to_value(result?).context("the pool's answer")
    };
    let value = tokio::time::timeout(timeout, work)
        .await
        .map_err(|_| anyhow!("the pool did not answer within {}s", timeout.as_secs()))??;
    read_answer(&value)
}

/// `e`, with the server's last stderr lines when it wrote any.
fn with_stderr(e: anyhow::Error, path: &Path) -> anyhow::Error {
    let said = std::fs::read_to_string(path).unwrap_or_default();
    let tail: Vec<&str> = said.lines().rev().take(3).collect();
    if tail.is_empty() {
        return e;
    }
    let tail: Vec<&str> = tail.into_iter().rev().collect();
    anyhow!("{e:#}; the pool server said: {}", tail.join(" / "))
}

#[cfg(test)]
#[path = "pool_tests.rs"]
pub(crate) mod tests;
