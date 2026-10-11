//! The on-device model, reached without a server (T1).
//!
//! Every model the summarizer's chain named was served by the one Osaurus
//! process, `foundation` included, so when that server was down the whole
//! chain failed at once (2026-10-03) -- a fallback in the primary's failure
//! domain is not a fallback. macOS 27 ships `fm` (`/usr/bin/fm`), Apple's own
//! command line for the on-device Foundation model: one process per answer,
//! no server to be down. `conf/twitter.toml [fallback] on_device` names the
//! binary; with it set, the chain's `foundation` is answered here.

use std::io::{Read, Write};
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

/// The chain's name for the on-device model.
pub const ON_DEVICE_MODEL: &str = "foundation";

/// `fm`'s exit status when the machine has not agreed to its terms.
const FM_LICENSE_NOT_AGREED: i32 = 69;

/// How often a running answer is checked for having finished.
const POLL: Duration = Duration::from_millis(50);

/// One answer from the on-device model: `fm respond` with `prompt` on stdin,
/// streamed off so stdout is the answer and nothing else.
///
/// # Errors
///
/// The prompt cannot fit the model's window (refused before spawning); `fm`
/// is missing, cannot be run, has not had its terms agreed to, exits non-zero,
/// answers nothing, or runs past `timeout_secs` (it is killed).
pub fn respond(fm: &Path, prompt: &str, timeout_secs: u64) -> Result<String> {
    if let Some(reason) =
        crate::ztools::eval::context_fit::context_refusal(ON_DEVICE_MODEL, prompt.len())
    {
        bail!("{reason}");
    }
    let mut child = Command::new(fm)
        .args(["respond", "--no-stream"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // Its own process group, so a timeout stops everything it started,
        // not only the process this side holds.
        .process_group(0)
        .spawn()
        .with_context(|| format!("cannot run the on-device model at {}", fm.display()))?;
    // Both pipes drain on their own threads while the answer runs, and the
    // prompt is fed on a third: written first, a child that answers before it
    // has read everything would block on a full stdout pipe while this side
    // blocked on a full stdin one. Dropping `stdin` closes the pipe, which is
    // how `fm` knows the prompt is complete.
    let out = drain(child.stdout.take());
    let err = drain(child.stderr.take());
    if let Some(mut stdin) = child.stdin.take() {
        let prompt = prompt.to_string();
        std::thread::spawn(move || {
            let _ = stdin.write_all(prompt.as_bytes());
        });
    }
    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if Instant::now() >= deadline {
            // The group's id is the child's pid (`process_group(0)`).
            let _ = Command::new("/bin/kill")
                .args(["-KILL", &format!("-{}", child.id())])
                .status();
            let _ = child.kill();
            let _ = child.wait();
            bail!("the on-device model ran past {timeout_secs}s and was stopped");
        }
        std::thread::sleep(POLL);
    };
    let out = out.join().unwrap_or_default();
    let err = err.join().unwrap_or_default();
    if status.code() == Some(FM_LICENSE_NOT_AGREED) {
        bail!(
            "the on-device model's terms are not agreed on this machine: run `sudo {} license`",
            fm.display()
        );
    }
    if !status.success() {
        bail!(
            "the on-device model exited {status}: {}",
            err.trim().lines().last().unwrap_or("no message")
        );
    }
    let answer = out.trim();
    if answer.is_empty() {
        bail!("the on-device model answered nothing");
    }
    Ok(answer.to_string())
}

/// Read a pipe to its end on a thread of its own.
fn drain(pipe: Option<impl Read + Send + 'static>) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let mut text = String::new();
        if let Some(mut pipe) = pipe {
            let _ = pipe.read_to_string(&mut text);
        }
        text
    })
}

#[cfg(test)]
#[path = "on_device_tests.rs"]
mod tests;
