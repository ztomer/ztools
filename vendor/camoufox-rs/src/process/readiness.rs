//! Readiness detection for the Camoufox browser process.
//!
//! After spawning, Camoufox writes `"Juggler listening to the pipe\n"` to
//! **stdout** once the Juggler engine is initialized and the pipe transport is
//! ready to accept commands.
//!
//! This module watches stdout for that sentinel string, with a configurable
//! timeout. If the process exits or the timeout expires before the sentinel
//! appears, an appropriate error is returned.
//!
//! # Why stdout, when `docs/PROTOCOL.md` says stderr
//!
//! PROTOCOL.md § 3 ("Startup Detection") and § 12 specify watching *stderr*.
//! The patched binary disagrees: launched with `-no-remote -headless -profile
//! <tmp> -juggler-pipe -silent` and the two streams captured to separate
//! files, the sentinel lands on **stdout** (31 bytes — exactly the sentinel
//! line and nothing else) while stderr carries 840 bytes with no Juggler line
//! at all. Observed 2026-10-04 on macOS against the pinned build
//! `152.0.4-beta.31-7b8d12d6`.
//!
//! The sentinel text itself checks out against that build's bundle: it occurs
//! in `Camoufox.app/Contents/Resources/omni.ja`, while the `"Juggler pipe
//! initialized"` spelling this module used to document occurs nowhere in it.
//!
//! So the code reads stdout because that is where the sentinel actually is.
//! Switching it to stderr to agree with PROTOCOL.md would break every launch —
//! this path is load-bearing for the ztools twitter collector. See the
//! divergence note at the top of `docs/PROTOCOL.md`, and the two regression
//! tests at the bottom of this file — `sentinel_on_stdout_is_ready` and
//! `sentinel_on_stderr_alone_is_not_ready` — which between them fail if the
//! read ever moves back to stderr.

use crate::process::ProcessError;

use std::io::{BufRead, BufReader};
use std::process::Child;
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// The sentinel string that Camoufox writes to stdout when the Juggler pipe
/// transport is initialized and ready to accept commands.
///
/// This is the vanilla Playwright Firefox wording; the patched build ships no
/// `"Juggler pipe initialized"` string to be found (see the module docs).
const READINESS_SENTINEL: &str = "Juggler listening to the pipe";

/// Outcome of the stdout reader thread.
enum ReadResult {
    /// The sentinel string was found. The `String` contains all stdout output
    /// collected up to and including the sentinel line.
    Ready(String),
    /// The child's stdout stream reached EOF (process exited) before the
    /// sentinel was found. The `String` contains all collected stdout output.
    Eof(String),
    /// An I/O error occurred while reading stdout.
    Error(std::io::Error),
}

/// Wait for the Camoufox process to signal readiness on stdout.
///
/// Spawns a background thread that reads from the child's stdout line by line,
/// looking for the `READINESS_SENTINEL` substring. Uses a channel with
/// timeout to enforce the deadline.
///
/// # Arguments
///
/// * `child` - The spawned child process. Its `stdout` will be taken
///   (consumed) by this function. After this call, `child.stdout` is `None`.
/// * `timeout` - Maximum time to wait for the sentinel string.
///
/// # Returns
///
/// * `Ok(stdout_output)` - The sentinel was found. Returns all stdout output
///   collected up to that point (useful for logging/debugging).
/// * `Err(ProcessError::ExitedBeforeReady)` - The process exited before the
///   sentinel appeared.
/// * `Err(ProcessError::Timeout)` - The timeout expired. The child process
///   is still running; the caller should kill it.
/// * `Err(ProcessError::Io)` - An I/O error occurred reading stdout, or
///   stdout was not piped.
pub fn wait_for_ready(child: &mut Child, timeout: Duration) -> Result<String, ProcessError> {
    // Take ownership of stdout. If it was not piped, return an error.
    let stdout = child.stdout.take().ok_or_else(|| {
        ProcessError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            "child stdout is not piped (was it already taken?)",
        ))
    })?;

    let (tx, rx) = mpsc::channel::<ReadResult>();

    // Spawn a reader thread. This thread owns the stdout handle and reads
    // line by line until it finds the sentinel, hits EOF, or encounters an
    // I/O error.
    std::thread::Builder::new()
        .name("camoufox-stdout-reader".into())
        .spawn(move || {
            stdout_reader(stdout, tx);
        })
        .map_err(ProcessError::Io)?;

    // Wait for the reader thread to report, with timeout.
    let deadline = Instant::now() + timeout;
    let result = rx.recv_timeout(timeout);

    match result {
        Ok(ReadResult::Ready(output)) => Ok(output),

        Ok(ReadResult::Eof(output)) => {
            // Process stdout closed. Check if the process has exited.
            let code = child.try_wait().ok().flatten().and_then(|s| s.code());
            Err(ProcessError::ExitedBeforeReady {
                code,
                captured: output,
            })
        }

        Ok(ReadResult::Error(e)) => Err(ProcessError::Io(e)),

        Err(mpsc::RecvTimeoutError::Timeout) => {
            // The reader thread may still be running (blocking on stdout read).
            // We report timeout; the caller is responsible for killing the
            // child, which will cause the reader thread to see EOF and exit.
            //
            // Try to collect any stdout output that may have been captured.
            // We cannot get it from the reader thread easily, so we report
            // what we know.
            let remaining = deadline.saturating_duration_since(Instant::now());
            let _ = remaining;

            // Give the reader thread a brief moment to send a partial result
            // in case it finished just after the timeout.
            let stdout_output = match rx.recv_timeout(Duration::from_millis(10)) {
                Ok(ReadResult::Eof(s) | ReadResult::Ready(s)) => s,
                _ => String::new(),
            };

            Err(ProcessError::Timeout {
                timeout,
                captured: stdout_output,
            })
        }

        Err(mpsc::RecvTimeoutError::Disconnected) => {
            // The reader thread panicked or was dropped without sending.
            // Check if the process exited.
            let code = child.try_wait().ok().flatten().and_then(|s| s.code());
            Err(ProcessError::ExitedBeforeReady {
                code,
                captured: String::new(),
            })
        }
    }
}

/// Background function that reads stdout line by line and looks for the
/// readiness sentinel.
///
/// Sends exactly one [`ReadResult`] on the channel before returning.
fn stdout_reader(stdout: std::process::ChildStdout, tx: mpsc::Sender<ReadResult>) {
    let reader = BufReader::new(stdout);
    let mut collected = String::new();

    for line_result in reader.lines() {
        match line_result {
            Ok(line) => {
                log::debug!("browser stdout: {line}");
                collected.push_str(&line);
                collected.push('\n');

                if line.contains(READINESS_SENTINEL) {
                    let _ = tx.send(ReadResult::Ready(collected));
                    return;
                }
            }
            Err(e) => {
                // I/O error reading stdout (unlikely but possible).
                log::warn!("stdout read error: {e}");
                let _ = tx.send(ReadResult::Error(e));
                return;
            }
        }
    }

    // EOF reached — the child closed its stdout (likely exited).
    let _ = tx.send(ReadResult::Eof(collected));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};

    /// Helper: spawn a child process that writes the given text to stdout
    /// and then exits. stdout is piped because that is the stream
    /// `wait_for_ready` reads.
    fn spawn_echo_stdout(text: &str) -> Child {
        Command::new("/bin/sh")
            .arg("-c")
            .arg(format!("echo '{text}'"))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn /bin/sh")
    }

    /// Helper: spawn a child process that writes to stdout after a delay.
    fn spawn_delayed_stdout(text: &str, delay_secs: f64) -> Child {
        Command::new("/bin/sh")
            .arg("-c")
            .arg(format!("sleep {delay_secs} && echo '{text}'"))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn /bin/sh")
    }

    /// Helper: spawn a child that writes the given text to **stderr only** and
    /// exits, with stdout piped but silent.
    ///
    /// The mirror image of [`spawn_echo_stdout`]; used to prove the readiness
    /// sentinel is only ever accepted from stdout.
    fn spawn_echo_stderr_only(text: &str) -> Child {
        Command::new("/bin/sh")
            .arg("-c")
            .arg(format!("echo '{text}' >&2"))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn /bin/sh")
    }

    #[test]
    fn detects_readiness_sentinel() {
        let mut child = spawn_echo_stdout(READINESS_SENTINEL);
        let result = wait_for_ready(&mut child, Duration::from_secs(5));
        assert!(result.is_ok(), "expected Ok, got: {:?}", result.err());
        let output = result.unwrap();
        assert!(
            output.contains(READINESS_SENTINEL),
            "output should contain sentinel: {output:?}"
        );
        let _ = child.wait();
    }

    #[test]
    fn detects_sentinel_among_other_output() {
        let text = format!("some startup noise\nmore noise\n{READINESS_SENTINEL}\nand more after");
        // Use printf to handle newlines properly.
        let mut child = Command::new("/bin/sh")
            .arg("-c")
            .arg(format!("printf '{}\\n'", text.replace('\n', "\\n")))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn");

        let result = wait_for_ready(&mut child, Duration::from_secs(5));
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("some startup noise"));
        assert!(output.contains(READINESS_SENTINEL));
        let _ = child.wait();
    }

    #[test]
    fn returns_exited_before_ready_when_no_sentinel() {
        // Process that writes something else and exits.
        let mut child = spawn_echo_stdout("no sentinel here");
        let result = wait_for_ready(&mut child, Duration::from_secs(5));
        assert!(
            matches!(result, Err(ProcessError::ExitedBeforeReady { .. })),
            "expected ExitedBeforeReady, got: {result:?}"
        );

        if let Err(ProcessError::ExitedBeforeReady { captured, .. }) = result {
            assert!(
                captured.contains("no sentinel here"),
                "captured output should contain what the child wrote: {captured:?}"
            );
        }
        let _ = child.wait();
    }

    #[test]
    fn returns_timeout_when_process_hangs() {
        // Spawn `sleep` DIRECTLY rather than through `/bin/sh -c`. The shell
        // form forks `sleep` as a grandchild, so killing the child reaps only
        // the shell and the sleeper survives -- reparented, outliving the test
        // binary, and still running minutes later holding nothing. The house
        // orphan check caught exactly that (`sleep 60`, alive after the run).
        // Nothing is lost by dropping the shell: the `echo` was never asserted
        // on, and a process that writes nothing and hangs is what this asserts.
        let mut child = Command::new("sleep")
            .arg("60")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn");

        let result = wait_for_ready(&mut child, Duration::from_millis(200));
        assert!(
            matches!(result, Err(ProcessError::Timeout { .. })),
            "expected Timeout, got: {result:?}"
        );

        // Clean up: kill the hanging process. `wait` is what reaps it, so a
        // failure here must not be swallowed into an unreaped child.
        child.kill().expect("kill the sleeper");
        child.wait().expect("reap the sleeper");
    }

    #[test]
    fn returns_error_when_stdout_not_piped() {
        // stderr is piped on purpose: the ONLY reason this can fail is that
        // stdout is missing, so a read switched to stderr would sail past it.
        let mut child = Command::new("/bin/sh")
            .arg("-c")
            .arg("true")
            .stdin(Stdio::null())
            .stdout(Stdio::null()) // Not piped!
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn");

        let result = wait_for_ready(&mut child, Duration::from_secs(1));
        assert!(
            matches!(result, Err(ProcessError::Io(_))),
            "expected Io error, got: {result:?}"
        );
        let _ = child.wait();
    }

    #[test]
    fn returns_error_when_stdout_already_taken() {
        let mut child = Command::new("/bin/sh")
            .arg("-c")
            .arg("true")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn");

        // Take stdout manually first; stderr stays available, so the Io error
        // can only come from the missing stdout handle.
        let _stolen = child.stdout.take();

        let result = wait_for_ready(&mut child, Duration::from_secs(1));
        assert!(
            matches!(result, Err(ProcessError::Io(_))),
            "expected Io error for missing stdout, got: {result:?}"
        );
        let _ = child.wait();
    }

    #[test]
    fn sentinel_with_prefix_is_detected() {
        // Firefox may prefix the sentinel with other text on the same line.
        // Our detection uses `contains`, so partial matches work.
        let text = format!("[timestamp] {READINESS_SENTINEL}");
        let mut child = spawn_echo_stdout(&text);
        let result = wait_for_ready(&mut child, Duration::from_secs(5));
        assert!(result.is_ok());
        let _ = child.wait();
    }

    #[test]
    fn delayed_sentinel_is_detected_within_timeout() {
        let mut child = spawn_delayed_stdout(READINESS_SENTINEL, 0.1);
        let result = wait_for_ready(&mut child, Duration::from_secs(5));
        assert!(result.is_ok(), "expected Ok, got: {:?}", result.err());
        let _ = child.wait();
    }

    #[test]
    fn empty_stdout_returns_exited_before_ready() {
        // Process that immediately exits without writing anything to stdout.
        let mut child = Command::new("/bin/sh")
            .arg("-c")
            .arg("true")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn");

        let result = wait_for_ready(&mut child, Duration::from_secs(5));
        assert!(
            matches!(result, Err(ProcessError::ExitedBeforeReady { .. })),
            "expected ExitedBeforeReady, got: {result:?}"
        );
        let _ = child.wait();
    }

    /// Pins WHICH stream the sentinel is accepted from, from both sides.
    ///
    /// PROTOCOL.md § 3 says the sentinel arrives on stderr; the patched build
    /// emits it on stdout (see this module's docs). Both halves of that
    /// disagreement are asserted here so neither direction can be "fixed" by
    /// accident:
    ///
    /// - `sentinel_on_stdout_is_ready` — sentinel on stdout ⇒ ready. Goes red
    ///   if the read stream moves to stderr, or if the sentinel match breaks.
    /// - `sentinel_on_stderr_alone_is_not_ready` — sentinel on stderr only, and
    ///   stdout silent ⇒ not ready. Goes red if the read stream moves to
    ///   stderr, which would silently accept a stream the browser never uses.
    #[test]
    fn sentinel_on_stdout_is_ready() {
        let mut child = spawn_echo_stdout(READINESS_SENTINEL);
        let result = wait_for_ready(&mut child, Duration::from_secs(5));
        let output =
            result.unwrap_or_else(|e| panic!("sentinel on stdout must be ready, got: {e:?}"));
        assert!(
            output.contains(READINESS_SENTINEL),
            "returned output must carry the sentinel: {output:?}"
        );
        let _ = child.wait();
    }

    #[test]
    fn sentinel_on_stderr_alone_is_not_ready() {
        let mut child = spawn_echo_stderr_only(READINESS_SENTINEL);
        // stdout closes empty the instant `echo` exits, so this reports
        // ExitedBeforeReady (with empty captured output) rather than blocking
        // to the deadline. That is the real behaviour, asserted as-is.
        let result = wait_for_ready(&mut child, Duration::from_secs(5));
        match result {
            Err(ProcessError::ExitedBeforeReady { captured, .. }) => assert!(
                !captured.contains(READINESS_SENTINEL),
                "stderr-only sentinel must not be reported as captured: {captured:?}"
            ),
            Err(other) => panic!("expected ExitedBeforeReady, got: {other:?}"),
            Ok(output) => panic!("sentinel on stderr must NOT signal ready, but got: {output:?}"),
        }
        let _ = child.wait();
    }
}
