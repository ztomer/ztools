//! A spawned child process that can be signalled, watched, and reaped.
//!
//! Split from `support` because only the drain suite spawns one, and compiling
//! it into every other test binary made its half-dozen items dead code there --
//! which is a `-D warnings` failure, not a warning.
//!
//! THE SAFETY ARGUMENT for signalling lives on [`ChildRun::signal`]; the short
//! form is that this process is never a signal target, the pid comes from the
//! live `Child`, and the child stays unreaped (a zombie holds its pid, so it
//! cannot be recycled onto a stranger) until `wait_for_exit`.

use std::io::Read;
use std::path::Path;
use std::process::{Child, Command, Output, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Instant;

#[cfg(unix)]
use std::os::unix::process::CommandExt;

use super::{DEADLINE, POLL, take_lock, wait_for};

/// A child whose stdout and stderr are drained on their own threads WHILE it
/// runs.
///
/// Live because the drain test has to read the handler's banner before it sends
/// the second Ctrl-C: that banner is the evidence the first signal was
/// delivered and dispatched, and reading it at exit would be too late to act on.
/// Drained rather than read at exit because a pipe nobody reads fills at 64KB
/// and blocks its writer -- which would present as this suite hanging rather
/// than as the bug it is.
pub struct ChildRun {
    child: Child,
    stdout: Arc<Mutex<String>>,
    stderr: Arc<Mutex<String>>,
    drained: Vec<JoinHandle<()>>,
}

/// Reap on unwind. A `wait()` parked under the assertions is skipped by every
/// panic between the spawn and here, and the orphan would keep holding this
/// test's sandbox and, on the suite's next run, its ports.
impl Drop for ChildRun {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        for handle in self.drained.drain(..) {
            let _ = handle.join();
        }
    }
}

impl ChildRun {
    /// Spawn `cmd` in its own process group and start draining both streams.
    ///
    /// `process_group(0)` makes the child a group leader whose group holds
    /// exactly one member, which is what [`ChildRun::signal`] leans on. Nothing
    /// here reads that group id.
    #[must_use]
    pub fn spawn(mut cmd: Command) -> Self {
        #[cfg(unix)]
        cmd.process_group(0);
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = cmd.spawn().expect("spawn the binary under test");
        let (stdout, out_reader) = drain(child.stdout.take().expect("piped stdout"));
        let (stderr, err_reader) = drain(child.stderr.take().expect("piped stderr"));
        Self {
            child,
            stdout,
            stderr,
            drained: vec![out_reader, err_reader],
        }
    }

    #[must_use]
    pub fn stderr(&self) -> String {
        take_lock(&self.stderr).clone()
    }

    /// Wait until the child has printed `needle` on stderr.
    ///
    /// The condition the drain contract turns on: the Ctrl-C banner is written
    /// by the handler, so seeing it proves the signal was delivered AND
    /// dispatched, rather than assuming it because the child was still running.
    pub fn await_stderr(&self, needle: &str) {
        wait_for(&format!("the child to print {needle:?} on stderr"), || {
            self.stderr().contains(needle)
        });
    }

    /// Deliver `sig` to THIS child and to nothing else.
    ///
    /// Four things make that safe and each one is load-bearing:
    ///
    ///   * the pid comes from the live `Child` -- never a name, never a lookup;
    ///   * the child is not reaped until [`ChildRun::wait_for_exit`], and an
    ///     unreaped child is a zombie whose pid the kernel hands to nobody
    ///     else, so the signal cannot land on a recycled stranger even if the
    ///     child died first;
    ///   * `process_group(0)` means even a group-directed signal would have
    ///     exactly this one member to reach;
    ///   * the handler and its `exit(130)` live in the child, so the 130 this
    ///     proves is the CHILD's exit code and this test process is never a
    ///     signal target at all.
    pub fn signal(&self, sig: &str) {
        let pid = self.child.id();
        let status = Command::new(kill_binary())
            .arg(format!("-{sig}"))
            .arg(pid.to_string())
            .status()
            .unwrap_or_else(|e| {
                panic!(
                    "{} -{sig} {pid} could not run: {e}",
                    kill_binary().display()
                )
            });
        assert!(
            status.success(),
            "{} -{sig} {pid} failed: {status}. The child is gone or unreachable, \
             so the signal never reached it.",
            kill_binary().display()
        );
    }

    /// Wait for the child to exit, returning what it did and everything it said.
    ///
    /// Polled rather than `wait()`ed so that a child which never exits is a
    /// FAILURE that names itself, instead of a suite that hangs until the
    /// harness kills it and reports silence -- silence being the one symptom
    /// that hides its own cause.
    pub fn wait_for_exit(&mut self, what: &str) -> Output {
        let deadline = Instant::now() + DEADLINE;
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => {
                    for handle in self.drained.drain(..) {
                        handle.join().expect("output reader thread");
                    }
                    return Output {
                        status,
                        stdout: take_lock(&self.stdout).clone().into_bytes(),
                        stderr: take_lock(&self.stderr).clone().into_bytes(),
                    };
                }
                Ok(None) => {}
                Err(e) => panic!("polling the child while waiting for {what} failed: {e}"),
            }
            assert!(
                Instant::now() < deadline,
                "timed out after {DEADLINE:?} waiting for {what}\n--- stderr ---\n{}",
                self.stderr()
            );
            thread::sleep(POLL);
        }
    }
}

/// Read `source` to EOF on its own thread, appending into a shared buffer.
fn drain(mut source: impl Read + Send + 'static) -> (Arc<Mutex<String>>, JoinHandle<()>) {
    let buf = Arc::new(Mutex::new(String::new()));
    let sink = Arc::clone(&buf);
    let handle = thread::spawn(move || {
        let mut chunk = [0u8; 4096];
        loop {
            match source.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(n) => take_lock(&sink).push_str(&String::from_utf8_lossy(&chunk[..n])),
            }
        }
    });
    (buf, handle)
}

/// Where `kill(1)` is.
///
/// Absolute, so nothing between here and the signal can be intercepted by a
/// shell function, an alias, or an empty `PATH`. The second spelling is Linux's;
/// if neither is there the test fails loudly rather than quietly not signalling.
fn kill_binary() -> &'static Path {
    for candidate in ["/bin/kill", "/usr/bin/kill"] {
        if Path::new(candidate).is_file() {
            return Path::new(candidate);
        }
    }
    panic!("no kill(1) at /bin/kill or /usr/bin/kill: this test cannot deliver a real signal");
}
