//! The waits the integration tests share: poll a CONDITION, with a deadline and
//! a failure message that says what never happened.
//!
//! Every one of these used to be a `thread::sleep` after `TcpListener::bind` --
//! a guess about someone else's scheduling, eleven times across six files. A
//! guess has two bad outcomes and one useless failure: too short is a flake that
//! gets blamed on the machine, too long is dead time in every run, and neither
//! reports WHICH thing was missing, which is the only part of a flake report
//! anyone can act on.
//!
//! THE RULE STUBS MUST HONOUR, because the readiness probe creates one: a TCP
//! connection carrying no HTTP request is not a request. The probe connects and
//! closes without sending anything; a stub that recorded every connection would
//! count that as its first request and spend its scripted failure budget on it.
//! Recording stubs therefore ignore a read of zero bytes.
//!
//! Mutating-env tests take `ztools::test_env::TestEnv` instead; nothing here
//! touches the environment.
//!
//! The child-process half lives in `support::child`, which only the drain suite
//! includes: compiled into the others, every item in it would be dead code there.

use std::fs;
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::{Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

/// How long any one wait may take before it is a failure.
///
/// Generous enough that a loaded machine never trips it, short enough that a
/// condition which never arrives reports itself instead of looking like a hang.
pub const DEADLINE: Duration = Duration::from_secs(30);

/// How often a wait re-checks. This is the granularity of noticing and nothing
/// more: it guesses about no one's timing, and costs one cheap predicate a tick.
const POLL: Duration = Duration::from_millis(5);

/// Poll `ready` until it holds, or fail naming what never happened.
///
/// The message is the whole point. `assertion failed` on a readiness wait tells
/// the next person nothing; "the stub on 127.0.0.1:41234 never accepted a
/// connection" tells them the stub never came up, which is a different bug from
/// a client whose request was never answered.
///
/// # Panics
///
/// If `ready` has not held within [`DEADLINE`], naming `what` was being waited
/// for.
pub fn wait_for(what: &str, mut ready: impl FnMut() -> bool) {
    let deadline = Instant::now() + DEADLINE;
    loop {
        if ready() {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out after {DEADLINE:?} waiting for {what}"
        );
        thread::sleep(POLL);
    }
}

/// A poisoned lock must not kill a stub or reader thread: that turns one failed
/// request into every later connection hanging out its full timeout.
pub fn take_lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Wait until the stub on `127.0.0.1:port` is accepting connections.
///
/// The connect IS the probe, and it is not a guess about the serving thread: a
/// bound listener completes TCP handshakes out of its own backlog whether or
/// not `accept()` has been called yet, so a successful connect is precisely the
/// condition a client needs. It can be a false NEGATIVE (nothing is listening),
/// which is why this polls instead of connecting once.
pub fn await_stub(port: u16) {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    wait_for(
        &format!("the stub on 127.0.0.1:{port} to accept a connection"),
        || TcpStream::connect_timeout(&addr, Duration::from_millis(250)).is_ok(),
    );
}

/// The binary under test. Cargo sets `CARGO_BIN_EXE_*` for every
/// integration-test target, so this resolves wherever the suite is run from.
#[must_use]
pub fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_ztools"))
}

/// A private `HOME` for one test, named so two tests cannot collide.
///
/// The name is qualified by this process's pid because each integration-test
/// file is its own binary, and a pid alone would not separate two suites run
/// back to back against the same directory.
///
/// # Panics
///
/// If the sandbox directory cannot be created.
#[must_use]
pub fn fresh(name: &str) -> PathBuf {
    let mut d = std::env::temp_dir();
    d.push(format!("ztools-cli-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// The binary, with `HOME` inside the sandbox and the sandbox `--config`.
#[must_use]
pub fn ztool(home: &Path) -> Command {
    let mut c = Command::new(bin());
    c.env("HOME", home)
        .arg("--config")
        .arg(home.join("ztools.toml"));
    c
}

/// Write the flat `ZtoolsConfig` TOML the `--config` flag loads.
///
/// # Panics
///
/// If the file cannot be written.
pub fn write_config(home: &Path, content: &str) {
    fs::write(home.join("ztools.toml"), content).unwrap();
}

/// Assert a refusal: THIS exit code, AND stderr that names the cause.
///
/// Both halves, or the assertion is theatre. A non-zero exit with an unexplained
/// message is the guard-that-never-guards shape one level up: the program did
/// refuse and the operator still cannot tell why. A message with a zero exit is
/// worse -- a warning nobody is required to act on. `what` names the refusal so
/// a regression says WHICH one broke rather than "an assertion failed".
///
/// # Panics
///
/// If the exit code is not `code`, or if stderr contains none of
/// `names_the_cause`.
pub fn assert_refused(what: &str, out: &Output, code: i32, names_the_cause: &[&str]) {
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.code() == Some(code),
        "{what}: expected exit {code}, got {:?}\n--- stdout ---\n{}\n--- stderr ---\n{stderr}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
    );
    for needle in names_the_cause {
        assert!(
            stderr.contains(needle),
            "{what}: exited {code} but its stderr never says {needle:?}, so the \
             refusal arrives with no diagnosis.\n--- stderr ---\n{stderr}"
        );
    }
}

/// The exit code, named for the case that has none: a process killed by a
/// signal never chose a code, and `None` there must not read as "0".
///
/// # Panics
///
/// If the process died on a signal rather than exiting.
#[must_use]
pub fn exit_code(out: &Output) -> i32 {
    out.status.code().unwrap_or_else(|| {
        panic!(
            "the child died on a signal instead of exiting: {}\n--- stderr ---\n{}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        )
    })
}
