//! The Ctrl-C drain contract, driven by a REAL signal to a REAL child process.
//!
//! `eval/drain.rs` shipped as the v3.2.0 behaviour: the first Ctrl-C finishes the
//! task in flight, records the run as truncated and releases the GPU lock, and a
//! second one exits 130. Every test of it set the flag through `DrainRequest` --
//! which exercises the flag and nothing else. Signal delivery, handler dispatch,
//! the truncation verdict and the 130 exit code were all untested, and each is
//! exactly what a handler can get wrong while the flag keeps working.
//!
//! WHAT EACH CASE ASSERTS, and why it earns its place:
//!
//!   * the FIRST Ctrl-C, delivered while a task request is in flight, produces
//!     the handler's own banner on the child's stderr -- that banner is the
//!     evidence the signal was DELIVERED and DISPATCHED, rather than assumed
//!     because the child was still running -- and then the run drains: the task
//!     finishes, the completeness line says truncated, and the command exits 1
//!     naming the interrupt;
//!   * a SECOND Ctrl-C inside the window the first one opened exits 130. It is
//!     sent with the task still held by the stub, so the child is provably
//!     mid-request and there is no race about whether the run had already
//!     drained;
//!   * a Ctrl-C that lands during the measurement phase, before any task has
//!     started, drains NOTHING and still refuses -- the difference between
//!     "truncated" and "did not start" is the difference between a re-runnable
//!     resume and a wasted one.
//!
//! NOTHING OUTSIDE THE CHILD IS EVER SIGNALLED. `ChildRun::signal` takes the pid
//! from the live `Child`, keeps the child unreaped so its pid cannot be recycled
//! onto a stranger, spawns it in its own process group, and addresses that one
//! pid. There is no code path here by which this test process or its shell
//! becomes a signal target.
//!
//! THE STUB HOLDS requests instead of answering on a timer, which is what removes
//! the last guess. "A task request is in flight" is something the stub reports,
//! and "the handler ran" is something the handler printed. Nothing here sleeps
//! waiting for either.

#[path = "support/child.rs"]
mod child;
#[path = "support/mod.rs"]
mod support;
// See `eval_runner.rs`: the shared module's items are reachable API of this
// test binary, so one consumer not needing one is not dead code.
pub use support::*;

use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use child::ChildRun;

/// A task model name. `eval::report::is_test_model` treats a `mock`-prefixed name
/// as a test double and skips writing `eval_history.json` for it at all, which is
/// what keeps these runs from appending to the repo's own history. The redirected
/// `HOME` is the second line of defence: `default_eval_dir()` resolves under it.
const MODEL: &str = "mock-drain-model";

const BANNER: &str = "Ctrl-C: finishing the task in flight";
const SECOND_BANNER: &str = "Ctrl-C again: quitting now";

/// How often a held request re-checks its gate. The granularity of noticing, not
/// a guess about anyone's timing.
const POLL: Duration = Duration::from_millis(5);

/// The stub, and the two conditions the tests wait on.
struct Stub {
    port: u16,
    /// Probe requests seen. One has arrived means the handler is installed: the
    /// install happens on the line before the model is resolved, so a probe in
    /// flight means a SIGINT can no longer land before the handler exists -- and
    /// an unhandled SIGINT would kill the child with the default action instead
    /// of draining it, which is the race this counter closes.
    probes: Arc<AtomicUsize>,
    /// Task requests seen, all of them held until [`Stub::release`].
    tasks: Arc<AtomicUsize>,
    open_probes: Arc<AtomicBool>,
    open_tasks: Arc<AtomicBool>,
}

impl Stub {
    fn open_probes(&self) {
        self.open_probes.store(true, Ordering::SeqCst);
    }

    fn release(&self) {
        self.open_tasks.store(true, Ordering::SeqCst);
        self.open_probes.store(true, Ordering::SeqCst);
    }
}

/// How long a stub waits for its gate, so a gate that is never opened fails the
/// test instead of parking a thread forever. Ten times the longest wait above.
const GATE_CAP: Duration = Duration::from_secs(60);

/// A loopback stub that HOLDS requests until the test opens the gate.
///
/// The three measurement calls are recognised by their BUDGET (`max_tokens` of 1
/// or 64) rather than by counting them. Counting would be a guess about how many
/// probes a run makes, and if that number changed the first Ctrl-C would land
/// during measurement instead of mid-task -- a different feature that would then
/// need a different test. Keying on the shape means the waits below land on a
/// task request whatever the probe count becomes.
///
/// `GET /v1/models` is answered immediately and never held: `model-eval --suite
/// full` resolves its models through it, and holding it would deadlock before the
/// loop is ever entered.
fn blocking_stub() -> Stub {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let probes = Arc::new(AtomicUsize::new(0));
    let tasks = Arc::new(AtomicUsize::new(0));
    let open_probes = Arc::new(AtomicBool::new(false));
    let open_tasks = Arc::new(AtomicBool::new(false));
    let (probes_c, tasks_c) = (Arc::clone(&probes), Arc::clone(&tasks));
    let (open_probes_c, open_tasks_c) = (Arc::clone(&open_probes), Arc::clone(&open_tasks));

    thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let mut stream = stream;
            let mut buf = vec![0u8; 65_536];
            // A connection carrying no request is not a request: the readiness
            // probe in `support` opens one and sends nothing.
            let n = stream.read(&mut buf).unwrap_or(0);
            if n == 0 {
                continue;
            }
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            let first = request.lines().next().unwrap_or("").to_string();

            if !first.starts_with("GET /v1/models") {
                let is_probe =
                    request.contains("\"max_tokens\":1") || request.contains("\"max_tokens\":64");
                let gate = if is_probe {
                    probes_c.fetch_add(1, Ordering::SeqCst);
                    &open_probes_c
                } else {
                    tasks_c.fetch_add(1, Ordering::SeqCst);
                    &open_tasks_c
                };
                let until = std::time::Instant::now() + GATE_CAP;
                while !gate.load(Ordering::SeqCst) && std::time::Instant::now() < until {
                    thread::sleep(POLL);
                }
            }
            // An answer that satisfies no check: this file is about the SIGNAL,
            // never about a score.
            let body = r#"{"choices":[{"message":{"content":"drained"},"finish_reason":"stop"}]}"#;
            let http = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(http.as_bytes());
            let _ = stream.flush();
        }
    });
    support::await_stub(port);
    Stub {
        port,
        probes,
        tasks,
        open_probes,
        open_tasks,
    }
}

/// A sandbox holding everything the run reads: `HOME`, the config, the eval conf
/// dir the roster loads from, and the redirected GPU lock.
///
/// The roster files are COPIES of the shipped ones, not minimal stubs. They are
/// data the production path parses into every task's prompt and into the drawn
/// image fixtures, so a hand-written stub would make this file measure a program
/// that does not exist; copying means a change to the shipped roster reaches here
/// instead of drifting past it.
fn sandbox(name: &str) -> (PathBuf, PathBuf) {
    let home = support::fresh(name);
    let conf = home.join("conf");
    fs::create_dir_all(&conf).unwrap();
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the checkout that holds rust/");
    for file in ["eval_inputs.toml", "eval_vision.toml"] {
        fs::copy(checkout.join("conf").join(file), conf.join(file))
            .unwrap_or_else(|e| panic!("copy the shipped {file} into the sandbox: {e}"));
    }
    let gpu = home.join("gpu-lock");
    fs::create_dir_all(&gpu).unwrap();
    (home, gpu)
}

/// The `ztools model-eval --suite full` child, wired to the stub and sandbox.
///
/// `--task json` narrows the run to ONE task, and that is what makes the drain
/// observable: the first Ctrl-C arrives while that task is in flight, the task
/// finishes, and the next boundary is where the run stops -- so "after 1 task(s)"
/// is a statement about a known number rather than about however many of two dozen
/// the stub happened to have answered.
fn child(home: &Path, gpu: &Path, port: u16) -> Command {
    support::write_config(
        home,
        &format!(
            "osaurus_url = \"http://127.0.0.1:{port}\"\n\
             llm_timeout_secs = 20\n\
             llm_quick_timeout_secs = 5\n\
             eval_conf_dirs = [\"{}\"]\n",
            home.join("conf").display()
        ),
    );
    let mut cmd = Command::new(support::bin());
    cmd.env("HOME", home)
        // The machine-wide GPU lock, redirected: this run is a stub talking to a
        // stub and must never take the lock a peer's real measurement holds.
        .env("ZTOOLS_GPU_LOCK_DIR", gpu)
        // And the two directories the learning path writes.
        .env("EVAL_SIGNALS_DIR", home.join("signals"))
        .env("EVAL_OUTPUT_DIR", home.join("outputs"))
        .arg("--config")
        .arg(home.join("ztools.toml"))
        .args([
            "model-eval",
            "--suite",
            "full",
            "--model",
            MODEL,
            "--task",
            "json",
        ]);
    cmd
}

/// The first Ctrl-C drains: the task in flight finishes, the run is recorded as
/// truncated, and the command exits non-zero so a sweep files it as FAILED and
/// `--resume` re-runs it.
#[test]
fn the_first_ctrl_c_drains_the_task_in_flight_and_exits_1_naming_the_interrupt() {
    let stub = blocking_stub();
    stub.open_probes();
    let (home, gpu) = sandbox("drain-first");
    let mut run = ChildRun::spawn(child(&home, &gpu, stub.port));

    // Wait until a TASK request is genuinely in flight. The stub holding it is
    // the condition; "the child has been running a while" is not, and is the
    // flake this test exists to stop having.
    wait_for(
        "the child to have a task request in flight against the stub",
        || stub.tasks.load(Ordering::SeqCst) > 0,
    );

    run.signal("INT");
    // The handler's own words. Nothing else prints this line, so seeing it is
    // proof the signal was delivered AND dispatched -- the half the DrainRequest
    // tests could never reach.
    run.await_stderr(BANNER);

    // Now let the held request answer, so the task can finish: the entire point
    // of the first Ctrl-C is that the answer already paid for is KEPT.
    stub.release();
    let out = run.wait_for_exit("the drained run to exit");
    let stderr = String::from_utf8_lossy(&out.stderr);

    assert_refused(
        "a run drained by the first Ctrl-C",
        &out,
        1,
        &[
            BANNER,
            // The drain verdict, with the task count in it: one task ran and the
            // second alias of it was never started.
            &format!("Stopping {MODEL} on request after 1 task(s)"),
            "recorded as truncated",
            // And the top-level refusal, which is what files the run as FAILED.
            "interrupted by Ctrl-C after 1 model run(s)",
        ],
    );

    // The drained task's outcome is on stdout, not only in the exit code: drain
    // mode exists so the answer that already arrived is recorded rather than
    // lost, and this is where it is visible.
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("1 tasks"),
        "the drained run must report the one task it finished: {stdout}"
    );
    assert!(
        !stderr.contains(SECOND_BANNER),
        "one Ctrl-C must not take the quit-now path: {stderr}"
    );
}

/// A second Ctrl-C, inside the window the first one opened, exits 130.
///
/// `130` and not `0` or `1`: those are the two wrong answers, and each is a real
/// mistake rather than a hypothetical one -- `0` files a truncated run as a
/// success, `1` files it as a failure OF THE PROGRAM, which sends the next
/// person looking for a bug that is not there.
#[test]
fn a_second_ctrl_c_within_the_drain_window_exits_130() {
    let stub = blocking_stub();
    stub.open_probes();
    let (home, gpu) = sandbox("drain-second");
    let mut run = ChildRun::spawn(child(&home, &gpu, stub.port));

    wait_for(
        "the child to have a task request in flight against the stub",
        || stub.tasks.load(Ordering::SeqCst) > 0,
    );
    run.signal("INT");
    run.await_stderr(BANNER);

    // Still holding the request, so the child is provably mid-task and the
    // second Ctrl-C lands in exactly the state "quit now" exists for.
    run.signal("INT");
    let out = run.wait_for_exit("the second Ctrl-C to quit the run now");
    let stderr = String::from_utf8_lossy(&out.stderr);

    assert_refused(
        "a run quit by the second Ctrl-C",
        &out,
        130,
        &[BANNER, SECOND_BANNER],
    );
    assert!(
        !stderr.contains("recorded as truncated"),
        "a run that quit NOW has no drained verdict to report, because nothing \
         was drained: {stderr}"
    );
    assert_eq!(
        stub.tasks.load(Ordering::SeqCst),
        1,
        "exactly the request that was in flight, and no retry after the quit: \
         a task that never finished must not be re-asked for"
    );
}

/// A Ctrl-C that lands during the measurement phase drains NOTHING, and still
/// refuses. `after 0 task(s)` and `after 1 task(s)` are different facts about the
/// world and only one of them is true here.
///
/// Deterministic because the stub holds the measurement request: the run cannot
/// reach the loop's first boundary until the test opens the gate, and the gate is
/// opened only after the banner proves the flag is set.
#[test]
fn a_ctrl_c_during_measurement_drains_nothing_and_still_refuses() {
    let stub = blocking_stub();
    let (home, gpu) = sandbox("drain-measure");
    let mut run = ChildRun::spawn(child(&home, &gpu, stub.port));

    // A held measurement request means the handler is installed AND the loop has
    // not been entered, so the flag cannot be cleared by finishing a task before
    // the boundary that reads it.
    wait_for(
        "the child to have a measurement request in flight against the stub",
        || stub.probes.load(Ordering::SeqCst) > 0,
    );
    run.signal("INT");
    run.await_stderr(BANNER);
    stub.release();

    let out = run.wait_for_exit("the measurement-time interrupted run to exit");

    assert_refused(
        "a run interrupted before its first task",
        &out,
        1,
        &[
            BANNER,
            "on request after 0 task(s)",
            "the run is recorded as truncated",
            "interrupted by Ctrl-C",
        ],
    );
    assert_eq!(
        stub.tasks.load(Ordering::SeqCst),
        0,
        "no task may be started once the operator has asked to stop"
    );
}

/// The counters the waits above depend on are only meaningful if a connection
/// carrying no request is not counted as one -- the invariant that lets
/// `support::await_stub` probe readiness by connecting. If a bare connect were
/// recorded, "a task request is in flight" could be true while the child was
/// still starting, and these tests would pass or flake for a reason that has
/// nothing to do with Ctrl-C.
#[test]
fn the_readiness_probe_is_not_counted_as_a_request() {
    let stub = blocking_stub();

    assert_eq!(
        (
            stub.probes.load(Ordering::SeqCst),
            stub.tasks.load(Ordering::SeqCst)
        ),
        (0, 0),
        "await_stub's connection carried no request and must be counted as \
         neither a measurement nor a task"
    );
    stub.release();
}
