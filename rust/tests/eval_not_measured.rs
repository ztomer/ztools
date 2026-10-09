//! A MEASUREMENT THAT NEVER HAPPENED must exit non-zero and say so.
//!
//! `ztools model-eval --model <name>` against an unreachable server exited 0 and
//! printed a table of 0.0% rows. The transport error was swallowed into per-test
//! scores — `model_eval.rs` read `if let Ok(r) = resp && r.status().is_success()`
//! and left every other outcome as an empty string, which the scorer read as
//! `0/total` — so "the server was down" and "the model is bad" produced the same
//! exit code and nearly the same screen. Three more paths in the same family,
//! all reporting a refusal as a result:
//!
//!   * `--suite full` against a dead server: every outcome came back `INFRA` with
//!     score 0, so completeness was HAPPY (it counts tasks that REPORTED, and a
//!     dead server's tasks all report), the run printed "mean score 0.0", and it
//!     wrote a 0-mean entry into the history `load_historical_stats` ranks from;
//!   * a model refused for being over budget: `eprintln!("✗ Skipping …")` then
//!     `continue`, exit 0, nothing in the output at all;
//!   * `--model all` with one unreachable model: `if let Ok(res)` dropped that
//!     model, so a twenty-model sweep reported success over eighteen.
//!
//! WHY THIS IS ITS OWN FILE rather than more cases in `cli_refusals.rs`: that
//! file is the same contract at the same altitude, and it was already at the
//! house 500-line cap. The sibling's rules apply verbatim — assert the exit code
//! AND the message, or the assertion is theatre — and the calibration note from
//! `differential-blindness` applies hardest here: a table full of dashes is the
//! fix, so the control that a healthy run still exits 0 and still prints
//! percentages is what keeps the fix from being "report everything as failed".
//!
//! Hermetic: `127.0.0.1` only, the eval roster COPIED from the shipped `conf/`
//! (a hand-written stub would measure a program that does not exist), and the
//! GPU lock redirected into the sandbox so a peer's real measurement is never
//! disturbed. `EVAL_ALLOW_OVERSIZE` is cleared, or a developer with it exported
//! would make the over-budget refusal stop happening and this file would pass by
//! testing nothing.

#[path = "support/mod.rs"]
mod support;
// See `eval_runner.rs`: the shared module's items are reachable API of this
// test binary, so one consumer not needing one is not dead code.
pub use support::*;

use std::fs;
use std::net::TcpListener;
use std::path::Path;
use std::process::{Command, Output};

/// The marker the program uses for "this row holds no score", spelled here so a
/// rename cannot silently un-assert every case below.
const NOT_MEASURED: &str = "NOT MEASURED";

/// The glyph that stands in for a score nobody took.
const NO_SCORE: &str = "—";

/// A loopback port nothing is listening on: bound, then released.
///
/// `bind` then `drop` rather than a constant, because a constant is either
/// permanently wrong or eventually wrong — some other process takes the port and
/// the test quietly starts measuring whatever now owns it. Losing the race makes
/// the assertion go red naming the URL that answered.
fn closed_loopback_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// A loopback stub that answers every request with `body` for the life of the
/// test, with the connect-wait a client actually needs rather than a sleep.
fn stub_answering(body: &'static str) -> u16 {
    use std::io::{Read, Write};
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let mut stream = stream;
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            let http = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                 Connection: close\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(http.as_bytes());
            let _ = stream.flush();
        }
    });
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(250))
        .is_err()
    {
        assert!(
            std::time::Instant::now() < deadline,
            "the stub on 127.0.0.1:{port} never accepted a connection"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    port
}

/// A sandbox holding everything the full suite reads: `HOME`, the config, the
/// eval conf dir with the shipped roster copied in, and a place for the GPU lock.
///
/// The roster files are COPIES, for the reason `tests/drain_signal.rs` gives.
fn sandbox_with_roster(name: &str) -> std::path::PathBuf {
    let home = fresh(name);
    let conf = home.join("conf");
    fs::create_dir_all(&conf).unwrap();
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the checkout that holds rust/");
    for file in ["eval_inputs.toml", "eval_vision.toml"] {
        fs::copy(checkout.join("conf").join(file), conf.join(file))
            .unwrap_or_else(|e| panic!("copy the shipped {file} into the sandbox: {e}"));
    }
    home
}

fn run_eval_cmd(home: &Path, args: &[&str]) -> Command {
    let gpu = home.join("gpu-lock");
    fs::create_dir_all(&gpu).unwrap();
    let mut cmd = Command::new(bin());
    cmd.env("HOME", home)
        .env("ZTOOLS_GPU_LOCK_DIR", &gpu)
        .env("EVAL_SIGNALS_DIR", home.join("signals"))
        .env("EVAL_OUTPUT_DIR", home.join("outputs"))
        .arg("--config")
        .arg(home.join("ztools.toml"))
        .args(args);
    cmd
}

/// The binary with `HOME` sandboxed, the GPU lock redirected, the learning-path
/// directories inside the sandbox, and the oversize override cleared.
fn run_eval(home: &Path, args: &[&str]) -> Output {
    run_eval_cmd(home, args)
        .env_remove("EVAL_ALLOW_OVERSIZE")
        .output()
        .expect("run the binary under test")
}

/// A config pointing at `port` with fast timeouts, and the roster on `home`.
fn config_for(home: &Path, port: u16) {
    write_config(
        home,
        &format!(
            "osaurus_url = \"http://127.0.0.1:{port}\"\n\
             llm_timeout_secs = 5\nllm_quick_timeout_secs = 2\n\
             eval_conf_dirs = [\"{}\"]\n",
            home.join("conf").display()
        ),
    );
}

/// The smoke path against a dead server FAILS, and its table holds no score.
///
/// The headline defect. Before the fix this exited 0 and printed five rows of
/// `0.0%` / `failed`. The exit code is the half an operator's tooling reads; the
/// table is the half a human reads, and either alone is a failure that hides — a
/// non-zero exit beside a table of zeros still invites "it scored badly", and a
/// table of dashes with exit 0 is still filed as a success by a sweep.
#[test]
fn a_single_model_eval_against_a_dead_server_fails_and_scores_nothing() {
    let home = fresh("not-measured-dead-server-single");
    let port = closed_loopback_port();
    config_for(&home, port);

    let out = run_eval(&home, &["model-eval", "--model", "probe-model"]);
    let dialled = format!("http://127.0.0.1:{port}/v1/chat/completions");

    assert_refused(
        "a smoke eval against an inference server that is not listening",
        &out,
        1,
        &[
            "ztools:",
            NOT_MEASURED,
            // The URL it dialled and the verdict: "an error happened" alone
            // leaves three candidates (server down, wrong port, wrong machine).
            &dialled,
            "gave no answer",
        ],
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains(NOT_MEASURED) && stdout.contains(NO_SCORE),
        "the table must show that nothing was measured:\n{stdout}"
    );
    assert!(
        !stdout.contains("0.0%"),
        "a run that measured nothing must not print a percentage:\n{stdout}"
    );
    assert!(
        !stdout.contains("NaN"),
        "the absent score must not leak its placeholder:\n{stdout}"
    );
}

/// The full suite against a dead server FAILS, and prints no mean.
///
/// The INFRA-scored-as-a-failure case. Every task comes back `score: 0` with
/// `failure_category: INFRA`, so the completeness check is satisfied and the old
/// exit code was 0. The mean is the number a ranking reads, so its ABSENCE is
/// asserted directly: "mean score 0.0" for a model that was never asked is the
/// defect in one string.
#[test]
fn a_full_suite_eval_against_a_dead_server_fails_and_prints_no_mean() {
    let home = sandbox_with_roster("not-measured-dead-server-full");
    let port = closed_loopback_port();
    config_for(&home, port);

    let out = run_eval_cmd(
        &home,
        &[
            "model-eval",
            "--suite",
            "full",
            "--model",
            "probe-full-model",
            "--task",
            "json",
        ],
    )
    .env("EVAL_ALLOW_OVERSIZE", "1")
    .output()
    .expect("run the binary under test");

    assert_refused(
        "a full-suite eval against an inference server that is not listening",
        &out,
        1,
        &[
            "ztools:",
            NOT_MEASURED,
            "probe-full-model",
            "0 of 2 task(s) reached the model",
        ],
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains(NO_SCORE) && stdout.contains(NOT_MEASURED),
        "the task table must show that no task reached the model:\n{stdout}"
    );
    assert!(
        !stdout.contains("mean score"),
        "a run that measured nothing must not print a mean:\n{stdout}"
    );
}

/// A model refused for being over budget is distinguishable from a zero.
///
/// The third path, and the one with the least excuse: the refusal was printed
/// and then `continue`d, so the command exited 0 having measured nothing. The
/// model's name and the refusal's own advice are both asserted — the advice is
/// what makes the message actionable rather than merely explanatory, and a
/// refusal that lost the override would strand an operator who knows exactly
/// what to do next.
#[test]
fn an_over_budget_model_is_refused_and_the_run_fails_saying_so() {
    let home = sandbox_with_roster("not-measured-oversize");
    let port = closed_loopback_port();
    config_for(&home, port);

    // The size gate reads the parameter count out of the NAME when nothing is on
    // disk (nothing is, in a sandbox), so the name is the input. "qwen3-999b"
    // and not "probe-999b": `estimate_model_memory_gb` takes the digits before
    // the FIRST "b", and a name with a "b" in its first word falls through to
    // the 4GB default — a refusal test that measured a 4GB model tests nothing.
    let out = run_eval(
        &home,
        &[
            "model-eval",
            "--suite",
            "full",
            "--model",
            "qwen3-999b",
            "--task",
            "json",
        ],
    );

    assert_refused(
        "a model the oversize gate refuses to measure",
        &out,
        1,
        &[
            "ztools:",
            NOT_MEASURED,
            "qwen3-999b",
            // The override, which both refusal wordings carry: the message must
            // still say how to measure it deliberately.
            "EVAL_ALLOW_OVERSIZE=1",
        ],
    );

    // The refusal is on stdout too, next to the table it explains: a message on
    // stderr alone is invisible to whoever reads the saved log a sweep keeps.
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains(NOT_MEASURED) && stdout.contains("qwen3-999b"),
        "the refusal must be visible in the run's own output:\n{stdout}"
    );
}

/// The control: a server that lists no models exits 0.
///
/// Exit 0 has to be REACHABLE in this sandbox, or every "exits 1" above is a
/// constant of the harness rather than a finding about the program. A stub that
/// answers `/v1/models` with an empty roster measures nothing and has nothing to
/// fail about, which is the honest zero.
#[test]
fn a_server_that_lists_no_models_exits_0() {
    let port = stub_answering(r#"{"data":[]}"#);
    let home = fresh("not-measured-empty-roster");
    config_for(&home, port);

    let out = run_eval(&home, &["model-eval", "--model", "all"]);

    assert_eq!(
        exit_code(&out),
        0,
        "a discovery sweep over an empty roster has nothing to fail about: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}
