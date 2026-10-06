//! Refusals, asserted on the exit code AND on the message.
//!
//! `src/main.rs` holds the only non-zero exit in the program, and until now
//! nothing drove it: the single assertion on a child's status anywhere in the
//! suite checked `success()`, so every refusal in the product -- an unreadable
//! `--config`, a roster that is not there, an inference server that is not
//! answering, a cache that was never written -- exited through a path with no
//! test on it. That is the guard-that-never-guards shape: the check exists, and
//! the failure branch is never taken.
//!
//! Why the MESSAGE is asserted too, and not only the code: a refusal whose text
//! does not say what went wrong is a failure with no diagnosis. An operator who
//! sees exit 1 and nothing else has learned nothing they could not learn from
//! exit 1 alone. For these commands the stderr string IS the contract, so it is
//! pinned verbatim -- including the part that names the fix.
//!
//! All of it is hermetic: `HOME` and every path the binary reads sit in a temp
//! sandbox, every URL is `127.0.0.1`, no browser is launched, and the GPU lock
//! is never reached (its holder is redirected, and the one refusal that gets
//! close happens before the lock is taken at all).

#[path = "support/mod.rs"]
mod support;
// See `eval_runner.rs`: the shared module's items are reachable API of this
// test binary, so one consumer not needing one is not dead code.
pub use support::*;

use std::fs;
use std::net::TcpListener;
use std::path::Path;
use std::process::{Command, Output};

/// A loopback port nothing is listening on: bound, then released.
///
/// `bind` then `drop` rather than a hard-coded number, because a constant is
/// either permanently wrong or eventually wrong -- some other process takes it
/// and this refusal quietly starts testing whatever now owns that port. The
/// window between the two calls is microseconds, and losing it makes the
/// assertion go red naming the URL the program actually dialled, which is the
/// failure worth having.
fn closed_loopback_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// The binary with `HOME` sandboxed and `--config` pointed at `config`.
fn run(home: &Path, config: &Path, args: &[&str]) -> Output {
    let mut cmd = Command::new(bin());
    cmd.env("HOME", home).arg("--config").arg(config).args(args);
    cmd.output().expect("run the binary under test")
}

/// `HOME` sandboxed and `--config` at the sandbox's own `ztools.toml`.
fn run_sandboxed(home: &Path, args: &[&str]) -> Output {
    run(home, &home.join("ztools.toml"), args)
}

/// `--config` names a file that is not there.
///
/// The earliest refusal in the program: config loads before any subcommand is
/// dispatched, so nothing is written and no URL is dialled. The exit code is
/// about the config and about nothing else.
#[test]
fn a_missing_config_file_exits_1_and_names_the_path() {
    let home = fresh("refuse-missing-config");
    let missing = home.join("not-here.toml");

    let out = run(&home, &missing, &["image-renamer", "."]);
    let path = missing.display().to_string();

    assert_refused(
        "a --config path that does not exist",
        &out,
        1,
        &["ztools:", "cannot read config", &path],
    );
}

/// A `--config` file that is not valid `ZtoolsConfig` TOML.
///
/// The other half of the same site. A typo in a key must be reported against
/// that file, never swallowed into defaults that then point the run at the
/// developer's real server -- which is the outcome this assertion rules out.
#[test]
fn a_malformed_config_file_exits_1_and_names_the_file_and_the_column() {
    let home = fresh("refuse-bad-config");
    // An unclosed array: the TOML parser's own error, quoted verbatim below.
    fs::write(home.join("ztools.toml"), "osaurus_url = [[[\n").unwrap();

    let out = run_sandboxed(&home, &["image-renamer", "."]);
    let path = home.join("ztools.toml").display().to_string();

    assert_refused(
        "a --config file that is not ZtoolsConfig TOML",
        &out,
        1,
        &[
            "ztools:",
            "cannot parse config",
            &path,
            // The parser's own reason, not just the fact: "cannot parse" alone
            // sends the reader to the file instead of to the line and column.
            "TOML parse error",
        ],
    );
}

/// `--use-cache` with nothing cached.
///
/// The refusal that must NOT fabricate: asked for a cache it does not have, the
/// command says so and exits non-zero rather than inventing a document out of
/// an empty timeline. The message must also name the flag that would have
/// worked, or the operator is left guessing which invocation was meant.
#[test]
fn asking_for_a_cache_that_was_never_written_exits_1_and_says_so() {
    let home = fresh("refuse-no-cache");
    fs::write(
        home.join("ztools.toml"),
        "osaurus_url = \"http://127.0.0.1:1\"\n",
    )
    .unwrap();

    let out = run_sandboxed(&home, &["twitter-summarize", "--use-cache"]);

    assert_refused(
        "--use-cache with no cache file in the sandboxed HOME",
        &out,
        1,
        &[
            "ztools:",
            "No cached tweets found",
            // The ACTION, not just the fact.
            "Run without --use-cache first",
        ],
    );
}

/// The eval roster is not installed.
///
/// The eval entry point refuses to guess its inputs rather than measuring a task
/// set nobody named. The message has to carry both the paths it tried and the
/// knob that overrides them, or it is a dead end: "no roster" and "your roster
/// is in the wrong place" need the same fix and the operator cannot tell them
/// apart without them.
#[test]
fn an_eval_with_no_roster_installed_exits_1_and_lists_where_it_looked() {
    let home = fresh("refuse-no-roster");
    // A conf dir that EXISTS and holds no `eval_inputs.toml`: the state of a
    // fresh checkout, which is exactly when the message has to be useful.
    let conf = home.join("conf");
    fs::create_dir_all(&conf).unwrap();
    write_config(
        &home,
        &format!(
            "osaurus_url = \"http://127.0.0.1:1\"\n\
             llm_timeout_secs = 5\n\
             eval_conf_dirs = [\"{}\"]\n",
            conf.display()
        ),
    );

    let out = run_sandboxed(
        &home,
        &["model-eval", "--suite", "full", "--model", "mock-nothing"],
    );
    let conf_path = conf.display().to_string();

    assert_refused(
        "an eval run with no eval_inputs.toml under any eval conf dir",
        &out,
        1,
        &[
            "ztools:",
            "no eval_inputs.toml found",
            &conf_path,
            "eval_conf_dirs",
        ],
    );
}

/// The inference server is not answering.
///
/// The refusal an operator meets most: the configured URL points at nothing.
/// What the contract owes them is the URL actually dialled -- "connection
/// refused" alone leaves three candidates (server down, wrong port, wrong
/// machine) and guessing wrong costs a debugging session.
///
/// `--model all` rather than a named model on purpose. Discovery issues
/// `GET /v1/models`, so the failure lands on the FIRST request and the message
/// is about reaching the server rather than about one model being absent -- a
/// distinction worth more here than the shorter test would be.
#[test]
fn an_unreachable_inference_server_exits_1_and_names_the_url_it_dialled() {
    let home = fresh("refuse-no-server");
    let port = closed_loopback_port();
    write_config(
        &home,
        &format!(
            "osaurus_url = \"http://127.0.0.1:{port}\"\n\
             llm_timeout_secs = 5\n\
             llm_quick_timeout_secs = 2\n"
        ),
    );

    let out = run_sandboxed(&home, &["model-eval", "--model", "all"]);
    let dialled = format!("http://127.0.0.1:{port}/v1/models");

    assert_refused(
        "a model-eval against an inference server that is not listening",
        &out,
        1,
        &["ztools:", "error sending request", &dialled],
    );
}

/// The control the four refusals above need: exit 0 has to be REACHABLE in this
/// same sandbox, or "exits 1" is a constant of the harness rather than a
/// finding about the program.
///
/// The image renamer is the one command in this file that can succeed with no
/// network at all, and the unreachable `osaurus_url` above stays in the config
/// deliberately -- the refusals must be about their own cause, not about an
/// incidental clean config.
#[test]
fn a_command_with_nothing_to_refuse_exits_0_in_the_same_sandbox() {
    let home = fresh("refuse-control");
    let port = closed_loopback_port();
    write_config(
        &home,
        &format!("osaurus_url = \"http://127.0.0.1:{port}\"\nllm_timeout_secs = 5\n"),
    );
    let pics = home.join("pics");
    fs::create_dir_all(&pics).unwrap();
    let pics_arg = pics.to_str().unwrap().to_string();

    let out = run_sandboxed(&home, &["image-renamer", &pics_arg]);

    assert_eq!(
        exit_code(&out),
        0,
        "the control must exit 0 with a dead server URL still configured, or \
         the exit codes asserted beside it prove nothing about refusals: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}
