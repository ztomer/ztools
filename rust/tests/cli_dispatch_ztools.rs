//! CLI dispatch tests for the ztools subcommands: `twitter-summarize`,
//! `weekend-plan`, `image-renamer` and `model-eval`.
//!
//! These commands talk to a local LLM and a web search endpoint, so each test
//! stands up a stub HTTP server on a free port and points the `--config` TOML
//! at it. `HOME` is redirected too: the summarizer writes into `~/Documents`
//! and reads `~/.cache`, and a test must never touch either of those for real.
//!
//! Ported from `routines/tests/cli_dispatch_ztools.rs` when the ztools modules
//! moved into their own crate. The config seam changed from routines'
//! `ROUTINES_HOME` + `[ztools]` block to this binary's `--config` flag.

use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::Command;
use std::thread;

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_ztools"))
}

fn fresh(name: &str) -> PathBuf {
    let mut d = std::env::temp_dir();
    d.push(format!("ztools-cli-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// Run the binary with `HOME` inside the sandbox and the stub `--config`.
fn ztool(home: &std::path::Path) -> Command {
    let mut c = Command::new(bin());
    c.env("HOME", home).arg("--config").arg(home.join("ztools.toml"));
    c
}

/// Write the flat `ZtoolsConfig` TOML the `--config` flag loads.
fn write_config(home: &std::path::Path, content: &str) {
    fs::write(home.join("ztools.toml"), content).unwrap();
}

fn stdout_of(out: &std::process::Output) -> String {
    assert!(
        out.status.success(),
        "command failed — stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// A stub server that answers every request with `body` for the life of the
/// test. Requests arrive concurrently (the planner fans its searches out over
/// threads), so it must keep serving rather than answering once and stopping.
fn stub_server(body: &'static str) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let body = body.to_string();
            thread::spawn(move || {
                let mut stream = stream;
                let mut buf = [0u8; 16384];
                let _ = stream.read(&mut buf);
                let http = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(http.as_bytes());
                let _ = stream.flush();
            });
        }
    });
    thread::sleep(std::time::Duration::from_millis(50));
    port
}

const LLM_TEXT: &str = r####"{"choices":[{"message":{"content":"## Highlights\n- one thing happened\n- another thing happened\n- a third thing happened"}}]}"####;

const LLM_EVENTS: &str = r####"{"choices":[{"message":{"content":"{\"transient_events\":[{\"name\":\"Rib Fest\",\"location\":\"Vaughan Park\",\"target_ages\":\"6-12\",\"price\":\"Free\",\"day\":\"Saturday\",\"description\":\"An outdoor festival for kids\"},{\"name\":\"Bare Listing\",\"location\":\"Vaughan\",\"target_ages\":\"all\",\"price\":\"Free\",\"day\":\"Sunday\",\"description\":\"\"}]}"}}]}"####;

/// The planner's other real outcome: the model found nothing for this weekend.
const LLM_NO_EVENTS: &str =
    r####"{"choices":[{"message":{"content":"{\"transient_events\":[]}"}}]}"####;

#[test]
fn twitter_summarize_writes_a_summary_and_an_md_copy() {
    let home = fresh("twitter");
    let port = stub_server(LLM_TEXT);
    write_config(
        &home,
        &format!("osaurus_url = \"http://127.0.0.1:{port}\"\nllm_timeout_secs = 10\n"),
    );

    // Tweets must be supplied: with an empty list the summarizer falls back to
    // reading the configured cache, and with none the run must fail rather than
    // fabricate a document.
    let tweets = home.join("tweets.json");
    fs::write(
        &tweets,
        r#"[{"screen_name":"john_doe","text":"Just launched the new API!",
             "created_at":"2026-08-01","favorite_count":10,"retweet_count":2,"reply_to":null},
           {"screen_name":"jane_smith","text":"The new API is incredibly fast.",
             "created_at":"2026-08-02","favorite_count":5,"retweet_count":1,"reply_to":null}]"#,
    )
    .unwrap();

    let md_out = home.join("summary-copy.md");
    let out = Command::new(bin())
        .env("HOME", &home)
        .arg("--config")
        .arg(home.join("ztools.toml"))
        .arg("twitter-summarize")
        .arg("--json")
        .arg(&tweets)
        .arg("--md-out")
        .arg(&md_out)
        .output()
        .unwrap();

    let stdout = stdout_of(&out);
    assert!(stdout.contains("twitter summary generated at"), "{stdout}");
    assert!(stdout.contains("copy saved to"), "{stdout}");
    assert!(md_out.exists(), "--md-out copy was not written");

    let doc = fs::read_to_string(&md_out).unwrap();
    assert!(doc.contains("Twitter Timeline Summary"), "{doc}");
    assert!(doc.contains("2 fetched"), "both tweets should count: {doc}");
    assert!(doc.contains("Highlights"), "LLM body missing: {doc}");
    // The write must land under the redirected HOME, never the real one.
    assert!(
        home.join("Documents/twitter_summaries").is_dir(),
        "summary was written outside the sandboxed HOME"
    );
}

/// An unreadable or absent tweets file is not fatal: the command still runs,
/// it just has nothing of its own to summarize.
#[test]
fn twitter_summarize_tolerates_a_missing_tweets_file() {
    let home = fresh("twitter-nofile");
    let port = stub_server(LLM_TEXT);
    write_config(
        &home,
        &format!("osaurus_url = \"http://127.0.0.1:{port}\"\nllm_timeout_secs = 10\n"),
    );
    let out = Command::new(bin())
        .env("HOME", &home)
        .arg("--config")
        .arg(home.join("ztools.toml"))
        .arg("twitter-summarize")
        .arg("--json")
        .arg(home.join("does-not-exist.json"))
        .arg("--model")
        .arg("stub-model")
        .output()
        .unwrap();
    let stdout = stdout_of(&out);
    assert!(stdout.contains("twitter summary generated at"), "{stdout}");
    assert!(
        stdout.contains("stub-model"),
        "the --model override should reach the document: {stdout}"
    );
}

#[test]
fn weekend_plan_renders_and_writes_the_markdown() {
    let home = fresh("weekend");
    // One stub answers both the search fan-out and the LLM extraction: the
    // search parser simply finds no snippets in a JSON body, which is the same
    // shape as a search that returned nothing useful.
    let port = stub_server(LLM_EVENTS);
    write_config(
        &home,
        &format!(
            "osaurus_url = \"http://127.0.0.1:{port}\"\n\
             duckduckgo_url = \"http://127.0.0.1:{port}/\"\n\
             llm_timeout_secs = 10\n"
        ),
    );

    let md_out = home.join("weekend.md");
    let out = ztool(&home)
        .arg("weekend-plan")
        .arg("--location")
        .arg("Vaughan")
        .arg("--ages")
        .arg("6,12")
        .arg("--md-out")
        .arg(&md_out)
        .output()
        .unwrap();

    let stdout = stdout_of(&out);
    assert!(stdout.contains("saved to"), "{stdout}");
    assert!(stdout.contains("Weekend Plan"), "{stdout}");
    assert!(md_out.exists(), "--md-out plan was not written");

    let doc = fs::read_to_string(&md_out).unwrap();
    assert!(doc.contains("Vaughan"), "{doc}");
    assert!(
        doc.contains("Rib Fest"),
        "the extracted event should reach the plan: {doc}"
    );
}

#[test]
fn image_renamer_dry_run_proposes_names_without_touching_files() {
    let home = fresh("img-dry");
    write_config(&home, "osaurus_url = \"http://127.0.0.1:1\"\n");
    let pics = home.join("pics");
    fs::create_dir_all(&pics).unwrap();
    fs::write(pics.join("My Vacation Photo.PNG"), b"x").unwrap();
    fs::write(pics.join("notes.txt"), b"not an image").unwrap();

    let stdout = stdout_of(&ztool(&home).arg("image-renamer").arg(&pics).output().unwrap());
    assert!(stdout.contains("DRY-RUN"), "{stdout}");
    assert!(stdout.contains("1 file(s) processed"), "{stdout}");
    assert!(stdout.contains("my_vacation_photo.png"), "{stdout}");
    assert!(
        pics.join("My Vacation Photo.PNG").exists(),
        "a dry run must not rename anything"
    );
}

#[test]
fn image_renamer_apply_renames_the_files() {
    let home = fresh("img-apply");
    write_config(&home, "osaurus_url = \"http://127.0.0.1:1\"\n");
    let pics = home.join("pics");
    fs::create_dir_all(&pics).unwrap();
    fs::write(pics.join("Beach Day.JPG"), b"x").unwrap();

    let stdout = stdout_of(
        &ztool(&home)
            .arg("image-renamer")
            .arg(&pics)
            .arg("--apply")
            .output()
            .unwrap(),
    );
    assert!(stdout.contains("APPLIED"), "{stdout}");
    assert!(pics.join("beach_day.jpg").exists(), "file was not renamed");
    assert!(!pics.join("Beach Day.JPG").exists(), "old name still there");
}

/// An empty directory is not an error: nothing to name, so it reports the
/// no-op and exits cleanly.
#[test]
fn image_renamer_dry_run_on_empty_dir() {
    let home = fresh("img-empty");
    write_config(&home, "osaurus_url = \"http://127.0.0.1:1\"\n");
    let pics = home.join("pics");
    fs::create_dir_all(&pics).unwrap();

    let stdout = stdout_of(&ztool(&home).arg("image-renamer").arg(&pics).output().unwrap());
    assert!(stdout.contains("0 file(s) processed"), "{stdout}");
    assert!(stdout.contains("DRY-RUN"), "{stdout}");
}

/// `model-eval` defaults to `all`, which goes through model discovery first.
/// Every other test passes an explicit model and skips that branch entirely.
#[test]
fn model_eval_all_evaluates_every_discovered_model() {
    let home = fresh("model-eval-all");
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            thread::spawn(move || {
                let mut stream = stream;
                let mut buf = [0u8; 16384];
                let n = stream.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]).into_owned();
                let body = if req.contains("GET /v1/models") {
                    r#"{"data":[{"id":"stub-a"},{"id":"foundation-skip"}]}"#.to_string()
                } else {
                    LLM_TEXT.to_string()
                };
                let http = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(http.as_bytes());
                let _ = stream.flush();
            });
        }
    });
    thread::sleep(std::time::Duration::from_millis(50));

    write_config(
        &home,
        &format!(
            "osaurus_url = \"http://127.0.0.1:{port}\"\n\
             llm_timeout_secs = 10\nllm_quick_timeout_secs = 5\n"
        ),
    );

    let stdout = stdout_of(&ztool(&home).arg("model-eval").output().unwrap());
    assert!(stdout.contains("Model Quality Evaluation"), "{stdout}");
    assert!(stdout.contains("stub-a"), "{stdout}");
    assert!(
        !stdout.contains("foundation-skip"),
        "foundation models must be filtered out of discovery: {stdout}"
    );
}

/// A weekend with nothing on must say so, not render an empty table. This is
/// the common outcome in a quiet week and it has to read as an answer.
#[test]
fn weekend_plan_says_so_when_nothing_is_on() {
    let home = fresh("weekend-empty");
    let port = stub_server(LLM_NO_EVENTS);
    write_config(
        &home,
        &format!(
            "osaurus_url = \"http://127.0.0.1:{port}\"\n\
             duckduckgo_url = \"http://127.0.0.1:{port}/\"\n\
             llm_timeout_secs = 10\n"
        ),
    );
    let md_out = home.join("weekend.md");
    let out = ztool(&home).arg("weekend-plan").arg("--md-out").arg(&md_out).output().unwrap();
    stdout_of(&out);
    let doc = fs::read_to_string(&md_out).unwrap();
    assert!(
        doc.contains("No transient events scheduled"),
        "an empty weekend must be stated: {doc}"
    );
}