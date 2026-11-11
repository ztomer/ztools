//! Every ztools request to the model server is made under the machine-wide GPU
//! lock (ZT.TWGPULOCK, owner rule: "never a request in the middle of another
//! session's measurement").
//!
//! THE CLASS. A caller of the server that forgets the lock is invisible: the
//! request works. The twitter summary took the lock for its chat and not for
//! its embeddings and roster, and nothing failed. Two instruments close it:
//!
//! 1. [`ENTRY_POINTS`] lists, per source file, every production file that
//!    holds a request primitive (a server URL, `chat`, `chat_with`,
//!    `get_available_models`), and how that file is under the lock. A file
//!    that gains a primitive and is not listed FAILS the scan; so does a
//!    listed file that lost its primitive (a stale entry is a rug).
//! 2. Each entry names the test that DRIVES it against the recording fake
//!    ([`FakeOsaurus::unlocked_requests`]); the name must exist, so an entry
//!    cannot point at nothing. The fake records a request as unlocked when no
//!    session holds the lock at the moment it arrives.
//!
//! The list of files with no excuse is empty: there is no "known violation"
//! row, and adding one is the defect.

use crate::test_env::TestEnv;
use crate::ztools::residency::fake::{FakeOsaurus, Spec};
use serial_test::serial;
use std::path::{Path, PathBuf};

/// One production file that makes requests to the model server.
struct EntryPoint {
    /// Path under `rust/src`.
    file: &'static str,
    /// How its requests are under the lock.
    how: &'static str,
    /// Tests that drive it against the fake and assert no unlocked request.
    covered_by: &'static [&'static str],
}

const ENTRY_POINTS: &[EntryPoint] = &[
    EntryPoint {
        file: "cli_ztools.rs",
        how: "get_available_models takes the lock itself (residency::locked)",
        covered_by: &["the_roster_is_requested_under_the_gpu_lock"],
    },
    EntryPoint {
        file: "ztools/embeddings.rs",
        how: "cluster_tweets takes the lock itself (residency::locked)",
        covered_by: &["clustering_requests_embeddings_under_the_gpu_lock"],
    },
    EntryPoint {
        file: "ztools/eval/transport.rs",
        how: "the eval holds the lock (GpuLockGuard) around its roster",
        covered_by: &["the_spot_eval_loads_its_model_and_leaves_nothing_resident"],
    },
    EntryPoint {
        file: "ztools/llm.rs",
        how: "the primitive: every caller holds the lock (Held) before asking",
        covered_by: &[
            "a_summary_run_makes_no_request_outside_the_gpu_lock",
            "a_weekend_run_makes_no_request_outside_the_gpu_lock",
        ],
    },
    EntryPoint {
        file: "ztools/model_eval.rs",
        how: "eval_model holds the lock; get_available_models takes it (residency::locked)",
        covered_by: &[
            "the_roster_is_requested_under_the_gpu_lock",
            "the_full_suite_loads_its_model_and_leaves_nothing_resident",
        ],
    },
    EntryPoint {
        file: "ztools/rename/vlm.rs",
        how: "the renamer's Session holds the model (Held) before it is asked",
        covered_by: &["a_rename_makes_no_request_outside_the_gpu_lock"],
    },
    EntryPoint {
        file: "ztools/residency.rs",
        how: "the seam itself: load, with_model and locked take the lock",
        covered_by: &["a_summary_run_makes_no_request_outside_the_gpu_lock"],
    },
    EntryPoint {
        file: "ztools/twitter/mod.rs",
        how: "with_model per candidate; the roster and embeddings take the lock",
        covered_by: &["a_summary_run_makes_no_request_outside_the_gpu_lock"],
    },
    EntryPoint {
        file: "ztools/weekend/calls.rs",
        how: "warm_model's Held keeps the lock through the phases",
        covered_by: &["a_weekend_run_makes_no_request_outside_the_gpu_lock"],
    },
    EntryPoint {
        file: "ztools/weekend/phases.rs",
        how: "the roster takes the lock (residency::locked); warm_model holds it",
        covered_by: &["a_weekend_run_makes_no_request_outside_the_gpu_lock"],
    },
];

/// What marks a request to the model server in production source.
const PRIMITIVES: &[&str] = &[
    "/v1/chat/completions",
    "/v1/embeddings",
    "/v1/models",
    "/health\"",
    "chat_with(",
    "llm::chat(",
    "get_available_models(",
];

fn src_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn is_test_or_fake(rel: &str) -> bool {
    rel.ends_with("_tests.rs")
        || rel.ends_with("tests.rs")
        || rel.contains("/tests/")
        || rel.ends_with("fake.rs")
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Whether production code in `text` (comments dropped) holds a primitive.
fn holds_primitive(text: &str) -> bool {
    text.lines()
        .map(str::trim_start)
        .filter(|l| !l.starts_with("//"))
        .any(|l| PRIMITIVES.iter().any(|p| l.contains(p)))
}

/// The production files that hold a primitive, relative to `src`.
fn files_with_primitives(root: &Path) -> Vec<String> {
    let mut all = Vec::new();
    rust_files(root, &mut all);
    let mut found: Vec<String> = all
        .iter()
        .map(|p| p.strip_prefix(root).unwrap().to_string_lossy().into_owned())
        .filter(|rel| !is_test_or_fake(rel))
        .filter(|rel| holds_primitive(&std::fs::read_to_string(root.join(rel)).unwrap()))
        .collect();
    found.sort();
    found
}

#[test]
fn every_file_that_calls_the_model_server_is_listed_with_how_it_is_locked() {
    let found = files_with_primitives(&src_root());
    let listed: Vec<String> = ENTRY_POINTS.iter().map(|e| e.file.to_string()).collect();
    let unlisted: Vec<&String> = found.iter().filter(|f| !listed.contains(f)).collect();
    let stale: Vec<&String> = listed.iter().filter(|f| !found.contains(f)).collect();
    assert!(
        unlisted.is_empty(),
        "these files call the model server and are not in ENTRY_POINTS: {unlisted:?}. Add the \
         entry with the test that proves its requests are made under the GPU lock."
    );
    assert!(
        stale.is_empty(),
        "these ENTRY_POINTS no longer call the model server (delete the row): {stale:?}"
    );
}

#[test]
fn every_entry_point_names_tests_that_exist() {
    let mut sources = String::new();
    let mut all = Vec::new();
    rust_files(&src_root(), &mut all);
    rust_files(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests"),
        &mut all,
    );
    for path in all {
        sources.push_str(&std::fs::read_to_string(path).unwrap());
    }
    for entry in ENTRY_POINTS {
        assert!(
            !entry.how.is_empty() && !entry.covered_by.is_empty(),
            "{}",
            entry.file
        );
        for test in entry.covered_by {
            assert!(
                sources.contains(&format!("fn {test}(")),
                "{} names the test `{test}`, which does not exist",
                entry.file
            );
        }
    }
}

#[test]
fn the_scan_sees_a_new_caller_it_was_not_told_about() {
    assert!(holds_primitive("let url = format!(\"{b}/v1/embeddings\");"));
    assert!(holds_primitive("    crate::ztools::llm::chat_with(&req)"));
    assert!(!holds_primitive(
        "    // POSTs to /v1/embeddings, in a comment"
    ));
}

/// The embeddings request, alone: this is the one the twitter summary made
/// with nobody holding the lock.
#[test]
#[serial]
fn clustering_requests_embeddings_under_the_gpu_lock() {
    let env = TestEnv::new();
    let fake = FakeOsaurus::serve(Spec::chat("{}"));
    let cfg = crate::config::ZtoolsConfig::default();
    let tweet = crate::ztools::twitter::Tweet {
        id: String::new(),
        screen_name: "u".to_string(),
        text: "one tweet about a thing".to_string(),
        created_at: "14:00".to_string(),
        favorite_count: 0,
        retweet_count: 0,
        reply_to: None,
    };

    let clustering = crate::ztools::embeddings::cluster_tweets(&[tweet], &fake.url, &cfg);

    assert!(clustering.status_line().contains("clustering"));
    assert!(!fake.calls().is_empty(), "the request must have been made");
    assert_eq!(fake.unlocked_requests(), Vec::<String>::new());
    drop(env);
}

/// The roster, alone: the capabilities command and the twitter chain both ask.
#[test]
#[serial]
fn the_roster_is_requested_under_the_gpu_lock() {
    let env = TestEnv::new();
    let fake = FakeOsaurus::serve(Spec {
        models: r#"{"data":[{"id":"some-model"}]}"#.to_string(),
        ..Spec::chat("{}")
    });
    let cfg = crate::config::ZtoolsConfig::default();

    let roster = crate::ztools::model_eval::get_available_models(&fake.url, &cfg).unwrap();

    assert_eq!(roster, vec!["some-model".to_string()]);
    assert_eq!(fake.unlocked_requests(), Vec::<String>::new());
    drop(env);
}

/// A lock another live session holds is waited for, not stepped over: the
/// request is not made, and the reason names the holder.
#[test]
#[serial]
fn a_request_waits_for_a_lock_another_session_holds() {
    let env = TestEnv::new();
    let fake = FakeOsaurus::serve(Spec::chat("{}"));
    let dir = crate::ztools::eval::gpu_lock::lock_dir();
    std::fs::create_dir_all(&dir).unwrap();
    // A live pid, with the inheritance marker cleared by `TestEnv`: to the
    // lock this is another session holding it.
    let me = std::process::id();
    std::fs::write(
        dir.join("owner"),
        format!(
            "{me}\n{}\nanother session (pid {me})\n",
            crate::ztools::eval::gpu_lock::start_time(me)
        ),
    )
    .unwrap();
    let timing = crate::ztools::residency::Timing {
        lock_wait: std::time::Duration::from_millis(200),
        ..crate::ztools::residency::Timing::default()
    };

    let outcome = crate::ztools::residency::locked("test", timing, || {
        crate::ztools::model_eval::get_available_models(
            &fake.url,
            &crate::config::ZtoolsConfig::default(),
        )
    });

    let err = outcome.expect_err("a foreign holder is waited for, then refused");
    assert!(format!("{err:#}").contains("another session"), "{err:#}");
    assert!(fake.calls().is_empty(), "{:?}", fake.calls());
    drop(env);
}
