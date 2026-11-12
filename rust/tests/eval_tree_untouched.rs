//! A RUN NEVER DIRTIES THE TREE (ZT.EVALSIGNALS).
//!
//! `ztools model-eval` loaded `conf/eval_signals.json` from the checkout that
//! built the binary and saved its learned rates back to it, so every scheduled
//! models run left a tracked file modified and blocked the next ship. A run's
//! output is STATE (`~/.local/state/ztools/eval/`); the tracked file is the
//! shipped seed it starts from on a machine with no state of its own.
//!
//! This file is the class gate, not the one instance: it runs every CLI path
//! that writes learned data (the spot eval, the full suite, the signal-store
//! prune) from a sandbox `HOME` with NO `EVAL_SIGNALS_DIR` -- the production
//! default, which the other end-to-end files override away -- and asserts the
//! checkout's `git status` and the tracked seed's bytes are exactly what they
//! were before. A run that writes anywhere else under the repo shows up in the
//! same `git status`, whatever file it picks.
//!
//! The cases are `#[serial]`: they all compare the same checkout's status.
//!
//! If a case fails, the guard puts the tracked seed's bytes back, so a red run
//! of this test does not itself leave the tree dirty.

// Refuse a stale build of this file (ZT.MTIMECLASS).
ztools::built_from_disk!();

#[path = "support/mod.rs"]
mod support;
pub use support::*;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serial_test::serial;
use ztools::ztools::residency::fake::{FakeOsaurus, Spec};

const ANSWER: &str =
    r#"{"choices":[{"message":{"content":"{\"ok\": true}"},"finish_reason":"stop"}]}"#;

/// 64 GB available, no swap: the oversize guard reads this, not the host.
const MEMINFO_ROOMY: &str = "MemTotal: 67108864 kB\nMemFree: 60000000 kB\nMemAvailable: 64000000 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n";

fn checkout() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the checkout that holds rust/")
        .to_path_buf()
}

fn seed_path() -> PathBuf {
    checkout().join("conf/eval_signals.json")
}

/// `git status --porcelain` of the whole checkout.
fn tree_status() -> String {
    let out = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=all"])
        .current_dir(checkout())
        .output()
        .expect("run git status");
    assert!(out.status.success(), "git status failed");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Restores the tracked seed's bytes if the test panics after a run rewrote it.
struct RestoreSeed(Vec<u8>);

impl Drop for RestoreSeed {
    fn drop(&mut self) {
        if std::thread::panicking() {
            let _ = fs::write(seed_path(), &self.0);
        }
    }
}

fn sandbox(name: &str, fake: &FakeOsaurus) -> PathBuf {
    assert_built_from_disk(file!(), include_str!("eval_tree_untouched.rs"));
    let home = fresh(name);
    fake.watching_lock(&home.join("gpu-lock"));
    let conf = home.join("conf");
    fs::create_dir_all(&conf).unwrap();
    for file in ["eval_inputs.toml", "eval_vision.toml"] {
        fs::copy(checkout().join("conf").join(file), conf.join(file)).unwrap();
    }
    write_config(
        &home,
        &format!(
            "osaurus_url = \"{}\"\nllm_timeout_secs = 5\nllm_quick_timeout_secs = 2\n\
             eval_conf_dirs = [\"{}\"]\n",
            fake.url,
            conf.display()
        ),
    );
    fs::write(home.join("meminfo"), MEMINFO_ROOMY).unwrap();
    fs::create_dir_all(home.join("gpu-lock")).unwrap();
    home
}

/// The binary with a sandbox `HOME` and NOTHING redirecting the signal store.
fn run(home: &Path, args: &[&str]) -> Output {
    Command::new(bin())
        .env("HOME", home)
        .env("ZTOOLS_MEMINFO", home.join("meminfo"))
        .env("ZTOOLS_GPU_LOCK_DIR", home.join("gpu-lock"))
        .env_remove("EVAL_SIGNALS_DIR")
        .env_remove("ZTOOLS_CONF_DIR")
        .env_remove("XDG_STATE_HOME")
        .env_remove("EVAL_ALLOW_OVERSIZE")
        .arg("--config")
        .arg(home.join("ztools.toml"))
        .args(args)
        .output()
        .unwrap()
}

fn state_store(home: &Path) -> PathBuf {
    home.join(".local/state/ztools/eval/eval_signals.json")
}

fn model_names(path: &Path) -> Vec<String> {
    let v: serde_json::Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    v.as_object().unwrap().keys().cloned().collect()
}

#[test]
#[serial]
fn an_eval_run_writes_its_signals_to_state_and_leaves_the_tree_alone() {
    let before_status = tree_status();
    let seed_bytes = fs::read(seed_path()).unwrap();
    let _restore = RestoreSeed(seed_bytes.clone());
    let seed_models = model_names(&seed_path());
    assert!(!seed_models.is_empty(), "control: the seed holds models");

    let fake = FakeOsaurus::serve(Spec {
        models: r#"{"data":[{"id":"probe-model"}]}"#.to_string(),
        ..Spec::chat(ANSWER)
    });
    let home = sandbox("tree-untouched-full", &fake);
    let out = run(
        &home,
        &[
            "model-eval",
            "--suite",
            "full",
            "--model",
            "probe-model",
            "--task",
            "json",
        ],
    );
    assert!(
        out.status.success(),
        "the run itself must work: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    assert!(
        fs::read(seed_path()).unwrap() == seed_bytes,
        "the tracked conf/eval_signals.json is input only: a run rewrote it"
    );
    assert_eq!(
        tree_status(),
        before_status,
        "a model-eval run changed the checkout's git status"
    );

    let store = state_store(&home);
    assert!(
        store.exists(),
        "the run's signals belong in state: nothing at {}",
        store.display()
    );
    let kept = model_names(&store);
    assert!(
        kept.contains(&"probe-model".to_string()),
        "the run's own model is recorded in state: {kept:?}"
    );
    for m in &seed_models {
        assert!(
            kept.contains(m),
            "the state store starts from the shipped seed and loses nothing: missing {m}"
        );
    }
}

#[test]
#[serial]
fn the_spot_eval_and_the_prune_leave_the_tree_alone_too() {
    let before_status = tree_status();
    let seed_bytes = fs::read(seed_path()).unwrap();
    let _restore = RestoreSeed(seed_bytes.clone());

    let fake = FakeOsaurus::start(ANSWER);
    let home = sandbox("tree-untouched-spot", &fake);

    let out = run(&home, &["model-eval", "--model", "probe-model"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let out = run(&home, &["eval-signals", "--prune"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("Saved to")
            && String::from_utf8_lossy(&out.stdout).contains(".local/state/ztools/eval"),
        "the prune says it saved into state: {}",
        String::from_utf8_lossy(&out.stdout)
    );

    assert!(
        fs::read(seed_path()).unwrap() == seed_bytes,
        "the prune rewrote the tracked seed"
    );
    assert_eq!(
        tree_status(),
        before_status,
        "a spot eval or prune changed the checkout's git status"
    );
}
