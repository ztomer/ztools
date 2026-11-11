//! Every Osaurus caller loads its model and leaves nothing resident
//! (ZT.OSAURUSLOAD): the eval paths, against the recording Osaurus fake.
//!
//! The fake (`ztools::ztools::residency::fake`) records each one-token load,
//! each completion and each `/health` answer, so "loaded before it was asked"
//! and "the run ended with `/health` showing nothing resident" are facts about
//! requests, not about timing.

// Refuse a stale build of this file (ZT.MTIMECLASS).
ztools::built_from_disk!();

#[path = "support/mod.rs"]
mod support;
pub use support::*;

use std::fs;
use std::path::Path;
use std::process::Output;

use ztools::ztools::residency::fake::{Call, FakeOsaurus, Spec};

const ANSWER: &str =
    r#"{"choices":[{"message":{"content":"{\"ok\": true}"},"finish_reason":"stop"}]}"#;

/// The GPU lock directory the spawned binary holds and the fake watches: ONE
/// derivation, so the two cannot name different directories.
fn gpu_lock(home: &Path) -> std::path::PathBuf {
    home.join("gpu-lock")
}

fn sandbox(name: &str, fake: &FakeOsaurus) -> std::path::PathBuf {
    // A stale build of this file is the one way the fake and the binary have
    // disagreed about the lock (ZT.RESIDFLAKE): refuse it by name.
    assert_built_from_disk(file!(), include_str!("residency_callers.rs"));
    let home = fresh(name);
    // The lock the spawned binary will hold (see `run`): the fake records any
    // request that reaches it while that lock is free (ZT.TWGPULOCK).
    fake.watching_lock(&gpu_lock(&home));
    let conf = home.join("conf");
    fs::create_dir_all(&conf).unwrap();
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    for file in ["eval_inputs.toml", "eval_vision.toml"] {
        fs::copy(checkout.join("conf").join(file), conf.join(file)).unwrap();
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
    home
}

/// A pinned memory reading: 64 GB available, no swap. The oversize guard
/// reads it instead of the host, so whether the eval runs does not depend on
/// the RAM of the machine the test runs on (CI run 43031264189 refused
/// `probe-model` on a runner with 4 GB reclaimable).
const MEMINFO_ROOMY: &str = "MemTotal: 67108864 kB\nMemFree: 60000000 kB\nMemAvailable: 64000000 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n";

fn run(home: &Path, args: &[&str]) -> Output {
    let gpu = gpu_lock(home);
    fs::create_dir_all(&gpu).unwrap();
    let meminfo = home.join("meminfo");
    fs::write(&meminfo, MEMINFO_ROOMY).unwrap();
    std::process::Command::new(bin())
        .env("HOME", home)
        .env("ZTOOLS_MEMINFO", &meminfo)
        .env("ZTOOLS_GPU_LOCK_DIR", &gpu)
        .env("EVAL_SIGNALS_DIR", home.join("signals"))
        .env("EVAL_OUTPUT_DIR", home.join("outputs"))
        .env_remove("EVAL_ALLOW_OVERSIZE")
        .arg("--config")
        .arg(home.join("ztools.toml"))
        .args(args)
        .output()
        .unwrap()
}

/// The model was loaded before its first completion, and the last `/health`
/// showed nothing resident.
fn assert_loaded_then_released(fake: &FakeOsaurus, model: &str, out: &Output) {
    let calls = fake.calls();
    let load = calls.iter().position(|c| *c == Call::Load(model.into()));
    let chat = calls.iter().position(|c| *c == Call::Chat(model.into()));
    assert!(
        matches!((load, chat), (Some(l), Some(c)) if l < c),
        "{model} must be loaded before it is asked: {calls:?}\nstderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(fake.last_health(), Some(vec![]), "left resident: {calls:?}");
    assert_eq!(
        fake.unlocked_requests(),
        Vec::<String>::new(),
        "every request is made while the GPU lock is held: {calls:?}"
    );
}

#[test]
fn the_spot_eval_loads_its_model_and_leaves_nothing_resident() {
    let fake = FakeOsaurus::start(ANSWER);
    let home = sandbox("residency-spot", &fake);
    let out = run(&home, &["model-eval", "--model", "probe-model"]);
    assert_loaded_then_released(&fake, "probe-model", &out);
}

#[test]
fn the_full_suite_loads_its_model_and_leaves_nothing_resident() {
    let fake = FakeOsaurus::serve(Spec {
        models: r#"{"data":[{"id":"probe-model"}]}"#.to_string(),
        ..Spec::chat(ANSWER)
    });
    let home = sandbox("residency-full", &fake);
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
    assert_loaded_then_released(&fake, "probe-model", &out);
    // The cold start is the seam's load, not a second one-token call: the
    // eval's own LOAD would have found the model already resident.
    let loads = fake
        .calls()
        .iter()
        .filter(|c| matches!(c, Call::Load(_)))
        .count();
    assert_eq!(
        loads,
        2,
        "one seam load plus the prefill probe's one-token PROBE: {:?}",
        fake.calls()
    );
}
