//! Every Osaurus caller loads its model and leaves nothing resident
//! (ZT.OSAURUSLOAD): the eval paths, against the recording Osaurus fake.
//!
//! The fake (`ztools::ztools::residency::fake`) records each one-token load,
//! each completion and each `/health` answer, so "loaded before it was asked"
//! and "the run ended with `/health` showing nothing resident" are facts about
//! requests, not about timing.

#[path = "support/mod.rs"]
mod support;
pub use support::*;

use std::fs;
use std::path::Path;
use std::process::Output;

use ztools::ztools::residency::fake::{Call, FakeOsaurus, Spec};

const ANSWER: &str =
    r#"{"choices":[{"message":{"content":"{\"ok\": true}"},"finish_reason":"stop"}]}"#;

fn sandbox(name: &str, fake: &FakeOsaurus) -> std::path::PathBuf {
    let home = fresh(name);
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

fn run(home: &Path, args: &[&str]) -> Output {
    let gpu = home.join("gpu-lock");
    fs::create_dir_all(&gpu).unwrap();
    std::process::Command::new(bin())
        .env("HOME", home)
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
    let probe_ones = fake.calls().len();
    assert!(probe_ones > 0);
    assert_eq!(
        loads,
        2,
        "one seam load plus the prefill probe's one-token PROBE: {:?}",
        fake.calls()
    );
}
