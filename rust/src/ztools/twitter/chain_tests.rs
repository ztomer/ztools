use super::*;
use crate::test_env::TestEnv;
use std::io::Write as _;

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| (*x).to_string()).collect()
}

fn policy() -> FallbackPolicy {
    FallbackPolicy {
        models: s(&["foundation"]),
        preferred: s(&["foundation", "qwen", "gemma"]),
    }
}

fn write_conf(dir: &std::path::Path, body: &str) -> String {
    let path = dir.join("twitter.toml");
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(body.as_bytes()).unwrap();
    path.to_string_lossy().into_owned()
}

#[test]
#[serial_test::serial]
fn policy_loads_from_the_first_file_with_the_table() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir
        .path()
        .join("absent.toml")
        .to_string_lossy()
        .into_owned();
    let good = write_conf(
        dir.path(),
        "[fallback]\nmodels = [\"foundation\", \" spare \"]\npreferred = [\"qwen\"]\n",
    );
    // No `unset` call: the sandbox CLEARS every policy knob on construction, so
    // `FALLBACK_MODELS_ENV` is genuinely absent here and an operator's exported
    // override cannot decide which file wins.
    let _env = TestEnv::new();
    let got = load_fallback_policy(&[missing, good]).unwrap();
    assert_eq!(got.models, s(&["foundation", "spare"]));
    assert_eq!(got.preferred, s(&["qwen"]));
}

#[test]
#[serial_test::serial]
fn policy_is_required_not_defaulted() {
    let dir = tempfile::tempdir().unwrap();
    let no_table = write_conf(dir.path(), "model = \"x\"\n");
    // Same: the override is absent because the sandbox cleared it, so "no table
    // anywhere" is the file's doing and not an environment accident.
    let _env = TestEnv::new();
    let err = load_fallback_policy(&[no_table]).unwrap_err().to_string();
    assert!(err.contains("[fallback]"), "{err}");
}

#[test]
#[serial_test::serial]
fn env_override_replaces_the_models_list_only() {
    let dir = tempfile::tempdir().unwrap();
    let good = write_conf(
        dir.path(),
        "[fallback]\nmodels = [\"foundation\"]\npreferred = [\"qwen\"]\n",
    );
    let env = TestEnv::new();
    env.set(FALLBACK_MODELS_ENV, "alpha, beta,,");
    let got = load_fallback_policy(std::slice::from_ref(&good));
    // Back to absent before the next case: an empty override is no override.
    env.unset(FALLBACK_MODELS_ENV);
    let got = got.unwrap();
    assert_eq!(got.models, s(&["alpha", "beta"]));
    assert_eq!(got.preferred, s(&["qwen"]));
    env.set(FALLBACK_MODELS_ENV, " , ");
    let got = load_fallback_policy(&[good]);
    env.unset(FALLBACK_MODELS_ENV);
    assert_eq!(got.unwrap().models, s(&["foundation"]));
    drop(env);
}

#[test]
fn shipped_conf_carries_a_fallback_policy() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("conf/twitter.toml");
    let content = std::fs::read_to_string(path).unwrap();
    let val: toml::Value = toml::from_str(&content).unwrap();
    let table = val
        .get("fallback")
        .expect("conf/twitter.toml needs [fallback]");
    assert_nonempty!(string_list(table.get("models")));
    assert_nonempty!(string_list(table.get("preferred")));
}

#[test]
fn plan_keeps_a_served_target_and_appends_extras() {
    let chain = plan_chain("gemma-x", &s(&["gemma-x", "qwen-y"]), &policy());
    assert_eq!(chain, s(&["gemma-x", "foundation"]));
}

#[test]
fn plan_substitutes_an_unserved_target_by_preference() {
    let chain = plan_chain("gone", &s(&["other", "qwen-y"]), &policy());
    assert_eq!(chain, s(&["qwen-y", "foundation"]));
}

#[test]
fn plan_with_unknown_roster_keeps_the_intent_unfiltered() {
    let chain = plan_chain("gemma-x", &[], &policy());
    assert_eq!(chain, s(&["gemma-x", "foundation"]));
}

#[test]
fn first_usable_answer_from_the_intended_model_is_primary() {
    let chain = s(&["a", "b"]);
    let (got, prov) = run_chain(&chain, "a", |m| Ok(Some(m.to_uppercase()))).unwrap();
    assert_eq!(got, "A");
    assert_eq!(prov.tier, Tier::Primary);
    assert!(!prov.degraded());
    assert_empty!(&prov.reasons);
    assert_eq!(prov.banner(), "**Model:** a (osaurus, primary)");
}

#[test]
fn a_hollow_answer_moves_on_and_is_recorded() {
    let chain = s(&["a", "b"]);
    let (got, prov) = run_chain(&chain, "a", |m| {
        Ok(if m == "a" { None } else { Some(m.to_string()) })
    })
    .unwrap();
    assert_eq!(got, "b");
    assert_eq!(prov.tier, Tier::Fallback);
    assert_eq!(
        prov.reasons,
        s(&[
            "model a returned no usable summary",
            "answered by b instead of the intended a"
        ])
    );
    let banner = prov.banner();
    assert!(banner.starts_with("> ⚠ **DEGRADED OUTPUT**"));
    assert!(banner.contains("> **Why:** model a returned no usable summary"));
    assert!(banner.contains("> **Backend:** b (osaurus, fallback)"));
}

#[test]
fn a_transport_failure_moves_on_with_the_error_text() {
    let chain = s(&["a", "b"]);
    let ((), prov) = run_chain(&chain, "a", |m| {
        if m == "a" {
            Err(anyhow::anyhow!("connection refused"))
        } else {
            Ok(Some(()))
        }
    })
    .unwrap();
    assert_eq!(prov.reasons[0], "model a failed: connection refused");
}

#[test]
fn substituted_target_that_answers_first_is_still_degraded() {
    // The intended model was not served; the substitute answered first time.
    // That is not a primary run — the artifact must say which model wrote it.
    let chain = s(&["sub"]);
    let ((), prov) = run_chain(&chain, "intended", |_| Ok(Some(()))).unwrap();
    assert_eq!(prov.tier, Tier::Fallback);
    assert_eq!(
        prov.reasons,
        s(&["answered by sub instead of the intended intended"])
    );
}

#[test]
fn exhausting_the_chain_names_every_attempt() {
    let chain = s(&["a", "b"]);
    let err = run_chain::<()>(&chain, "a", |_| Ok(None))
        .unwrap_err()
        .to_string();
    assert!(err.contains("a → b"), "{err}");
    assert!(err.contains("model a returned no usable summary"));
    assert!(err.contains("model b returned no usable summary"));
}

#[test]
fn an_empty_chain_is_its_own_error() {
    let err = run_chain::<()>(&[], "a", |_| Ok(Some(())))
        .unwrap_err()
        .to_string();
    assert!(err.contains("chain is empty"), "{err}");
}
