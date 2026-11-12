use super::{finish_night, installed_names, replay};
use std::collections::BTreeMap;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tests/fixtures/select/replay.json"
);

#[test]
fn the_recorded_replay_reproduces_and_records_itself_back() {
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("again.json");
    let none = BTreeMap::new();
    replay(Some(&out), Some(FIXTURE.as_ref()), &none, &BTreeMap::new()).unwrap();
    let a: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(FIXTURE).unwrap()).unwrap();
    let b: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    assert_eq!(a, b);
}

#[test]
fn a_table_the_history_cannot_replay_exits_one() {
    let tmp = tempfile::tempdir().unwrap();
    let mut v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(FIXTURE).unwrap()).unwrap();
    v["best_models"]["think"] = "not-installed".into();
    let bad = tmp.path().join("bad.json");
    std::fs::write(&bad, v.to_string()).unwrap();
    let err = replay(None, Some(&bad), &BTreeMap::new(), &BTreeMap::new()).unwrap_err();
    let exit = err.downcast_ref::<crate::cli::Exit>().unwrap();
    assert_eq!(exit.code, 1);
}

#[test]
fn foundation_is_always_installed() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("org/Model-4bit");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), "{}").unwrap();
    assert_eq!(
        installed_names(tmp.path()),
        vec!["foundation", "model-4bit"]
    );
}

#[test]
fn a_night_whose_scout_reaches_nothing_writes_its_findings_line() {
    let tmp = tempfile::tempdir().unwrap();
    let state = tmp.path().join("state");
    let mut live = crate::ztools::select::live::LiveNight {
        hf: "http://127.0.0.1:9".into(),
        osaurus_url: "http://127.0.0.1:9".into(),
        slots: std::iter::once(("think".to_string(), "m".to_string())).collect(),
        current: BTreeMap::new(),
        models_root: tmp.path().join("MLXModels"),
        state_dir: state.clone(),
        exe: "/nonexistent".into(),
    };
    let _ = finish_night(&mut live, &state);
    let _ = finish_night(&mut live, &state);
    let text = std::fs::read_to_string(state.join("findings.md")).unwrap();
    assert_eq!(text.lines().count(), 2, "{text}");
    assert!(text.lines().all(|l| l.starts_with("LM ")), "{text}");
}
