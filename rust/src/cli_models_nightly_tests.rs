use super::{finish_night, installed_names, replay};
use crate::ztools::select::nightly::Outcome;
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
    // The night reads the GPU lock: without the sandbox it reads a peer
    // test's `ZTOOLS_GPU_LOCK_DIR` (or the machine's real lock).
    let _env = crate::test_env::TestEnv::new();
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
        osaurus_version: None,
    };
    let _ = finish_night(&mut live, &state);
    let _ = finish_night(&mut live, &state);
    let text = std::fs::read_to_string(state.join("findings.md")).unwrap();
    assert_eq!(text.lines().count(), 2, "{text}");
    assert!(text.lines().all(|l| l.starts_with("LM ")), "{text}");
}

fn decided_night(dir: &std::path::Path) -> (crate::ztools::select::live::LiveNight, Outcome) {
    use crate::ztools::select::Decision;
    use crate::ztools::select::nightly::Staged;
    let live = crate::ztools::select::live::LiveNight {
        hf: "http://127.0.0.1:9".into(),
        osaurus_url: "http://127.0.0.1:9".into(),
        slots: std::iter::once(("json".to_string(), "old".to_string())).collect(),
        current: BTreeMap::new(),
        models_root: dir.join("MLXModels"),
        state_dir: dir.join(".local/state/ztools/models"),
        exe: "/nonexistent".into(),
        osaurus_version: None,
    };
    live.write_ledger(&Staged {
        repo: "org/Cand".into(),
        model: "cand".into(),
        slots: vec!["json".into()],
        bytes: 1,
        date: "d".into(),
    })
    .unwrap();
    let outcome = Outcome::Decided {
        candidate: "cand".into(),
        decisions: vec![Decision {
            slot: "json".into(),
            incumbent: "old".into(),
            chosen: "cand".into(),
            judged: vec![],
        }],
        deleted: false,
    };
    (live, outcome)
}

#[test]
fn a_night_that_chose_a_switch_only_proposes_it_by_default() {
    use crate::ztools::select::nightly::Night as _;
    let sandbox = crate::test_env::TestEnv::new();
    let tmp = tempfile::tempdir().unwrap();
    let (live, outcome) = decided_night(tmp.path());
    let finding = super::act_on(&live, &outcome, (1.0, "d"));
    assert!(finding.contains("json -> cand propose-only"), "{finding}");
    assert!(!tmp.path().join(".config/ztools/config.toml").exists());
    assert_eq!(live.staged().len(), 1, "a proposed candidate stays staged");
    drop(sandbox);
}

#[test]
fn with_auto_on_the_night_switches_and_unstages_the_candidate() {
    use crate::ztools::select::nightly::Night as _;
    let sandbox = crate::test_env::TestEnv::new();
    let tmp = tempfile::tempdir().unwrap();
    let (live, outcome) = decided_night(tmp.path());
    std::fs::create_dir_all(tmp.path().join(".config/ztools")).unwrap();
    std::fs::write(tmp.path().join(".config/ztools/models_auto"), "on").unwrap();
    let finding = super::act_on(&live, &outcome, (1.0, "d"));
    assert!(
        finding.contains("json old -> cand switched, canary 0/3"),
        "{finding}"
    );
    let sel = std::fs::read_to_string(tmp.path().join(".config/ztools/config.toml")).unwrap();
    assert!(sel.contains("json = \"cand\""), "{sel}");
    assert_eq!(
        live.staged(),
        Vec::<crate::ztools::select::nightly::Staged>::new(),
        "a switched candidate is no longer staged"
    );
    drop(sandbox);
}

#[test]
fn rollback_by_hand_needs_a_switch_and_then_restores_it() {
    let sandbox = crate::test_env::TestEnv::new();
    let home = sandbox.path("HOME");
    assert!(super::rollback().is_err(), "nothing to roll back");
    let (live, outcome) = decided_night(&home);
    std::fs::create_dir_all(home.join(".config/ztools")).unwrap();
    std::fs::write(home.join(".config/ztools/models_auto"), "on").unwrap();
    let _ = super::act_on(&live, &outcome, (1.0, "d"));
    super::rollback().unwrap();
    assert!(!home.join(".config/ztools/config.toml").exists());
    let findings =
        std::fs::read_to_string(home.join(".local/state/ztools/models/findings.md")).unwrap();
    assert!(
        findings.contains("json rolled back cand -> old (by hand)"),
        "{findings}"
    );
    drop(sandbox);
}

#[test]
fn the_night_deletes_a_switched_in_incumbent_14_days_after_its_successor() {
    use crate::ztools::autoswitch::{DAY, Ledger, RETIRE_DAYS, Retired};
    let sandbox = crate::test_env::TestEnv::new();
    let tmp = tempfile::tempdir().unwrap();
    let (live, _) = decided_night(tmp.path());
    std::fs::create_dir_all(live.model_dir("org/Cand")).unwrap();
    live.unstage("org/Cand").unwrap();
    assert_eq!(live.owned_dirs("cand").map(|d| d.len()), Some(2));
    assert_eq!(live.owned_dirs("old"), None, "the owner's own");
    let paths = live.switch_paths();
    let ledger = Ledger {
        retired: vec![
            Retired {
                model: "cand".into(),
                slot: "json".into(),
                since: 0.0,
            },
            Retired {
                model: "mine".into(),
                slot: "vlm".into(),
                since: 0.0,
            },
        ],
        ..Ledger::default()
    };
    ledger.save(&paths).unwrap();
    let mut finding = String::from("LM d: x");
    super::sweep_retired(&live, RETIRE_DAYS.mul_add(DAY, -1.0), &mut finding);
    assert_eq!(finding, "LM d: x", "not yet");
    super::sweep_retired(&live, RETIRE_DAYS * DAY, &mut finding);
    assert!(
        finding.contains("; cand replaced 14 days ago: weights deleted"),
        "{finding}"
    );
    assert!(
        finding.contains("; replaced mine kept (not staged by this job"),
        "{finding}"
    );
    assert!(!live.model_dir("org/Cand").exists());
    assert_eq!(live.owned_dirs("cand"), None);
    drop(sandbox);
}
