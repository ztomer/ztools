//! The replay over today's recorded `eval_history.json`
//! (`ztools models nightly --replay --record tests/fixtures/select/replay.json`).

use super::replay::{ReplayInput, replay};
use super::rule::Rule;
use std::collections::BTreeMap;

fn input() -> ReplayInput {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tests/fixtures/select/replay.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// `[best_models]` as the shipped `conf/config.toml` states it.
fn shipped_best_models() -> BTreeMap<String, String> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../conf/config.toml");
    let conf: toml::Value = toml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    conf["best_models"]
        .as_table()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
        .collect()
}

#[test]
fn replaying_history_reproduces_todays_best_models_exactly() {
    let input = input();
    assert_eq!(
        input.best_models,
        shipped_best_models(),
        "the fixture is today's table"
    );
    let r = replay(&input, &Rule::DESIGN);
    assert_eq!(
        r.derived(),
        shipped_best_models(),
        "{}",
        r.lines().join("\n")
    );
    assert!(r.reproduces(&input.best_models));
    assert_eq!(r.decisions.len(), 5, "every slot was measured");
}

#[test]
fn a_zero_margin_switches_think_so_the_replay_can_go_red() {
    let rule = Rule {
        margin_floor: 0.0,
        ..Rule::DESIGN
    };
    let r = replay(&input(), &rule);
    assert_eq!(r.derived()["think"], "gemma-4-e4b-it-8bit");
    assert!(!r.reproduces(&shipped_best_models()));
}

#[test]
fn the_json_keep_rests_on_the_second_measure() {
    let r = replay(&input(), &Rule::DESIGN);
    let json = r.decisions.iter().find(|d| d.slot == "json").unwrap();
    let crack = json
        .judged
        .iter()
        .find(|j| j.model == "qwen3.8-27b-jang_6d-crack")
        .unwrap();
    assert!(
        crack.band && crack.gates.is_empty() && crack.second.is_none(),
        "{crack:?}"
    );
    assert!(!json.switched());
}

#[test]
fn an_incumbent_with_no_rows_never_reproduces() {
    let mut input = input();
    input
        .best_models
        .insert("think".into(), "not-a-model".into());
    let r = replay(&input, &Rule::DESIGN);
    assert_eq!(r.unmeasured, vec!["think".to_string()]);
    assert!(!r.reproduces(&input.best_models));
    assert!(r.lines().iter().any(|l| l.contains("no rows to replay")));
}
