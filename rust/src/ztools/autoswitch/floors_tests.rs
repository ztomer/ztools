//! LM.CANARYFLOORS: every slot that may switch has a canary floor, a slot
//! without one refuses, and a replaced incumbent's weights go 14 days on.

use super::canary::{Run, Verdict, floor_seconds, observe};
use super::retire::{Swept, sweep};
use super::switch::{Promotion, promote};
use super::{DAY, Ledger, Mode, Paths, RETIRE_DAYS, SWITCHABLE};
use std::collections::{BTreeMap, BTreeSet};

const T0: f64 = 1_800_000_000.0;

fn base() -> BTreeMap<String, String> {
    [
        ("think", "inc-think"),
        ("json", "inc-json"),
        ("summarize", "inc-sum"),
        ("filename", "inc-name"),
        ("vlm", "inc-sum"),
    ]
    .iter()
    .map(|(a, b)| ((*a).to_string(), (*b).to_string()))
    .collect()
}

fn pass(slot: &str, model: &str, seconds: f64) -> Run {
    Run {
        slot: slot.into(),
        model: model.into(),
        verdict: Verdict::Passed,
        seconds,
    }
}

#[test]
fn only_the_slots_whose_consumer_runs_were_measured_have_floors() {
    for slot in ["json", "summarize"] {
        assert!(floor_seconds(slot).is_some_and(|f| f > 0.0), "{slot}");
    }
    for slot in ["think", "filename", "vlm"] {
        assert_eq!(
            floor_seconds(slot),
            None,
            "{slot}: no measured consumer run yet"
        );
    }
}

#[test]
fn a_slot_without_a_floor_refuses_to_switch() {
    let tmp = tempfile::tempdir().unwrap();
    let paths = Paths::under(tmp.path());
    let p = promote(&paths, &Mode::Auto, (T0, "d"), &base(), "think", "cand").unwrap();
    assert!(
        matches!(&p, Promotion::Refused { why, .. } if why.contains("no canary speed floor")),
        "{p:?}"
    );
    assert!(!paths.selection.exists(), "nothing written");
    for slot in SWITCHABLE {
        let refused = matches!(
            promote(
                &paths,
                &Mode::Propose("p".into()),
                (T0, "d"),
                &base(),
                slot,
                "c"
            )
            .unwrap(),
            Promotion::Refused { .. }
        );
        assert_eq!(refused, floor_seconds(slot).is_none(), "{slot}");
    }
}

/// Switch `summarize` to `cand` and confirm it at `T0`.
fn confirmed(paths: &Paths) {
    let p = promote(paths, &Mode::Auto, (T0, "d"), &base(), "summarize", "cand").unwrap();
    assert!(matches!(p, Promotion::Switched { .. }), "{p:?}");
    for _ in 0..3 {
        observe(paths, (T0, "d"), &pass("summarize", "cand", 1.0)).unwrap();
    }
    let ledger = Ledger::load(paths).unwrap();
    assert_eq!(ledger.retired.len(), 1);
    assert_eq!(ledger.retired[0].model, "inc-sum");
}

#[test]
fn a_replaced_incumbent_this_job_staged_is_deleted_on_day_14() {
    let tmp = tempfile::tempdir().unwrap();
    let paths = Paths::under(tmp.path());
    confirmed(&paths);
    let weights = tmp.path().join("MLXModels/org/Inc-Sum");
    std::fs::create_dir_all(&weights).unwrap();
    let owned = |m: &str| (m == "inc-sum").then(|| vec![weights.clone()]);
    let none = BTreeSet::new();
    let early = sweep(&paths, RETIRE_DAYS.mul_add(DAY, T0) - 1.0, &none, &owned);
    assert_eq!(early.unwrap(), vec![], "kept until day 14");
    assert!(weights.exists());
    let due = sweep(&paths, RETIRE_DAYS.mul_add(DAY, T0), &none, &owned).unwrap();
    assert_eq!(due, vec![Swept::Deleted("inc-sum".into())]);
    assert!(!weights.exists(), "rm -rf, never Trash");
    assert_eq!(Ledger::load(&paths).unwrap().retired.len(), 0);
    assert!(
        due[0].line().contains("weights deleted"),
        "{}",
        due[0].line()
    );
}

#[test]
fn an_owner_installed_or_reselected_incumbent_is_kept() {
    let tmp = tempfile::tempdir().unwrap();
    let paths = Paths::under(tmp.path());
    confirmed(&paths);
    let later = RETIRE_DAYS.mul_add(DAY, T0);
    let in_use: BTreeSet<String> = std::iter::once("inc-sum".to_string()).collect();
    let back = sweep(&paths, T0 + DAY, &in_use, &|_| panic!("never asked")).unwrap();
    assert!(
        matches!(&back[..], [Swept::Kept(m, why)] if m == "inc-sum" && why.contains("selects it again"))
    );
    let tmp = tempfile::tempdir().unwrap();
    let paths = Paths::under(tmp.path());
    confirmed(&paths);
    let mine = sweep(&paths, later, &BTreeSet::new(), &|_| None).unwrap();
    assert!(
        matches!(&mine[..], [Swept::Kept(_, why)] if why.contains("yours to delete")),
        "{mine:?}"
    );
    assert_eq!(Ledger::load(&paths).unwrap().retired.len(), 0, "said once");
}

#[test]
fn a_removal_that_fails_is_said_and_tried_again() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let paths = Paths::under(tmp.path());
    confirmed(&paths);
    let parent = tmp.path().join("ro");
    let weights = parent.join("Inc-Vlm");
    std::fs::create_dir_all(weights.join("sub")).unwrap();
    std::fs::set_permissions(&parent, std::fs::Permissions::from_mode(0o555)).unwrap();
    let owned = |_: &str| Some(vec![weights.clone()]);
    let out = sweep(
        &paths,
        RETIRE_DAYS.mul_add(DAY, T0),
        &BTreeSet::new(),
        &owned,
    )
    .unwrap();
    std::fs::set_permissions(&parent, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(
        matches!(&out[..], [Swept::Kept(_, why)] if why.contains("tried again")),
        "{out:?}"
    );
    assert_eq!(Ledger::load(&paths).unwrap().retired.len(), 1);
}

#[test]
fn no_ledger_sweeps_nothing() {
    let tmp = tempfile::tempdir().unwrap();
    let out = sweep(&Paths::under(tmp.path()), T0, &BTreeSet::new(), &|_| None).unwrap();
    assert_eq!(out, vec![]);
}
