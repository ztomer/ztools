//! LM.RENOMINATE: a candidate the rule rejected stays rejected until something
//! the rejection rested on changes.

use super::live::LiveNight;
use super::nightly::{Night, Outcome, Staged, run};
use super::nightly_tests::{CAND, fake, live, serve};
use super::rejected;
use super::rule::Rule;
use crate::ztools::eval::report::HistoryEntry;
use crate::ztools::eval::task_fingerprint::TaskIdentities;
use std::collections::BTreeMap;

fn current() -> TaskIdentities {
    ["file_summary", "taxes_qa", "refusal_war_en", "summarize"]
        .iter()
        .map(|t| ((*t).to_string(), format!("fp-{t}")))
        .collect()
}

/// Three runs each of `inc` (80) and `cand-1b-4bit` (82), stamped far ahead
/// of any night's start: a +2.0 the rule rejects.
fn seed_history(home: &std::path::Path) {
    let mut all: BTreeMap<String, Vec<HistoryEntry>> = BTreeMap::new();
    for (model, score, offset) in [("inc", 80, 0.0), ("cand-1b-4bit", 82, 100.0)] {
        for run in 0..3 {
            for t in current().keys() {
                all.entry(model.into()).or_default().push(HistoryEntry {
                    date: "2096-01-01".into(),
                    timestamp: f64::from(run).mul_add(10.0, 4.0e9 + offset),
                    task: t.clone(),
                    score: if t.starts_with("refusal") { 100 } else { score },
                    time: Some(5.0),
                    complete: true,
                    fingerprint: Some(format!("fp-{t}")),
                });
            }
        }
    }
    let dir = home.join(".config/ztools");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("eval_history.json"),
        serde_json::to_string(&all).unwrap(),
    )
    .unwrap();
}

fn staged_cand(night: &LiveNight) -> Staged {
    let s = Staged {
        repo: CAND.into(),
        model: "cand-1b-4bit".into(),
        slots: vec!["think".into()],
        bytes: 1,
        date: "2026-11-14".into(),
        revision: None,
    };
    night.write_ledger(&s).unwrap();
    std::fs::create_dir_all(night.model_dir(CAND)).unwrap();
    s
}

/// A live night under the sandboxed `$HOME`, whose history is seeded.
fn night_under(dir: &std::path::Path) -> LiveNight {
    seed_history(&dirs::home_dir().expect("the sandbox sets HOME"));
    let mut night = live(dir);
    night.exe = "/usr/bin/true".into();
    night.osaurus_version = Some("1.0.4".into());
    night.slots = std::iter::once(("think".to_string(), "inc".to_string())).collect();
    night.current = current();
    night
}

#[test]
fn a_rule_rejected_candidate_is_not_nominated_again() {
    let _env = crate::test_env::TestEnv::new();
    let tmp = tempfile::tempdir().unwrap();
    let mut night = night_under(tmp.path());
    staged_cand(&night);
    let out = run(&mut night, &Rule::DESIGN);
    assert!(
        matches!(&out, Outcome::Decided { deleted: true, .. }),
        "{out:?}"
    );
    assert!(!night.model_dir(CAND).exists(), "deleted that night");
    let why = night.excluded(CAND).expect("the rejection must stand");
    assert!(why.contains("rejected by the rule"), "{why}");
    assert!(
        night.excluded("cand-1b-4bit").is_some(),
        "by short name too"
    );
}

fn record(dir: &std::path::Path, revision: Option<&str>) {
    let slots = std::iter::once(("think".to_string(), "inc".to_string())).collect();
    rejected::record(
        dir,
        CAND,
        rejected::Rejected {
            model: "cand-1b-4bit".into(),
            revision: revision.map(str::to_string),
            osaurus: Some("1.0.4".into()),
            slots: rejected::basis(&current(), &slots, &["think".into()]),
            date: "2026-11-14".into(),
            why: "think: +2.0, gated".into(),
        },
    )
    .unwrap();
}

fn ask(
    dir: &std::path::Path,
    revision: &dyn Fn(&str) -> Option<String>,
    osaurus: Option<&str>,
    slots: &BTreeMap<String, String>,
    current: &TaskIdentities,
) -> Option<String> {
    let n = rejected::Now {
        revision,
        osaurus,
        slots,
        current,
    };
    rejected::why(dir, CAND, &n)
}

#[test]
fn a_rejection_lifts_only_when_something_it_rested_on_changes() {
    let tmp = tempfile::tempdir().unwrap();
    record(tmp.path(), Some("abc"));
    let slots: BTreeMap<String, String> =
        std::iter::once(("think".to_string(), "inc".to_string())).collect();
    let cur = current();
    let same = |_: &str| Some("abc".to_string());
    let now =
        |rev: &dyn Fn(&str) -> Option<String>, os, slots, cur| ask(tmp.path(), rev, os, slots, cur);
    assert!(
        now(&same, Some("1.0.4"), &slots, &cur).is_some(),
        "nothing changed"
    );
    assert!(
        now(&|_| None, None, &slots, &cur).is_some(),
        "unknowns do not lift it"
    );
    assert_eq!(
        now(&|_| Some("def".into()), Some("1.0.4"), &slots, &cur),
        None
    );
    assert_eq!(now(&same, Some("1.0.5"), &slots, &cur), None);
    let other: BTreeMap<String, String> =
        std::iter::once(("think".to_string(), "inc2".to_string())).collect();
    assert_eq!(
        now(&same, Some("1.0.4"), &other, &cur),
        None,
        "new incumbent"
    );
    let mut moved = cur.clone();
    moved.insert("taxes_qa".into(), "fp-new".into());
    assert_eq!(
        now(&same, Some("1.0.4"), &slots, &moved),
        None,
        "task replaced"
    );
    assert_eq!(
        now(&same, Some("1.0.4"), &BTreeMap::new(), &cur),
        None,
        "slot gone"
    );
    let n = rejected::Now {
        revision: &same,
        osaurus: Some("1.0.4"),
        slots: &slots,
        current: &cur,
    };
    assert_eq!(rejected::why(tmp.path(), "org/Other", &n), None);
}

#[test]
fn a_record_with_no_revision_is_never_lifted_by_one() {
    let tmp = tempfile::tempdir().unwrap();
    record(tmp.path(), None);
    let slots = std::iter::once(("think".to_string(), "inc".to_string())).collect();
    let n = rejected::Now {
        revision: &|_| Some("def".into()),
        osaurus: Some("1.0.4"),
        slots: &slots,
        current: &current(),
    };
    assert!(rejected::why(tmp.path(), CAND, &n).is_some());
}

#[test]
fn the_night_records_why_the_rule_rejected_and_a_failed_record_fails_it() {
    let mut f = fake(82, 186);
    let out = run(&mut f, &Rule::DESIGN);
    assert!(
        matches!(out, Outcome::Decided { deleted: true, .. }),
        "{out:?}"
    );
    assert_eq!(f.ruled.len(), 1, "{:?}", f.ruled);
    assert!(
        f.ruled[0].starts_with(&format!("{CAND}: think: +2.0, ")),
        "{:?}",
        f.ruled
    );

    let mut f = fake(84, 186);
    let _ = run(&mut f, &Rule::DESIGN);
    assert_eq!(f.ruled.len(), 0, "a winner is not recorded as rejected");

    let mut f = fake(82, 186);
    f.unrecordable = true;
    let out = run(&mut f, &Rule::DESIGN);
    assert!(
        matches!(&out, Outcome::Failed(w) if w.contains("rejected, but not recorded")),
        "{out:?}"
    );
    assert_eq!(f.deleted, vec![CAND.to_string()], "deleted all the same");
}

#[test]
fn the_live_night_records_the_hub_revision_it_downloaded() {
    let _env = crate::test_env::TestEnv::new();
    let tmp = tempfile::tempdir().unwrap();
    let mut night = night_under(tmp.path());
    night.hf = serve(1, r#"{"id": "org/Cand-1B-4bit", "sha": "abc123"}"#);
    assert_eq!(night.revision(CAND).as_deref(), Some("abc123"));
    night.hf = "http://127.0.0.1:9".into();
    assert_eq!(
        night.revision(CAND),
        None,
        "an unreachable Hub says nothing"
    );
    let mut s = staged_cand(&night);
    s.revision = Some("abc123".into());
    night.reject_by_rule(&s, "think: +2.0").unwrap();
    let rec = &rejected::load(&night.state_dir)[CAND];
    assert_eq!(
        (rec.revision.as_deref(), rec.osaurus.as_deref()),
        (Some("abc123"), Some("1.0.4"))
    );
    assert_eq!(rec.slots["think"].incumbent, "inc");
    // The Hub cannot be read: the record stands.
    assert!(night.excluded(CAND).is_some());
}
