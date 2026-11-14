//! The autoswitch seen from the nightly and the consumers' gates.

use super::canary::{Verdict, twitter_verdict, weekend_verdict};
use super::{Ledger, Paths};
use crate::ztools::select::Rule;
use crate::ztools::select::live::LiveNight;
use crate::ztools::select::nightly::{self, Night, Outcome, Staged};
use std::collections::BTreeMap;

fn live(dir: &std::path::Path) -> LiveNight {
    LiveNight {
        hf: "http://127.0.0.1:9".into(),
        osaurus_url: "http://127.0.0.1:9".into(),
        slots: BTreeMap::new(),
        current: BTreeMap::new(),
        models_root: dir.join("MLXModels"),
        state_dir: dir.join("state"),
        exe: "/nonexistent".into(),
        osaurus_version: None,
    }
}

fn staged() -> Staged {
    Staged {
        repo: "org/Bad-4bit".into(),
        model: "bad-4bit".into(),
        slots: vec!["json".into()],
        bytes: 1,
        date: "d".into(),
    }
}

#[test]
fn a_barred_staged_candidate_is_deleted_never_evaluated() {
    let tmp = tempfile::tempdir().unwrap();
    let mut night = live(tmp.path());
    let s = staged();
    night.write_ledger(&s).unwrap();
    let weights = night.model_dir(&s.repo);
    std::fs::create_dir_all(&weights).unwrap();
    assert_eq!(night.switch_paths().state, tmp.path().join("state"));
    assert_eq!(
        night.switch_paths(),
        Paths {
            state: tmp.path().join("state"),
            ..Paths::under(tmp.path())
        }
    );
    let (now, _) = super::clock();
    let mut ledger = Ledger::default();
    ledger.barred.insert(s.model, now + 86_400.0);
    ledger.save(&night.switch_paths()).unwrap();
    let out = nightly::run(&mut night, &Rule::DESIGN);
    assert!(
        matches!(out, Outcome::Nothing(ref w) if w.contains("barred") && w.contains("deleted")),
        "{out:?}"
    );
    assert!(!weights.exists());
    assert_eq!(night.staged(), Vec::<Staged>::new());
}

#[test]
fn a_switched_candidate_leaves_the_staged_set_but_stays_ledgered() {
    let tmp = tempfile::tempdir().unwrap();
    let night = live(tmp.path());
    let s = staged();
    night.write_ledger(&s).unwrap();
    night.unstage(&s.repo).unwrap();
    assert_eq!(night.staged(), Vec::<Staged>::new());
    assert!(
        tmp.path()
            .join("state/switched/org--Bad-4bit/ledger.json")
            .is_file()
    );
    assert!(night.unstage(&s.repo).is_err(), "nothing left to move");
}

#[test]
fn the_summarize_gate_fails_a_degraded_or_failed_run() {
    let tmp = tempfile::tempdir().unwrap();
    let ok = tmp.path().join("ok.md");
    std::fs::write(&ok, "# Twitter Timeline Summary\n\n**Model:** m\n").unwrap();
    let bad = tmp.path().join("bad.md");
    std::fs::write(&bad, "> ⚠ **DEGRADED** — fallback answered\n").unwrap();
    assert_eq!(twitter_verdict(&Ok(ok)), Verdict::Passed);
    assert!(matches!(twitter_verdict(&Ok(bad)), Verdict::Failed(ref w) if w.contains("degraded")));
    assert!(matches!(
        twitter_verdict(&Ok(tmp.path().join("gone.md"))),
        Verdict::Failed(_)
    ));
    assert!(
        matches!(twitter_verdict(&Err(anyhow::anyhow!("boom"))), Verdict::Failed(ref w) if w.contains("boom"))
    );
}

#[test]
fn the_json_gate_reads_the_plans_health() {
    use crate::ztools::weekend::{ModelHealth, PlanHealth};
    let mut h = PlanHealth::nominal();
    assert!(
        matches!(weekend_verdict(&h), Verdict::Inconclusive(_)),
        "no search answered"
    );
    h.search.answered = 2;
    assert!(matches!(weekend_verdict(&h), Verdict::Failed(ref w) if w.contains("JSON")));
    h.provenance.extracted = 3;
    assert_eq!(weekend_verdict(&h), Verdict::Passed);
    h.model = ModelHealth::Unavailable {
        model: "m".into(),
        reason: "cold".into(),
    };
    assert!(matches!(weekend_verdict(&h), Verdict::Failed(ref w) if w.contains("cold")));
    h.model = ModelHealth::NotInstalled {
        model: "m".into(),
        installed: vec![],
    };
    assert!(matches!(weekend_verdict(&h), Verdict::Failed(ref w) if w.contains("not installed")));
}
