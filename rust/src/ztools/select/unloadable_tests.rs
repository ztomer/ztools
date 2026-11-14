//! LM.RAPTOR07HANG: a candidate that fails to load is recorded, deleted and
//! never loaded again on the same Osaurus.

use super::live::{LiveNight, first_nominee};
use super::nightly::{Night, Outcome, Staged, findings_line, run};
use super::nightly_tests::{CAND, fake, live};
use super::rule::Rule;
use super::unloadable;

const RAPTOR: &str = "OsaurusAI/Raptor-0.7-4B-JANG_6M";

#[test]
fn a_candidate_whose_own_eval_fails_is_recorded_and_deleted() {
    let mut f = fake(82, 186);
    f.scores.remove("cand-1b-4bit");
    let out = run(&mut f, &Rule::DESIGN);
    assert!(
        matches!(&out, Outcome::Rejected { candidate, .. } if candidate == "cand-1b-4bit"),
        "{out:?}"
    );
    assert_eq!(f.rejected.len(), 1, "{:?}", f.rejected);
    assert!(f.rejected[0].starts_with(CAND), "{:?}", f.rejected);
    assert_eq!(f.deleted, vec![CAND.to_string()]);
    let line = findings_line("d", &out);
    assert!(
        line.starts_with("LM d: ✗ cand-1b-4bit failed its own eval (eval cand-1b-4bit on think"),
        "{line}"
    );
    assert!(
        line.ends_with("recorded unloadable on this Osaurus, deleted"),
        "{line}"
    );
}

#[test]
fn a_candidate_that_wedged_osaurus_fails_the_night_and_says_so() {
    let mut f = fake(82, 186);
    f.scores.remove("cand-1b-4bit");
    f.wedged = true;
    let out = run(&mut f, &Rule::DESIGN);
    assert!(
        matches!(&out, Outcome::Failed(w) if w.contains("wedged, restart it")),
        "{out:?}"
    );
    assert_eq!(f.deleted, vec![CAND.to_string()], "deleted all the same");
}

#[test]
fn an_incumbent_that_fails_never_blames_the_candidate() {
    let mut f = fake(82, 186);
    f.scores.remove("inc");
    let out = run(&mut f, &Rule::DESIGN);
    assert!(matches!(out, Outcome::Failed(_)), "{out:?}");
    assert_eq!((f.rejected.len(), f.deleted.len()), (0, 0));
}

#[test]
fn a_record_bars_its_osaurus_version_and_lifts_on_another() {
    let tmp = tempfile::tempdir().unwrap();
    let rec = unloadable::Unloadable {
        model: "raptor-0.7-4b-jang_6m".into(),
        osaurus: Some("1.0.3".into()),
        date: "2026-11-14".into(),
        why: "wedged".into(),
    };
    unloadable::record(tmp.path(), RAPTOR, rec).unwrap();
    let why = unloadable::why(tmp.path(), "raptor-0.7-4b-jang_6m", Some("1.0.3")).unwrap();
    assert!(why.contains("failed to load on Osaurus 1.0.3"), "{why}");
    assert!(unloadable::why(tmp.path(), &RAPTOR.to_lowercase(), None).is_some());
    assert_eq!(unloadable::why(tmp.path(), RAPTOR, Some("1.0.4")), None);
    assert_eq!(unloadable::why(tmp.path(), "other", Some("1.0.3")), None);
}

/// A fake Osaurus that accepts every connection and answers nothing, as the
/// wedged 1.0.3 did.
fn wedged_osaurus() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        for s in listener.incoming().take(4) {
            drop(s);
        }
    });
    url
}

/// A `ztools` that logs each eval and fails the candidate's.
fn eval_script(dir: &std::path::Path) -> std::path::PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let exe = dir.join("ztools");
    let log = dir.join("evals.log");
    std::fs::write(
        &exe,
        format!(
            "#!/bin/sh\necho \"$*\" >> '{}'\ncase \"$*\" in *raptor-0.7*) exit 1;; esac\nexit 0\n",
            log.display()
        ),
    )
    .unwrap();
    std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).unwrap();
    exe
}

fn staged_raptor(night: &LiveNight) -> Staged {
    let s = Staged {
        repo: RAPTOR.into(),
        model: "raptor-0.7-4b-jang_6m".into(),
        slots: vec!["json".into()],
        bytes: 1,
        date: "2026-11-14".into(),
    };
    night.write_ledger(&s).unwrap();
    std::fs::create_dir_all(night.model_dir(RAPTOR)).unwrap();
    std::fs::write(
        night.model_dir(RAPTOR).join("config.json"),
        r#"{"architectures": ["Spark2_5ForCausalLM"], "model_type": "spark2_5"}"#,
    )
    .unwrap();
    s
}

#[test]
fn the_live_night_unstages_a_wedging_candidate_and_never_loads_it_again() {
    let _env = crate::test_env::TestEnv::new();
    let tmp = tempfile::tempdir().unwrap();
    let mut night = live(tmp.path());
    night.osaurus_url = wedged_osaurus();
    night.osaurus_version = Some("1.0.3".into());
    night.slots = std::iter::once(("json".to_string(), "inc".to_string())).collect();
    night.exe = eval_script(tmp.path());
    staged_raptor(&night);
    let state = night.state_dir.clone();
    let err = crate::cli_models_nightly::finish_night(&mut night, &state).unwrap_err();
    assert!(err.to_string().contains("wedged, restart it"), "{err:#}");
    assert!(!night.model_dir(RAPTOR).exists(), "the weights are deleted");
    assert_eq!(night.staged().len(), 0, "no longer staged");
    let rec = &unloadable::load(&state)[RAPTOR];
    assert_eq!(rec.osaurus.as_deref(), Some("1.0.3"));
    assert!(rec.why.contains("model-eval exited"), "{rec:?}");
    let findings = std::fs::read_to_string(state.join("findings.md")).unwrap();
    assert!(
        findings.contains("Raptor-0.7-4B-JANG_6M failed its own eval"),
        "{findings}"
    );

    // Staged again (a crash, a hand copy): deleted unloaded, no eval runs.
    let log = tmp.path().join("evals.log");
    let before = std::fs::read_to_string(&log).unwrap().lines().count();
    staged_raptor(&night);
    let out = run(&mut night, &Rule::DESIGN);
    assert!(
        matches!(&out, Outcome::Nothing(w) if w.contains("failed to load on Osaurus 1.0.3") && w.ends_with("deleted")),
        "{out:?}"
    );
    assert_eq!(
        std::fs::read_to_string(&log).unwrap().lines().count(),
        before
    );
    assert!(!night.model_dir(RAPTOR).exists());

    // The scout's nominee skips it; a new Osaurus may try it again.
    assert!(night.excluded(RAPTOR).is_some());
    night.osaurus_version = Some("1.0.4".into());
    assert_eq!(night.excluded(RAPTOR), None);
}

#[test]
fn the_nominee_skips_an_excluded_repo() {
    use crate::ztools::scout::{Candidate, Check, Incumbent, Proposal};
    let cand = |repo: &str| Candidate {
        repo: repo.into(),
        nominated: "family".into(),
        provenance: Check::Pass(String::new()),
        format: Some(Check::Pass(String::new())),
        fit: Some(Check::Pass(String::new())),
        disk: None,
    };
    let p = Proposal {
        asked: 1,
        refused: vec![],
        incumbents: vec![Incumbent {
            slots: vec!["json".into()],
            name: "inc".into(),
            repo: None,
            family: Err(String::new()),
            retired: false,
            candidates: vec![cand(RAPTOR), cand("OsaurusAI/Next")],
        }],
        stage: None,
        notes: vec![],
    };
    assert_eq!(first_nominee(&p, &|_| false).unwrap().repo, RAPTOR);
    assert_eq!(
        first_nominee(&p, &|r| r == RAPTOR).unwrap().repo,
        "OsaurusAI/Next"
    );
}

#[test]
fn barring_by_hand_records_and_deletes_the_staged_copy_without_loading() {
    let tmp = tempfile::tempdir().unwrap();
    let mut night = live(tmp.path());
    night.osaurus_version = Some("1.0.3".into());
    night.exe = "/nonexistent".into();
    staged_raptor(&night);
    let line = crate::cli_models_nightly::bar_with(&mut night, RAPTOR, "wedges Osaurus").unwrap();
    assert!(
        line.ends_with("recorded unloadable on Osaurus 1.0.3; its staged copy deleted"),
        "{line}"
    );
    assert!(!night.model_dir(RAPTOR).exists());
    assert_eq!(night.staged().len(), 0);
    assert!(night.excluded("raptor-0.7-4b-jang_6m").is_some());
    let again = crate::cli_models_nightly::bar_with(&mut night, RAPTOR, "again").unwrap();
    assert!(!again.contains("staged copy"), "{again}");
    assert_eq!(unloadable::load(&night.state_dir).len(), 1);
}
