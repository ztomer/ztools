//! LM.AUTOSWITCH: every path under a temp HOME (`TestEnv`), never the real one.

use super::canary::{Run, Step, Verdict, observe, weekend_verdict};
use super::switch::{Promotion, promote, rollback};
use super::{Ledger, Mode, Paths, mode, mode_from, selection};
use crate::test_env::TestEnv;
use std::collections::BTreeMap;
use std::io::{Read, Write};

const DAY: f64 = 86_400.0;
const T0: f64 = 1_800_000_000.0;

fn base() -> BTreeMap<String, String> {
    [
        ("think", "inc-think"),
        ("json", "good-json"),
        ("summarize", "inc-sum"),
        ("filename", "inc-file"),
        ("vlm", "inc-vlm"),
    ]
    .into_iter()
    .map(|(a, b)| (a.to_string(), b.to_string()))
    .collect()
}

fn home(env: &TestEnv) -> Paths {
    Paths::under(&env.path("HOME"))
}

fn set_auto(paths: &Paths, text: &str) {
    std::fs::create_dir_all(paths.kill.parent().unwrap()).unwrap();
    std::fs::write(&paths.kill, text).unwrap();
}

/// A loopback chat server whose every completion is `content`, verbatim.
fn chat_stub(content: &'static str) -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        while let Ok((mut s, _)) = listener.accept() {
            let mut buf = Vec::new();
            let mut chunk = [0u8; 4096];
            loop {
                let n = s.read(&mut chunk).unwrap_or(0);
                buf.extend_from_slice(&chunk[..n]);
                let text = String::from_utf8_lossy(&buf).to_string();
                if let Some(end) = text.find("\r\n\r\n") {
                    let len = text
                        .lines()
                        .find_map(|l| {
                            l.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                        })
                        .unwrap_or(0);
                    if buf.len() >= end + 4 + len {
                        break;
                    }
                }
                if n == 0 {
                    break;
                }
            }
            let body =
                serde_json::json!({"choices": [{"message": {"content": content}}]}).to_string();
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            let _ = s.write_all(resp.as_bytes());
        }
    });
    format!("http://{addr}")
}

/// One json-consumer run: the config as the selection makes it, the weekend's
/// structure call against `url`, and the consumer's own gate on the result.
fn json_consumer_run(url: &str) -> Run {
    let mut cfg = crate::config::ZtoolsConfig::default().with_ztools_best_models();
    cfg.osaurus_url = url.to_string();
    cfg.llm_timeout_secs = 5;
    cfg.llm_extended_timeout_secs = 5;
    let resp = crate::ztools::weekend::phases::call_llm_json(Some("sys"), "rows", &cfg);
    let rows = resp
        .as_ref()
        .and_then(crate::ztools::weekend::parse_llm_events)
        .unwrap_or_default();
    let mut health = crate::ztools::weekend::PlanHealth::nominal();
    health.search.queries = 1;
    health.search.answered = 1;
    health.provenance.extracted = rows.len();
    Run {
        slot: "json".into(),
        model: cfg.weekend_model,
        verdict: weekend_verdict(&health),
        seconds: 1.0,
    }
}

fn selected_json() -> String {
    crate::config::ZtoolsConfig::default()
        .with_ztools_best_models()
        .weekend_model
}

/// THE DONE STATE: a seeded bad candidate (a server answering broken JSON) is
/// switched in, its first consumer run fails the canary, and that same run
/// restores the previous selection and bars it.
#[test]
fn a_bad_candidate_is_rolled_back_within_one_consumer_run() {
    let env = TestEnv::new();
    let paths = home(&env);
    set_auto(&paths, "on");
    let before = selected_json();
    let p = promote(
        &paths,
        &Mode::Auto,
        (T0, "2027-01-15"),
        &base(),
        "json",
        "bad-cand",
    )
    .unwrap();
    assert!(matches!(p, Promotion::Switched { .. }), "{p:?}");
    assert_eq!(
        selected_json(),
        "bad-cand",
        "the selection file outranks the shipped table"
    );
    assert_eq!(
        selection::slots(&std::fs::read_to_string(&paths.selection).unwrap()).unwrap()["think"],
        "inc-think",
        "a new selection file carries every slot"
    );

    let run = json_consumer_run(&chat_stub("{\"events\": [ {\"name\": \"x\", "));
    assert!(
        matches!(run.verdict, Verdict::Failed(_)),
        "{:?}",
        run.verdict
    );
    let step = observe(&paths, (T0 + 3600.0, "2027-01-15"), &run).unwrap();
    assert!(matches!(step, Step::RolledBack { .. }), "{step:?}");

    assert!(
        !paths.selection.exists(),
        "there was no selection file before"
    );
    assert_eq!(selected_json(), before);
    let ledger = Ledger::load(&paths).unwrap();
    assert!(ledger.canary.is_none());
    assert!(ledger.bar("bad-cand", DAY.mul_add(29.0, T0)).is_some());
    assert!(ledger.bar("bad-cand", DAY.mul_add(31.0, T0)).is_none());
    let findings = std::fs::read_to_string(paths.findings()).unwrap();
    assert!(
        findings.contains("json canary failed on bad-cand"),
        "{findings}"
    );
    let again = promote(
        &paths,
        &Mode::Auto,
        (T0 + DAY, "d"),
        &base(),
        "json",
        "bad-cand",
    )
    .unwrap();
    assert!(
        matches!(again, Promotion::Refused { ref why, .. } if why.contains("barred")),
        "{again:?}"
    );
    drop(env);
}

/// The kill switch makes the same run propose-only: nothing is written, and
/// the consumer run is not watched.
#[test]
fn the_kill_switch_makes_the_same_run_propose_only() {
    for (file, env_var) in [(Some("off"), None), (Some("on"), Some("0")), (None, None)] {
        let env = TestEnv::new();
        let paths = home(&env);
        if let Some(f) = file {
            set_auto(&paths, f);
        }
        if let Some(v) = env_var {
            env.set("ZTOOLS_MODEL_AUTO", v);
        }
        let m = mode(&paths);
        assert!(matches!(m, Mode::Propose(_)), "{file:?} {env_var:?}: {m:?}");
        let p = promote(&paths, &m, (T0, "d"), &base(), "json", "bad-cand").unwrap();
        assert!(matches!(p, Promotion::Proposed { .. }), "{p:?}");
        assert!(p.line().contains("propose-only"), "{}", p.line());
        assert!(!paths.selection.exists());
        assert!(!paths.ledger().exists());
        let run = json_consumer_run(&chat_stub("not json"));
        assert_ne!(
            run.model, "bad-cand",
            "the consumer still runs the incumbent"
        );
        assert_eq!(observe(&paths, (T0, "d"), &run).unwrap(), Step::NotWatched);
        drop(env);
    }
}

#[test]
fn only_on_in_the_file_turns_auto_on_and_either_off_wins() {
    let tmp = tempfile::tempdir().unwrap();
    let paths = Paths::under(tmp.path());
    assert!(
        matches!(mode_from(&paths, None), Mode::Propose(_)),
        "ships propose-only"
    );
    set_auto(&paths, "on\n");
    assert_eq!(mode_from(&paths, None), Mode::Auto);
    assert_eq!(mode_from(&paths, Some("1")), Mode::Auto);
    assert!(matches!(mode_from(&paths, Some("0")), Mode::Propose(_)));
    set_auto(&paths, "off");
    assert!(matches!(mode_from(&paths, None), Mode::Propose(_)));
    set_auto(&paths, "yes");
    assert!(matches!(mode_from(&paths, None), Mode::Propose(_)));
}

#[test]
fn embedding_models_are_never_switched() {
    let tmp = tempfile::tempdir().unwrap();
    let paths = Paths::under(tmp.path());
    for slot in ["embedding", "embed", "unknown"] {
        let p = promote(&paths, &Mode::Auto, (T0, "d"), &base(), slot, "potion-x").unwrap();
        assert!(matches!(p, Promotion::Refused { .. }), "{slot}: {p:?}");
    }
    let p = promote(&paths, &Mode::Auto, (T0, "d"), &base(), "embedding", "x").unwrap();
    assert!(p.line().contains("never switched"), "{}", p.line());
    assert!(!paths.selection.exists());
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
fn three_passes_confirm_then_the_rate_limits_hold() {
    let tmp = tempfile::tempdir().unwrap();
    let paths = Paths::under(tmp.path());
    promote(
        &paths,
        &Mode::Auto,
        (T0, "d0"),
        &base(),
        "summarize",
        "new-sum",
    )
    .unwrap();
    let other = promote(&paths, &Mode::Auto, (T0, "d0"), &base(), "think", "t2").unwrap();
    assert!(matches!(other, Promotion::Refused { ref why, .. } if why.contains("canary is open")));
    assert_eq!(
        observe(&paths, (T0, "d"), &pass("summarize", "inc-sum", 1.0)).unwrap(),
        Step::NotWatched
    );
    let skip = Run {
        verdict: Verdict::Inconclusive("no tweets".into()),
        ..pass("summarize", "new-sum", 1.0)
    };
    assert!(matches!(
        observe(&paths, (T0, "d"), &skip).unwrap(),
        Step::Inconclusive(_)
    ));
    assert_eq!(
        observe(&paths, (T0, "d"), &pass("summarize", "new-sum", 1.0)).unwrap(),
        Step::Passing(1)
    );
    assert_eq!(
        observe(&paths, (T0, "d"), &pass("summarize", "new-sum", 1.0)).unwrap(),
        Step::Passing(2)
    );
    assert_eq!(
        observe(&paths, (T0, "d"), &pass("summarize", "new-sum", 1.0)).unwrap(),
        Step::Confirmed("new-sum".into())
    );
    let mut b = base();
    b.insert("summarize".into(), "new-sum".into());
    let soon = promote(
        &paths,
        &Mode::Auto,
        (DAY.mul_add(6.0, T0), "d"),
        &b,
        "summarize",
        "s3",
    )
    .unwrap();
    assert!(
        matches!(soon, Promotion::Refused { ref why, .. } if why.contains("7 days")),
        "{soon:?}"
    );
    let later = promote(
        &paths,
        &Mode::Auto,
        (DAY.mul_add(8.0, T0), "d"),
        &b,
        "summarize",
        "s3",
    )
    .unwrap();
    assert!(matches!(later, Promotion::Switched { .. }), "{later:?}");
}

#[test]
fn a_run_slower_than_the_speed_floor_fails_the_canary() {
    let tmp = tempfile::tempdir().unwrap();
    let paths = Paths::under(tmp.path());
    promote(&paths, &Mode::Auto, (T0, "d"), &base(), "summarize", "slow").unwrap();
    let step = observe(&paths, (T0, "d"), &pass("summarize", "slow", 601.0)).unwrap();
    assert!(
        matches!(step, Step::RolledBack { ref why, .. } if why.contains("floor")),
        "{step:?}"
    );
}

#[test]
fn a_manual_rollback_keeps_other_keys_and_an_edit_made_since() {
    let tmp = tempfile::tempdir().unwrap();
    let paths = Paths::under(tmp.path());
    let original =
        "# mine\n[other]\nx = 1\n\n[best_models]\njson = \"good-json\"\nthink = \"inc-think\"\n";
    selection::write(&paths.selection, Some(original)).unwrap();
    promote(&paths, &Mode::Auto, (T0, "d"), &base(), "json", "j2").unwrap();
    let written = std::fs::read_to_string(&paths.selection).unwrap();
    assert!(
        written.contains("# mine") && written.contains("json = \"j2\""),
        "{written}"
    );
    rollback(&paths, (T0, "d"), "by hand").unwrap();
    assert_eq!(std::fs::read_to_string(&paths.selection).unwrap(), original);
    assert!(
        rollback(&paths, (T0, "d"), "again").is_err(),
        "nothing left to roll back"
    );

    promote(
        &paths,
        &Mode::Auto,
        (DAY.mul_add(8.0, T0), "d"),
        &base(),
        "json",
        "j3",
    )
    .unwrap();
    let edited = std::fs::read_to_string(&paths.selection)
        .unwrap()
        .replace("inc-think", "owner-think");
    std::fs::write(&paths.selection, &edited).unwrap();
    rollback(&paths, (DAY.mul_add(8.0, T0), "d"), "by hand").unwrap();
    let slots = selection::slots(&std::fs::read_to_string(&paths.selection).unwrap()).unwrap();
    assert_eq!(slots["json"], "good-json");
    assert_eq!(
        slots["think"], "owner-think",
        "the owner's edit since is kept"
    );
}

#[test]
fn a_corrupt_ledger_refuses_rather_than_reading_as_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let paths = Paths::under(tmp.path());
    std::fs::create_dir_all(&paths.state).unwrap();
    std::fs::write(paths.ledger(), "{").unwrap();
    assert!(promote(&paths, &Mode::Auto, (T0, "d"), &base(), "json", "x").is_err());
    assert!(observe(&paths, (T0, "d"), &pass("json", "x", 1.0)).is_err());
}

#[test]
fn the_nightly_switches_by_the_mode() {
    use crate::ztools::select::Decision;
    let tmp = tempfile::tempdir().unwrap();
    let paths = Paths::under(tmp.path());
    let d = |slot: &str, chosen: &str| Decision {
        slot: slot.into(),
        incumbent: base()[slot].clone(),
        chosen: chosen.into(),
        judged: vec![],
    };
    let decisions = [d("think", "inc-think"), d("json", "cand")];
    let off = Mode::Propose("ZTOOLS_MODEL_AUTO=0".into());
    let acts = super::night::act(&paths, &off, (T0, "d"), &base(), &decisions);
    assert_eq!(acts.len(), 1, "a kept slot is not acted on");
    assert!(acts[0].1.contains("propose-only"), "{}", acts[0].1);
    assert!(!super::night::switched_to(&acts, "cand"));
    assert!(!paths.selection.exists());
    let acts = super::night::act(&paths, &Mode::Auto, (T0, "d"), &base(), &decisions);
    assert!(super::night::switched_to(&acts, "cand"), "{acts:?}");
    assert!(acts[0].1.contains("canary 0/3"), "{}", acts[0].1);
}
