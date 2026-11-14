//! LM.INCUMBENTFAIL: one night at a time, and an eval's failure carries its
//! reason.
//!
//! On 2026-11-13 and 2026-11-14 the routines runner started `models nightly`
//! again while the 02:30 night was still running (the task had no
//! `exclusive_resource`, and a task is stamped only when it finishes). The
//! second start's first eval, the incumbent's, waited 5 s for the GPU lock the
//! first night held and exited 1 having asked Osaurus nothing: three
//! `✗ eval raptor-v0.5-8b-a1b-jang_6m on json: model-eval exited exit status:
//! 1` lines, each naming no reason, while the night beside them measured the
//! incumbent complete.

use super::nightly::Night;
use super::nightly_tests::live;

#[test]
fn a_second_night_does_nothing_while_the_first_runs() {
    let _env = crate::test_env::TestEnv::new();
    let tmp = tempfile::tempdir().unwrap();
    let mut night = live(tmp.path());
    night.slots = std::iter::once(("think".to_string(), "inc".to_string())).collect();
    night
        .write_ledger(&super::nightly::Staged {
            repo: super::nightly_tests::CAND.into(),
            model: "cand-1b-4bit".into(),
            slots: vec!["think".into()],
            bytes: 1,
            date: "2026-11-14".into(),
            revision: None,
        })
        .unwrap();
    let state = night.state_dir.clone();
    let first = super::once::claim(&state).unwrap();
    assert!(matches!(first, super::once::Claim::Ours(_)));
    crate::cli_models_nightly::finish_night(&mut night, &state).unwrap();
    assert!(
        !state.join("findings.md").exists(),
        "the second start wrote a findings line"
    );
    drop(first);
    let err = crate::cli_models_nightly::finish_night(&mut night, &state).unwrap_err();
    assert!(err.to_string().contains("eval inc on think"), "{err:#}");
    assert!(
        state.join("findings.md").exists(),
        "the night ran once freed"
    );
}

#[test]
fn a_dead_nights_claim_is_taken_over() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("nightly.lock");
    std::fs::create_dir_all(&dir).unwrap();
    let racing = super::once::claim(tmp.path()).unwrap();
    assert!(
        matches!(&racing, super::once::Claim::Held(w) if w == "a night"),
        "a claim whose owner is not yet written is someone's: {racing:?}"
    );
    std::fs::write(
        dir.join("owner"),
        "999999\n\nztools models nightly (pid 999999)\n",
    )
    .unwrap();
    let claim = super::once::claim(tmp.path()).unwrap();
    assert!(matches!(claim, super::once::Claim::Ours(_)), "{claim:?}");
    drop(claim);
    assert!(!dir.exists(), "released on drop");
}

#[test]
fn a_failed_eval_says_why() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let exe = tmp.path().join("ztools");
    std::fs::write(
        &exe,
        "#!/bin/sh\necho 'Testing m'\necho 'early noise' >&2\necho 'Error: GPU lock unavailable: GPU still held by eval x' >&2\nexit 1\n",
    )
    .unwrap();
    std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut night = live(tmp.path());
    night.exe = exe;
    let err = night.eval("m", &["json".into()]).unwrap_err();
    assert!(
        err.ends_with("model-eval exited exit status: 1: Error: GPU lock unavailable: GPU still held by eval x"),
        "{err}"
    );
}
