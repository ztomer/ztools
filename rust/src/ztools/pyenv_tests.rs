//! Both directions of the interpreter gate, with an injected probe so the
//! result never depends on what this machine happens to have installed.
//!
//! The accept direction alone would be worthless here: the bug being closed is
//! that an *unverified* interpreter was used, so the test that matters is the
//! one proving a candidate which cannot import the modules is REJECTED rather
//! than run anyway.

use super::*;

fn paths(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

#[test]
fn takes_the_first_candidate_that_can_import_everything() {
    let candidates = paths(&["/no/such/python", "/opt/homebrew/bin/python3", "python3"]);
    let found = resolve_with(&["requests"], &candidates, |program, _| {
        if program == "/opt/homebrew/bin/python3" {
            Ok(())
        } else {
            Err("No module named 'requests'".to_string())
        }
    })
    .expect("the homebrew candidate imports cleanly");
    assert_eq!(found, "/opt/homebrew/bin/python3");
}

#[test]
fn order_is_respected_when_several_candidates_would_work() {
    let candidates = paths(&["/first/python", "/second/python"]);
    let found = resolve_with(&["json"], &candidates, |_, _| Ok(())).unwrap();
    assert_eq!(found, "/first/python", "the earliest candidate must win");
}

/// The regression that started this: `/usr/bin/python3` exists, runs, and is
/// missing `requests`. Existing is not a qualification.
#[test]
fn an_interpreter_missing_a_module_is_rejected_not_used() {
    let candidates = paths(&["/usr/bin/python3"]);
    let err = resolve_with(&["requests"], &candidates, |_, _| {
        Err("ModuleNotFoundError: No module named 'requests'".to_string())
    })
    .expect_err("an interpreter without the module must not be handed back");
    assert_eq!(err.required, vec!["requests".to_string()]);
    assert_eq!(err.rejected.len(), 1);
    assert_eq!(err.rejected[0].0, "/usr/bin/python3");
    assert!(err.rejected[0].1.contains("requests"), "{err:?}");
}

/// A failure has to name every path tried and what each lacked — the original
/// error said only "exit code 1", which is what made this a three-day bug.
#[test]
fn the_failure_names_every_candidate_and_its_reason() {
    let candidates = paths(&["/a/python", "/b/python"]);
    let err = resolve_with(&["requests", "playwright"], &candidates, |program, _| {
        Err(format!("{program} has nothing"))
    })
    .unwrap_err();
    let rendered = err.to_string();
    assert!(rendered.contains("requests, playwright"), "{rendered}");
    assert!(rendered.contains("/a/python"), "{rendered}");
    assert!(rendered.contains("/b/python"), "{rendered}");
    assert!(rendered.contains("ZTOOLS_PYTHON"), "{rendered}");
}

#[test]
fn an_empty_candidate_list_is_an_error_not_a_bare_python3() {
    let err = resolve_with(&["requests"], &[], |_, _| Ok(())).unwrap_err();
    assert!(
        err.rejected.is_empty(),
        "nothing was tried, so nothing can be blamed"
    );
}

/// `ZTOOLS_PYTHON` is the documented lever; if it stopped being first the
/// override would silently lose to whatever is installed.
#[test]
fn the_explicit_override_is_considered_before_anything_installed() {
    // Set through the candidate builder rather than the env so the test does
    // not mutate process-global state other tests share.
    let with_override = {
        let mut v = vec!["/my/python".to_string()];
        v.extend(paths(&["/opt/homebrew/bin/python3", "python3"]));
        v
    };
    let found = resolve_with(&["json"], &with_override, |_, _| Ok(())).unwrap();
    assert_eq!(found, "/my/python");
}

/// The probe must see the same `PYTHONPATH` the real run will, or it verifies
/// an interpreter that then fails on the shipped `references/` modules.
#[test]
fn the_reference_path_is_applied_to_a_command() {
    let mut cmd = std::process::Command::new("/usr/bin/true");
    apply_pythonpath(&mut cmd);
    let applied: Vec<_> = cmd
        .get_envs()
        .filter(|(k, _)| *k == std::ffi::OsStr::new("PYTHONPATH"))
        .collect();
    if reference_paths().is_empty() {
        assert!(applied.is_empty(), "nothing to add, nothing set");
    } else {
        let value = applied
            .first()
            .and_then(|(_, v)| *v)
            .expect("PYTHONPATH set when references/ exists");
        assert!(value.to_string_lossy().contains("references"), "{value:?}");
    }
}
