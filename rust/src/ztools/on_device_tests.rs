use std::os::unix::fs::PermissionsExt;

use super::*;

/// A stand-in for `fm`: a shell script with the given body, made executable.
fn stub(dir: &Path, body: &str) -> std::path::PathBuf {
    let path = dir.join("fm");
    std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

/// The prompt goes in on stdin and the answer comes back from stdout; the
/// arguments ask for one non-streamed answer.
#[test]
fn the_prompt_goes_in_on_stdin_and_stdout_is_the_answer() {
    let dir = tempfile::tempdir().unwrap();
    let fm = stub(
        dir.path(),
        r#"[ "$1 $2" = "respond --no-stream" ] || exit 3; tr a-z A-Z"#,
    );
    assert_eq!(respond(&fm, "two words\n", 10).unwrap(), "TWO WORDS");
}

/// Each way `fm` can fail is said by name, never as an empty answer.
#[test]
fn every_failure_is_named() {
    let dir = tempfile::tempdir().unwrap();
    let why = |body: &str| {
        let fm = stub(dir.path(), body);
        respond(&fm, "p", 2).unwrap_err().to_string()
    };
    assert!(why("cat >/dev/null; exit 69").contains("sudo"), "license");
    assert!(
        why("cat >/dev/null; echo boom >&2; exit 1").contains("boom"),
        "exit"
    );
    assert!(why("cat >/dev/null").contains("answered nothing"), "empty");
    assert!(why("sleep 30").contains("ran past 2s"), "timeout");
    let missing = respond(&dir.path().join("absent"), "p", 2).unwrap_err();
    assert!(format!("{missing:#}").contains("cannot run"), "{missing:#}");
}

/// A timeout stops what the child started too: a grandchild left running
/// outlives the call (the gate's orphan canary caught one).
#[test]
fn a_timeout_leaves_nothing_running() {
    let dir = tempfile::tempdir().unwrap();
    let pidfile = dir.path().join("grandchild");
    let fm = stub(
        dir.path(),
        &format!("sleep 30 & echo $! > {}; wait", pidfile.display()),
    );
    assert!(respond(&fm, "p", 1).is_err());
    let pid = std::fs::read_to_string(&pidfile).unwrap();
    std::thread::sleep(Duration::from_millis(200));
    let alive = Command::new("/bin/kill")
        .args(["-0", pid.trim()])
        .stderr(Stdio::null())
        .status()
        .unwrap()
        .success();
    assert!(!alive, "grandchild {} survived the timeout", pid.trim());
}

/// A prompt the model's window cannot hold is refused before `fm` runs.
#[test]
fn an_oversized_prompt_is_refused_before_spawning() {
    let env = crate::test_env::TestEnv::new();
    let conf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("conf");
    env.set_managed("ZTOOLS_CONF_DIR", conf.as_os_str());
    let dir = tempfile::tempdir().unwrap();
    let fm = stub(dir.path(), "touch ran; cat");
    let why = respond(&fm, &"word ".repeat(100_000), 10)
        .unwrap_err()
        .to_string();
    assert!(why.contains("prompt does not fit"), "{why}");
    assert!(!dir.path().join("ran").exists());
    drop(env);
}

/// A child that answers while the prompt is still arriving cannot wedge the
/// call: the pipes drain while the prompt is fed.
#[test]
fn a_large_prompt_echoed_back_does_not_deadlock() {
    let dir = tempfile::tempdir().unwrap();
    let fm = stub(dir.path(), "cat");
    let prompt = "x".repeat(16_000);
    assert_eq!(respond(&fm, &prompt, 10).unwrap().len(), 16_000);
}
