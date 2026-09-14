//! One lock, two implementations: `tools/gpu_lock.sh` (bash, used by
//! `osaurus_one.sh`) and `eval::gpu_lock` (Rust, held by `model-eval`).
//!
//! If they disagree about the path or the owner-file format, each reads the
//! other's record as an impostor and silently grants a lock the peer is
//! holding — while both print reassuring "acquired" messages. These were the
//! `TestCrossLanguageParity` cases of the retired Python suite, re-expressed
//! against the implementation that replaced the Python half.
//!
//! Skips loudly when `gates_of_heck` (the shell lib the script sources) is not
//! checked out; never silently.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use ztools::eval::gpu_lock::{
    foreign_holder, is_owner_alive, read_owner, GpuLockGuard, DEFAULT_LOCK_DIR, DIR_ENV,
};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn goh_dir() -> Option<PathBuf> {
    let dir = std::env::var("GOH_DIR").map_or_else(
        |_| dirs::home_dir().unwrap().join("Projects/gates_of_heck"),
        PathBuf::from,
    );
    dir.join("tui/lib.sh").is_file().then_some(dir)
}

fn script(goh: &Path, body: &str) -> String {
    format!(
        "source \"{}/tui/lib.sh\"\nsource \"{}/tools/gpu_lock.sh\"\n{body}\n",
        goh.display(),
        repo().display()
    )
}

/// Run a shell snippet with the lock libs sourced, bounded so a wedged lock
/// cannot hang the suite.
fn sh(goh: &Path, body: &str, lock_dir: &Path) -> String {
    let out = Command::new("bash")
        .arg("-c")
        .arg(script(goh, body))
        .env(DIR_ENV, lock_dir)
        .env("NO_COLOR", "1")
        .env_remove("ZTOOLS_GPU_LOCK_OWNER")
        .current_dir(repo())
        .output()
        .expect("bash runs");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn both_halves_default_to_the_same_path() {
    let Some(goh) = goh_dir() else {
        eprintln!("SKIP: gates_of_heck not checked out; shell lock lib unavailable");
        return;
    };
    let td = tempfile::tempdir().unwrap();
    // Empty override: the shell falls back to its default, which must be Rust's.
    let out = Command::new("bash")
        .arg("-c")
        .arg(script(&goh, "printf \"%s\" \"$GPU_LOCK_DIR\""))
        .env(DIR_ENV, "")
        .env("NO_COLOR", "1")
        .current_dir(td.path())
        .output()
        .unwrap();
    assert_eq!(String::from_utf8_lossy(&out.stdout), DEFAULT_LOCK_DIR);
}

#[test]
#[serial_test::serial]
fn the_shell_half_can_read_a_rust_owner_file() {
    // The direction that matters most: an eval (Rust) holds the GPU and
    // osaurus_one.sh (bash) must see it.
    let Some(goh) = goh_dir() else {
        eprintln!("SKIP: gates_of_heck not checked out; shell lock lib unavailable");
        return;
    };
    let td = tempfile::tempdir().unwrap();
    let lock = td.path().join("gpu.lock");
    std::env::set_var(DIR_ENV, &lock);
    let guard = GpuLockGuard::acquire(
        "eval from rust",
        Duration::from_secs(5),
        Duration::from_secs(600),
    )
    .expect("fresh lock acquires");
    let out = sh(&goh, "gpu_lock_holder", &lock);
    drop(guard);
    std::env::remove_var(DIR_ENV);
    assert!(out.contains("eval from rust"), "shell saw: {out:?}");
}

#[test]
#[serial_test::serial]
fn rust_can_read_a_shell_owner_file() {
    let Some(goh) = goh_dir() else {
        eprintln!("SKIP: gates_of_heck not checked out; shell lock lib unavailable");
        return;
    };
    let td = tempfile::tempdir().unwrap();
    let lock = td.path().join("gpu.lock");
    // A live holder written by bash: keep the bash process alive while Rust
    // reads it, or the liveness check would correctly call it dead.
    let mut child = Command::new("bash")
        .arg("-c")
        .arg(script(
            &goh,
            "gpu_lock_acquire \"osaurus_one.sh --restart\" >/dev/null; read -r _",
        ))
        .env(DIR_ENV, &lock)
        .env("NO_COLOR", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .current_dir(repo())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while !lock.join("owner").exists() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    std::env::set_var(DIR_ENV, &lock);
    let owner = read_owner(&lock);
    let alive = is_owner_alive(&lock);
    let foreign = foreign_holder();
    std::env::remove_var(DIR_ENV);
    // Release the bash holder before asserting, so a failure cannot leak it.
    child.stdin.take().unwrap().write_all(b"\n").unwrap();
    let _ = child.wait();

    let (pid, _start, label) = owner.expect("owner file written by bash is readable");
    assert_eq!(pid, child.id().to_string());
    // The shell records "label (pid N)" as the label line; Rust reports it as is.
    let expected = format!("osaurus_one.sh --restart (pid {})", child.id());
    assert_eq!(label.trim(), expected);
    assert!(alive, "a live bash holder must read as alive");
    assert_eq!(foreign.as_deref(), Some(expected.as_str()));
}
