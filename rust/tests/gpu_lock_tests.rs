//! Integration tests for `eval/gpu_lock.rs` through its env seam
//! (`ZTOOLS_GPU_LOCK_DIR`) and explicit-path APIs, so the machine-wide lock at
//! `/tmp/mac-osaurus-gpu.lock` is never touched.
//!
//! Live-owner cases use a real spawned `sleep` child as the foreign holder --
//! `kill(pid, 0)` and `ps -o lstart=` must agree on a genuine process for the
//! recycled-PID guard to mean anything.

use std::process::{Child, Command};
use std::time::Duration;

use serial_test::serial;
use tempfile::TempDir;
use ztools::eval::gpu_lock::{
    foreign_holder, is_expired, is_owner_alive, lock_dir, read_owner, start_time, GpuLockGuard,
    DEFAULT_LOCK_DIR, OWNER_ENV,
};

/// A pid no process can have: above any sane pid_max, below i32::MAX so the
/// `kill(pid as i32, 0)` cast stays positive.
const IMPOSSIBLE_PID: u32 = 2_000_000_000;

/// Restore an env var (or its absence) when the test ends -- a leaked
/// ZTOOLS_GPU_LOCK_OWNER would make later inherits/failures in THIS binary,
/// and env vars cross into nothing else, but serial tests share the process.
struct EnvGuard {
    name: &'static str,
    prev: Option<std::ffi::OsString>,
}

impl EnvGuard {
    fn set(name: &'static str, value: &str) -> Self {
        let prev = std::env::var_os(name);
        std::env::set_var(name, value);
        Self { name, prev }
    }

    fn unset(name: &'static str) -> Self {
        let prev = std::env::var_os(name);
        std::env::remove_var(name);
        Self { name, prev }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match self.prev.take() {
            Some(v) => std::env::set_var(self.name, v),
            None => std::env::remove_var(self.name),
        }
    }
}

fn live_child() -> Child {
    Command::new("sleep")
        .arg("30")
        .spawn()
        .expect("sleep spawns everywhere")
}

/// Write a complete owner file the way `acquire_at` would.
fn write_owner(dir: &std::path::Path, pid: u32, start: &str, label: &str) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("owner"), format!("{pid}\n{start}\n{label}\n")).unwrap();
}

// --- lock_dir env seam -------------------------------------------------------

#[test]
#[serial]
fn lock_dir_defaults_when_env_absent_or_empty() {
    let _g = EnvGuard::unset(ztools::eval::gpu_lock::DIR_ENV);
    assert_eq!(lock_dir(), std::path::PathBuf::from(DEFAULT_LOCK_DIR));
    let _g2 = EnvGuard::set(ztools::eval::gpu_lock::DIR_ENV, "");
    assert_eq!(lock_dir(), std::path::PathBuf::from(DEFAULT_LOCK_DIR));
}

#[test]
#[serial]
fn lock_dir_prefers_nonempty_env_value() {
    let tmp = TempDir::new().unwrap();
    let seam = tmp.path().join("seam.lock");
    let _g = EnvGuard::set(ztools::eval::gpu_lock::DIR_ENV, seam.to_str().unwrap());
    assert_eq!(lock_dir(), seam);
}

// --- start_time --------------------------------------------------------------

#[test]
#[serial]
fn start_time_of_impossible_pid_is_empty_not_a_panic() {
    assert_eq!(start_time(IMPOSSIBLE_PID), "");
}

// --- read_owner --------------------------------------------------------------

#[test]
fn read_owner_rejects_missing_short_and_blank_records() {
    let tmp = TempDir::new().unwrap();
    // No owner file at all.
    assert_eq!(read_owner(tmp.path()), None);
    // Fewer than three lines.
    std::fs::write(tmp.path().join("owner"), "123\n").unwrap();
    assert_eq!(read_owner(tmp.path()), None);
    // Blank pid line is not an owner.
    std::fs::write(tmp.path().join("owner"), "\nx\ny\n").unwrap();
    assert_eq!(read_owner(tmp.path()), None);
    // A well-formed record parses back out.
    write_owner(tmp.path(), 42, "Fri Aug 21 10:00:00 2026", "lab");
    assert_eq!(
        read_owner(tmp.path()),
        Some(("42".into(), "Fri Aug 21 10:00:00 2026".into(), "lab".into()))
    );
}

// --- is_owner_alive ----------------------------------------------------------

#[test]
#[serial]
fn is_owner_alive_rejects_missing_garbage_dead_and_recycled_pids() {
    let tmp = TempDir::new().unwrap();

    // Missing owner file.
    assert!(!is_owner_alive(tmp.path()));

    // Unparsable pid.
    write_owner(tmp.path(), 1, "x", "garbage");
    std::fs::write(tmp.path().join("owner"), "not-a-pid\nx\nlab\n").unwrap();
    assert!(!is_owner_alive(tmp.path()));

    // Dead pid: kill(pid, 0) fails.
    write_owner(tmp.path(), IMPOSSIBLE_PID, "whenever", "dead");
    assert!(!is_owner_alive(tmp.path()));

    // Recycled PID: live pid, but recorded start time is not this process's.
    let me = std::process::id();
    write_owner(tmp.path(), me, "Thu Jan  1 00:00:00 1970", "recycled");
    assert!(
        !is_owner_alive(tmp.path()),
        "a live pid with a foreign start time must count as dead"
    );

    // Truly ours: live pid AND matching start time.
    let st = start_time(me);
    write_owner(tmp.path(), me, &st, "self");
    assert!(is_owner_alive(tmp.path()));
}

// --- is_expired --------------------------------------------------------------

#[test]
fn is_expired_handles_missing_future_and_stale_mtimes() {
    let tmp = TempDir::new().unwrap();
    let lock = tmp.path().join("exp.lock");
    // Missing directory: nothing to expire.
    assert!(!is_expired(&lock, Duration::from_secs(1)));

    std::fs::create_dir_all(&lock).unwrap();
    // Fresh mtime, generous idle budget.
    assert!(!is_expired(&lock, Duration::from_secs(3600)));

    // Future mtime: duration_since fails, treated as NOT expired (a clock
    // skew must not let us stomp a live holder).
    let future = std::time::SystemTime::now() + Duration::from_secs(3600);
    std::fs::File::open(&lock)
        .unwrap()
        .set_modified(future)
        .unwrap();
    assert!(!is_expired(&lock, Duration::from_secs(1)));

    // Stale mtime past the idle budget.
    let stale = std::time::SystemTime::now() - Duration::from_secs(3600);
    std::fs::File::open(&lock)
        .unwrap()
        .set_modified(stale)
        .unwrap();
    assert!(is_expired(&lock, Duration::from_secs(1)));
}

// --- foreign_holder ----------------------------------------------------------

#[test]
#[serial]
fn foreign_holder_is_none_for_missing_dead_and_own_locks() {
    let _g = EnvGuard::unset(OWNER_ENV);
    let tmp = TempDir::new().unwrap();
    let lock = tmp.path().join("fh.lock");

    // Nothing there at all.
    assert_eq!(foreign_holder(), None);

    // A dead holder is nobody.
    write_owner(&lock, IMPOSSIBLE_PID, "whenever", "corpse");
    let _d = EnvGuard::set(ztools::eval::gpu_lock::DIR_ENV, lock.to_str().unwrap());
    assert_eq!(foreign_holder(), None);

    // Ourselves holding it is not foreign.
    let me = std::process::id();
    write_owner(&lock, me, &start_time(me), "me");
    assert_eq!(foreign_holder(), None);

    // An inherited owner matching ZTOOLS_GPU_LOCK_OWNER is also not foreign.
    let mut child = live_child();
    write_owner(&lock, child.id(), &start_time(child.id()), "parent run");
    let _o = EnvGuard::set(OWNER_ENV, &child.id().to_string());
    assert_eq!(
        foreign_holder(),
        None,
        "the inherited owner must read as self"
    );
    let _ = child.kill();
    let _ = child.wait(); // reap, or clippy flags a zombie
}

#[test]
#[serial]
fn foreign_holder_names_the_live_foreign_run() {
    let _g = EnvGuard::unset(OWNER_ENV);
    let _d_env = EnvGuard::unset(ztools::eval::gpu_lock::DIR_ENV);
    let tmp = TempDir::new().unwrap();
    let lock = tmp.path().join("foreign.lock");
    let mut child = live_child();

    // Labeled holder: reported verbatim.
    write_owner(
        &lock,
        child.id(),
        &start_time(child.id()),
        "other eval sweep",
    );
    let _d = EnvGuard::set(ztools::eval::gpu_lock::DIR_ENV, lock.to_str().unwrap());
    assert_eq!(foreign_holder().as_deref(), Some("other eval sweep"));

    // Unlabeled holder: reported as unknown rather than an empty string.
    std::fs::write(
        lock.join("owner"),
        format!("{}\n{}\n\n", child.id(), start_time(child.id())),
    )
    .unwrap();
    assert_eq!(foreign_holder().as_deref(), Some("an unknown run"));

    let _ = child.kill();
    let _ = child.wait(); // reap, or clippy flags a zombie
}

// --- acquire_at --------------------------------------------------------------

#[test]
#[serial]
fn acquire_inherits_a_lock_already_held_by_our_own_chain() {
    let tmp = TempDir::new().unwrap();
    let lock = tmp.path().join("inherited.lock");
    let mut child = live_child();
    write_owner(&lock, child.id(), &start_time(child.id()), "parent run");

    let _o = EnvGuard::set(OWNER_ENV, &child.id().to_string());
    let guard = GpuLockGuard::acquire_at(
        &lock,
        "child run",
        Duration::from_secs(1),
        Duration::from_secs(3600),
    )
    .unwrap();
    assert!(
        !guard.acquired,
        "inheriting must not take over or rewrite the owner file"
    );
    // Heartbeat on an inherited (non-acquired) guard is a deliberate no-op.
    guard.heartbeat();
    drop(guard);
    // Dropping an inherited guard must NOT delete the parent's lock.
    assert!(lock.exists(), "inherited lock must survive the guard drop");
    let _ = child.kill();
    let _ = child.wait(); // reap, or clippy flags a zombie
}

#[test]
#[serial]
fn acquire_ignores_an_inherited_owner_that_matches_nobody_and_falls_through() {
    let tmp = TempDir::new().unwrap();
    let lock = tmp.path().join("mismatch.lock");
    // Lock held by THIS process; ZTOOLS_GPU_LOCK_OWNER names some other live
    // process entirely -- the inheritance shortcut must not fire.
    let me = std::process::id();
    write_owner(&lock, me, &start_time(me), "self-held");
    let mut child = live_child();
    let _o = EnvGuard::set(OWNER_ENV, &child.id().to_string());

    let err = GpuLockGuard::acquire_at(
        &lock,
        "impatient",
        Duration::from_millis(120),
        Duration::from_secs(3600),
    )
    .err()
    .expect("a foreign-named inheritance must fall through to normal rules");
    assert!(
        err.to_string().contains("still held by"),
        "must end in the timeout refusal, not a false inherit: {err}"
    );
    assert!(lock.exists());
    std::fs::remove_dir_all(&lock).unwrap();
    let _ = child.kill();
    let _ = child.wait(); // reap, or clippy flags a zombie
}

#[test]
#[serial]
fn acquire_times_out_naming_the_holder_instead_of_stomping_it() {
    let _g = EnvGuard::unset(OWNER_ENV);
    let tmp = TempDir::new().unwrap();
    let lock = tmp.path().join("held.lock");
    let me = std::process::id();
    write_owner(&lock, me, &start_time(me), "busy sweep (pid x)");

    let started = std::time::Instant::now();
    let err = GpuLockGuard::acquire_at(
        &lock,
        "impatient",
        Duration::from_millis(120),
        Duration::from_secs(3600),
    )
    .err()
    .expect("must refuse while a live holder keeps the lock");
    assert!(
        err.to_string().contains("busy sweep"),
        "timeout must name the holder: {err}"
    );
    assert!(started.elapsed() >= Duration::from_millis(120));
    // The refused acquisition must not have deleted the holder's lock.
    assert!(lock.exists());
    std::fs::remove_dir_all(&lock).unwrap();
}

#[test]
#[serial]
fn acquire_reclaims_an_expired_lock_even_with_a_live_holder() {
    let _g = EnvGuard::unset(OWNER_ENV);
    let tmp = TempDir::new().unwrap();
    let lock = tmp.path().join("wedged.lock");
    let me = std::process::id();
    write_owner(&lock, me, &start_time(me), "long-dead sweep");

    // Let the directory mtime age past the tiny idle budget.
    thread_sleep(Duration::from_millis(60));
    let stale = std::time::SystemTime::now() - Duration::from_millis(200);
    std::fs::File::open(&lock)
        .unwrap()
        .set_modified(stale)
        .unwrap();

    let guard = GpuLockGuard::acquire_at(
        &lock,
        "fresh",
        Duration::from_secs(2),
        Duration::from_millis(50),
    )
    .unwrap();
    assert!(guard.acquired, "a wedged lock must be reclaimable");
    let owner = read_owner(&lock).unwrap();
    assert_eq!(
        owner.2.trim(),
        "fresh (pid {me})".replace("{me}", &me.to_string())
    );
}

#[test]
#[serial]
fn acquire_fails_hard_when_the_directory_cannot_be_created() {
    let tmp = TempDir::new().unwrap();
    // Parent exists but is a FILE: create_dir fails with something other than
    // AlreadyExists, which must bail loudly instead of looping.
    let blocker = tmp.path().join("blocker");
    std::fs::write(&blocker, b"i am a file").unwrap();
    let impossible = blocker.join("sub.lock");
    let err = GpuLockGuard::acquire_at(
        &impossible,
        "doomed",
        Duration::from_secs(1),
        Duration::from_secs(60),
    )
    .err()
    .expect("uncreatable lock dir must be a hard failure");
    assert!(
        err.to_string().contains("failed to create GPU lock dir"),
        "{err}"
    );
}

#[test]
#[serial]
fn drop_leaves_a_lock_whose_owner_file_names_someone_else() {
    let _g = EnvGuard::unset(OWNER_ENV);
    let tmp = TempDir::new().unwrap();
    let lock = tmp.path().join("handoff.lock");
    let guard = GpuLockGuard::acquire_at(
        &lock,
        "first",
        Duration::from_secs(2),
        Duration::from_secs(3600),
    )
    .unwrap();
    assert!(guard.acquired);

    // Someone rewrites the owner underneath us (simulated handoff/corruption).
    write_owner(&lock, IMPOSSIBLE_PID, "whenever", "someone else");
    drop(guard);

    assert!(
        lock.exists(),
        "a guard must not delete a lock whose owner file is no longer ours"
    );
    std::fs::remove_dir_all(&lock).unwrap();
}

#[test]
#[serial]
fn acquire_default_path_follows_the_dir_env_seam() {
    let tmp = TempDir::new().unwrap();
    let lock = tmp.path().join("via_seam.lock");
    let _d = EnvGuard::set(ztools::eval::gpu_lock::DIR_ENV, lock.to_str().unwrap());
    let _o = EnvGuard::unset(OWNER_ENV);

    let guard = GpuLockGuard::acquire(
        "seam run",
        Duration::from_secs(2),
        Duration::from_secs(3600),
    )
    .unwrap();
    assert!(guard.acquired);
    assert!(lock.exists());
    assert!(read_owner(&lock).is_some());
    drop(guard);
    assert!(!lock.exists(), "drop must clean up the seam-placed lock");
}

fn thread_sleep(d: Duration) {
    std::thread::sleep(d);
}
