//! Machine-wide mutual exclusion for the osaurus server and the GPU.
//!
//! Rust implementation matching `lib/gpu_lock.py` and `tools/gpu_lock.sh`:
//! same lock path (`/tmp/mac-osaurus-gpu.lock`), same owner format (`pid\nstart_time\nlabel\n`),
//! same staleness rules, and identical start-time normalisation.

use anyhow::{Result, bail};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant, SystemTime};

pub const DEFAULT_LOCK_DIR: &str = "/tmp/mac-osaurus-gpu.lock";
pub const DEFAULT_TIMEOUT_SECS: u64 = 60;
pub const DEFAULT_MAX_IDLE_SECS: u64 = 14400; // 4 hours
pub const OWNER_ENV: &str = "ZTOOLS_GPU_LOCK_OWNER";
pub const DIR_ENV: &str = "ZTOOLS_GPU_LOCK_DIR";

// `unsafe extern` because edition 2024 requires every extern block to say so:
// the functions declared inside are only safe to call under a caller-guaranteed
// invariant, which is exactly what `kill(pid, 0)` is (see `is_owner_alive`).
unsafe extern "C" {
    fn kill(pid: i32, sig: i32) -> i32;
}

/// WHY THE OWNER IS PUBLISHED IN THE ENVIRONMENT AND NOT KEPT IN THE STRUCT.
///
/// `GpuLockGuard` could hold the holder's pid as a field, and for everything
/// in this process that would be enough. It would not survive being SPAWNED,
/// which is the case the variable exists for: an eval that holds the lock runs
/// helper processes, each of which calls [`GpuLockGuard::acquire`] or
/// [`foreign_holder`] and must recognise the lock as INHERITED rather than
/// report its own parent as a foreign holder competing for the GPU. An
/// in-memory field dies with the process; the only channel a child inherits is
/// the environment. So the mutation is load-bearing and stays.
///
/// Consequence worth stating, since it is why the mutation looks redundant:
/// [`GpuLockGuard::drop`] clears the marker rather than restoring a previous
/// value. That is only correct because two guards cannot BOTH be `acquired` in
/// one process — the nested acquire in `model_eval::eval_model`, reached from
/// `cli_ztools`'s `--suite full` path, hits the `owner.0 == inherited`
/// shortcut above and returns `acquired: false`, and an `acquired: false`
/// guard's `drop` is a no-op. Carrying the previous value in the struct would
/// make that true by construction rather than by coincidence; it is not done
/// here because nothing can currently reach the state it would fix.
fn record_owner(pid: u32) {
    // SAFETY: no other thread reads the environment while this runs. Two
    // threads in this process write it at all, and they are these two
    // functions, called from the thread that acquires or drops the guard --
    // which is the process's main thread in every production path
    // (`cli_ztools` and `model_eval` both acquire on it). The readers are
    // `foreign_holder` and `acquire_at`, reached from `signals.rs`
    // (`machine_is_uncontended`) and from the eval loop, all on that same
    // thread.
    //
    // The threads this crate DOES spawn cannot read it either: the request
    // thread in `llm.rs` holds a socket and a channel and nothing else, and
    // the warm-up thread in `weekend/fetch.rs` builds its `reqwest::Client`
    // -- whose system-proxy snapshot is the only environment a request
    // touches, and reqwest takes it in `ClientBuilder::build()`, not per
    // request (reqwest 0.13.5 `proxy.rs:515` `Matcher::from_system`, called
    // from `client.rs:418`). No write overlaps that window.
    //
    // The residual, stated rather than claimed away: this invariant is upheld
    // by where the calls happen today, not by a lock. The next code path that
    // mutates the environment while a spawned thread is live would break it
    // silently. The test-side writes do not reach it -- `TestEnv` clears
    // `OWNER_ENV` and holds a process-wide lock for its whole lifetime.
    unsafe { std::env::set_var(OWNER_ENV, pid.to_string()) }
}

/// The counterpart of [`record_owner`]: stop advertising this process as the
/// holder so a later sibling in the same session is not told its own parent
/// holds a foreign lock.
fn forget_owner() {
    // SAFETY: as `record_owner`, and the same reasoning about which thread
    // this runs on.
    unsafe { std::env::remove_var(OWNER_ENV) }
}

#[must_use]
pub fn lock_dir() -> PathBuf {
    if let Ok(val) = std::env::var(DIR_ENV)
        && !val.is_empty()
    {
        return PathBuf::from(val);
    }
    PathBuf::from(DEFAULT_LOCK_DIR)
}

/// Retrieve process start time with whitespace normalized (matching Python/Bash cross-language contract).
#[must_use]
pub fn start_time(pid: u32) -> String {
    let output = Command::new("ps")
        .args(["-o", "lstart=", "-p", &pid.to_string()])
        .output();
    match output {
        Ok(out) if out.status.success() => {
            let raw = String::from_utf8_lossy(&out.stdout);
            raw.split_whitespace().collect::<Vec<_>>().join(" ")
        }
        _ => String::new(),
    }
}

/// Read (pid, `start_time`, label) from the lock's `owner` file.
#[must_use]
pub fn read_owner(dir: &Path) -> Option<(String, String, String)> {
    let owner_path = dir.join("owner");
    let content = fs::read_to_string(owner_path).ok()?;
    let lines: Vec<&str> = content.split('\n').collect();
    if lines.len() < 3 || lines[0].trim().is_empty() {
        return None;
    }
    Some((
        lines[0].trim().to_string(),
        lines[1].to_string(),
        lines[2].to_string(),
    ))
}

/// Check whether the process recorded as holding the lock is still alive and running.
#[must_use]
pub fn is_owner_alive(dir: &Path) -> bool {
    let Some(owner) = read_owner(dir) else {
        return false;
    };
    let pid: u32 = match owner.0.parse() {
        Ok(p) => p,
        Err(_) => return false,
    };

    // SAFETY: `kill` is unsafe because the pid and signal are unchecked kernel
    // arguments. Signal `0` is the POSIX liveness probe: it performs the
    // permission and existence checks and delivers nothing, so the worst a wrong
    // pid can do here is report a process as alive or dead. The pid is not
    // arbitrary -- `units::pid` is the crate's checked conversion, which is what
    // keeps the `i32` this ABI takes from truncating or going negative.
    let ret = unsafe { kill(crate::units::pid(pid), 0) };
    if ret != 0 {
        return false;
    }

    // Verify start time to prevent recycled PID collision
    let current_start = start_time(pid);
    owner.1.is_empty() || current_start.is_empty() || owner.1.trim() == current_start.trim()
}

/// Check whether the lock directory mtime has exceeded the max idle threshold.
#[must_use]
pub fn is_expired(dir: &Path, max_idle: Duration) -> bool {
    let Ok(meta) = fs::metadata(dir) else {
        return false;
    };
    let Ok(mtime) = meta.modified() else {
        return false;
    };
    SystemTime::now()
        .duration_since(mtime)
        .is_ok_and(|elapsed| elapsed >= max_idle)
}

fn force_remove(dir: &Path) {
    let _ = fs::remove_dir_all(dir);
}

#[must_use]
pub fn foreign_holder() -> Option<String> {
    let dir = lock_dir();
    if !is_owner_alive(&dir) {
        return None;
    }
    let owner = read_owner(&dir)?;
    let current_pid = std::process::id().to_string();
    let inherited = std::env::var(OWNER_ENV).unwrap_or_default();
    if owner.0 == current_pid || (!inherited.is_empty() && owner.0 == inherited) {
        return None;
    }
    let label = owner.2.trim();
    Some(if label.is_empty() {
        "an unknown run".to_string()
    } else {
        label.to_string()
    })
}

/// RAII GPU Lock Guard.
pub struct GpuLockGuard {
    dir: PathBuf,
    pub acquired: bool,
}

impl GpuLockGuard {
    /// Acquire the GPU lock at default path.
    ///
    /// # Errors
    ///
    /// As [`Self::acquire_at`]: the lock is still held by a live session after
    /// `timeout`, or the lock directory cannot be created.
    pub fn acquire(label: &str, timeout: Duration, max_idle: Duration) -> Result<Self> {
        Self::acquire_at(&lock_dir(), label, timeout, max_idle)
    }

    /// Acquire the GPU lock at a specific path with timeout and automatic stale/wedged owner reclamation.
    ///
    /// # Errors
    ///
    /// When the GPU is still held by a live session after `timeout` -- the
    /// message names the holder, because the right response is to wait rather
    /// than to break the lock while another eval is measuring -- and when the
    /// lock directory cannot be created for any other reason. A STALE or
    /// wedged owner is not an error: it is reclaimed and the lock is taken.
    pub fn acquire_at(
        dir: &Path,
        label: &str,
        timeout: Duration,
        max_idle: Duration,
    ) -> Result<Self> {
        let inherited = std::env::var(OWNER_ENV).unwrap_or_default();
        if !inherited.is_empty()
            && is_owner_alive(dir)
            && let Some(owner) = read_owner(dir)
            && owner.0 == inherited
        {
            return Ok(Self {
                dir: dir.to_path_buf(),
                acquired: false,
            });
        }

        let start = Instant::now();
        let pid = std::process::id();

        loop {
            match fs::create_dir(dir) {
                Ok(()) => {
                    let st = start_time(pid);
                    let owner_content = format!("{pid}\n{st}\n{label} (pid {pid})\n");
                    let _ = fs::write(dir.join("owner"), owner_content);
                    record_owner(pid);
                    return Ok(Self {
                        dir: dir.to_path_buf(),
                        acquired: true,
                    });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    if !is_owner_alive(dir) {
                        force_remove(dir);
                        continue;
                    }
                    if is_expired(dir, max_idle) {
                        force_remove(dir);
                        continue;
                    }
                    if start.elapsed() >= timeout {
                        let holder_label =
                            read_owner(dir).map_or_else(|| "an unknown run".to_string(), |o| o.2);
                        bail!(
                            "GPU still held by {holder_label} after {timeout:?} — that session is measuring; do not restart osaurus under it"
                        );
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(e) => bail!("failed to create GPU lock dir {}: {}", dir.display(), e),
            }
        }
    }

    /// Touch the lock directory mtime to indicate active progress.
    pub fn heartbeat(&self) {
        if !self.acquired {
            return;
        }
        let now = SystemTime::now();
        let _ = fs::File::open(&self.dir).and_then(|f| f.set_modified(now));
    }
}

impl Drop for GpuLockGuard {
    fn drop(&mut self) {
        if self.acquired {
            forget_owner();
            let owner = read_owner(&self.dir);
            let current_pid = std::process::id().to_string();
            if let Some(o) = owner
                && o.0 == current_pid
            {
                force_remove(&self.dir);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_env::TestEnv;

    /// `OWNER_ENV` is MUTATED by production code -- set at
    /// [`GpuLockGuard::acquire_at`], removed in [`GpuLockGuard::drop`] -- so
    /// these tests had the same leak every other env-mutating test had: a
    /// panic between the two leaves the variable set for the rest of the
    /// binary, and `foreign_holder()` then reports every later lock as
    /// inherited rather than foreign. The guard clears it on the way in and
    /// restores it on the way out, and the last test here proves the round trip.
    #[test]
    #[serial_test::serial]
    fn test_lock_acquire_and_release() {
        let env = TestEnv::new();
        let temp = tempfile::tempdir().unwrap();
        let lock_path = temp.path().join("test_gpu.lock");

        {
            let guard = GpuLockGuard::acquire_at(
                &lock_path,
                "test-run",
                Duration::from_secs(2),
                Duration::from_secs(60),
            )
            .unwrap();
            assert!(lock_path.exists());
            assert!(lock_path.join("owner").exists());
            // Production SET the owner env on acquire.
            assert_eq!(
                std::env::var(OWNER_ENV).ok(),
                Some(std::process::id().to_string()),
                "acquiring records the holder for inheritance"
            );
            guard.heartbeat();
            assert!(!is_expired(&lock_path, Duration::from_secs(60)));
        }

        assert!(
            !lock_path.exists(),
            "lock directory should be cleaned up on drop"
        );
        assert!(
            std::env::var_os(OWNER_ENV).is_none(),
            "releasing clears the inherited owner, or every later test reads this \
             process as its own holder"
        );
        drop(env);
    }

    #[test]
    #[serial_test::serial]
    fn test_lock_reclaims_dead_owner() {
        let env = TestEnv::new();
        let temp = tempfile::tempdir().unwrap();
        let lock_path = temp.path().join("stale_gpu.lock");
        fs::create_dir_all(&lock_path).unwrap();
        // A fake dead PID (9999999)
        fs::write(
            lock_path.join("owner"),
            "9999999\nSun Aug 18 10:00:00 2026\nfake (pid 9999999)\n",
        )
        .unwrap();

        let guard = GpuLockGuard::acquire_at(
            &lock_path,
            "fresh-run",
            Duration::from_secs(2),
            Duration::from_secs(60),
        )
        .unwrap();
        assert!(guard.acquired);
        assert!(lock_path.exists());
        let owner = read_owner(&lock_path).unwrap();
        assert_eq!(owner.0, std::process::id().to_string());
        assert!(
            !is_owner_alive(&temp.path().join("never-locked")),
            "a directory with no owner file is nobody's lock"
        );
        drop(guard);
        drop(env);
    }

    /// The machine-wide lock path is unreachable from inside the sandbox, which
    /// is the whole reason `ZTOOLS_GPU_LOCK_DIR` is redirected and not cleared.
    #[test]
    #[serial_test::serial]
    fn the_sandboxed_lock_dir_is_not_the_machine_wide_one() {
        let env = TestEnv::new();
        assert_eq!(lock_dir(), env.path(DIR_ENV));
        assert_ne!(lock_dir(), PathBuf::from(DEFAULT_LOCK_DIR));
        assert!(
            !lock_dir().exists(),
            "an absent lock directory is a FREE lock, not a held one"
        );
        drop(env);
    }

    #[test]
    #[serial_test::serial]
    fn the_owner_env_survives_a_test_that_panics_mid_acquire() {
        // The property the guard buys: `OWNER_ENV` is process-global and a
        // panic between acquire and drop used to leave it set forever. Here it
        // is unwound through `catch_unwind` while the guard is alive, and the
        // value must be gone afterwards.
        crate::test_env::with_env_lock(|_lock| {
            let env = TestEnv::redirected();
            let temp = tempfile::tempdir().unwrap();
            let lock_path = temp.path().join("panicking.lock");
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _guard = GpuLockGuard::acquire_at(
                    &lock_path,
                    "panicking-run",
                    Duration::from_secs(2),
                    Duration::from_secs(60),
                )
                .unwrap();
                assert!(std::env::var(OWNER_ENV).is_ok(), "acquire set the owner");
                panic!("unwind with the lock held");
            }));
            assert!(result.is_err(), "the closure must have panicked");
            assert!(
                std::env::var_os(OWNER_ENV).is_none(),
                "dropping the guard through an unwind must clear the inherited \
                 owner; this is the leak that made a later test see its own pid \
                 as a foreign holder"
            );
            drop(env);
        });
    }
}
