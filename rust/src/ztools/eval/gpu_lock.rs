//! Machine-wide mutual exclusion for the osaurus server and the GPU.
//!
//! Rust implementation matching `lib/gpu_lock.py` and `tools/gpu_lock.sh`:
//! same lock path (`/tmp/mac-osaurus-gpu.lock`), same owner format (`pid\nstart_time\nlabel\n`),
//! same staleness rules, and identical start-time normalisation.

use anyhow::{bail, Result};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant, SystemTime};

pub const DEFAULT_LOCK_DIR: &str = "/tmp/mac-osaurus-gpu.lock";
pub const DEFAULT_TIMEOUT_SECS: u64 = 60;
pub const DEFAULT_MAX_IDLE_SECS: u64 = 14400; // 4 hours
pub const OWNER_ENV: &str = "ZTOOLS_GPU_LOCK_OWNER";
pub const DIR_ENV: &str = "ZTOOLS_GPU_LOCK_DIR";

extern "C" {
    fn kill(pid: i32, sig: i32) -> i32;
}

pub fn lock_dir() -> PathBuf {
    if let Ok(val) = std::env::var(DIR_ENV) {
        if !val.is_empty() {
            return PathBuf::from(val);
        }
    }
    PathBuf::from(DEFAULT_LOCK_DIR)
}

/// Retrieve process start time with whitespace normalized (matching Python/Bash cross-language contract).
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

/// Read (pid, start_time, label) from the lock's `owner` file.
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
pub fn is_owner_alive(dir: &Path) -> bool {
    let owner = match read_owner(dir) {
        Some(o) => o,
        None => return false,
    };
    let pid: u32 = match owner.0.parse() {
        Ok(p) => p,
        Err(_) => return false,
    };

    let ret = unsafe { kill(pid as i32, 0) };
    if ret != 0 {
        return false;
    }

    // Verify start time to prevent recycled PID collision
    let current_start = start_time(pid);
    owner.1.is_empty() || current_start.is_empty() || owner.1.trim() == current_start.trim()
}

/// Check whether the lock directory mtime has exceeded the max idle threshold.
pub fn is_expired(dir: &Path, max_idle: Duration) -> bool {
    let meta = match fs::metadata(dir) {
        Ok(m) => m,
        Err(_) => return false,
    };
    let mtime = match meta.modified() {
        Ok(t) => t,
        Err(_) => return false,
    };
    if let Ok(elapsed) = SystemTime::now().duration_since(mtime) {
        elapsed >= max_idle
    } else {
        false
    }
}

fn force_remove(dir: &Path) {
    let _ = fs::remove_dir_all(dir);
}

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
    pub fn acquire(label: &str, timeout: Duration, max_idle: Duration) -> Result<Self> {
        Self::acquire_at(&lock_dir(), label, timeout, max_idle)
    }

    /// Acquire the GPU lock at a specific path with timeout and automatic stale/wedged owner reclamation.
    pub fn acquire_at(
        dir: &Path,
        label: &str,
        timeout: Duration,
        max_idle: Duration,
    ) -> Result<Self> {
        let inherited = std::env::var(OWNER_ENV).unwrap_or_default();
        if !inherited.is_empty() && is_owner_alive(dir) {
            if let Some(owner) = read_owner(dir) {
                if owner.0 == inherited {
                    return Ok(Self {
                        dir: dir.to_path_buf(),
                        acquired: false,
                    });
                }
            }
        }

        let start = Instant::now();
        let pid = std::process::id();

        loop {
            match fs::create_dir(dir) {
                Ok(_) => {
                    let st = start_time(pid);
                    let owner_content = format!("{pid}\n{st}\n{label} (pid {pid})\n");
                    let _ = fs::write(dir.join("owner"), owner_content);
                    std::env::set_var(OWNER_ENV, pid.to_string());
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
                        let holder_label = read_owner(dir)
                            .map(|o| o.2)
                            .unwrap_or_else(|| "an unknown run".to_string());
                        bail!("GPU still held by {} after {:?} — that session is measuring; do not restart osaurus under it", holder_label, timeout);
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
            std::env::remove_var(OWNER_ENV);
            let owner = read_owner(&self.dir);
            let current_pid = std::process::id().to_string();
            if let Some(o) = owner {
                if o.0 == current_pid {
                    force_remove(&self.dir);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_start_time_current_process() {
        let pid = std::process::id();
        let st = start_time(pid);
        assert!(
            !st.is_empty(),
            "start_time for current pid should not be empty"
        );
        // Must contain day/month/time format without multiple adjacent whitespace
        assert!(
            !st.contains("  "),
            "start_time must have whitespace collapsed"
        );
    }

    #[test]
    fn test_lock_acquire_and_release() {
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
            guard.heartbeat();
            assert!(!is_expired(&lock_path, Duration::from_secs(60)));
        }

        assert!(
            !lock_path.exists(),
            "lock directory should be cleaned up on drop"
        );
    }

    #[test]
    fn test_lock_reclaims_dead_owner() {
        let temp = tempfile::tempdir().unwrap();
        let lock_path = temp.path().join("stale_gpu.lock");
        fs::create_dir_all(&lock_path).unwrap();
        // Write a fake dead PID (9999999)
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
    }
}
