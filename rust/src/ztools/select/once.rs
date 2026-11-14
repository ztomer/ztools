//! One night at a time (LM.INCUMBENTFAIL).
//!
//! The routines runner stamps a task when it finishes, and its per-minute tick
//! can start a three-hour task again while the first run is still going. A
//! second night beside the first measures nothing honestly: on 2026-11-13 and
//! 2026-11-14 the second start's incumbent eval could not take the GPU lock
//! the first held and failed the night three times, and a second night can
//! evaluate a candidate the first is still downloading. So the night claims
//! `state/nightly.lock` (a directory holding the owner's pid, start time and
//! label, the GPU lock's own format) for its whole run; a start that finds it
//! held by a live night does nothing, and a dead night's claim is taken over.

use crate::ztools::eval::gpu_lock::{is_owner_alive, read_owner, start_time};
use std::path::{Path, PathBuf};

/// The night's claim, released on drop.
#[derive(Debug)]
pub struct NightLock {
    dir: PathBuf,
}

/// What claiming found.
#[derive(Debug)]
pub enum Claim {
    Ours(NightLock),
    /// A live night holds it: its label.
    Held(String),
}

/// Claim the night under `state_dir`.
///
/// # Errors
/// When the claim cannot be created or a dead one removed.
pub fn claim(state_dir: &Path) -> std::io::Result<Claim> {
    std::fs::create_dir_all(state_dir)?;
    let dir = state_dir.join("nightly.lock");
    let pid = std::process::id();
    loop {
        match std::fs::create_dir(&dir) {
            Ok(()) => {
                let owner = format!(
                    "{pid}\n{}\nztools models nightly (pid {pid})\n",
                    start_time(pid)
                );
                std::fs::write(dir.join("owner"), owner)?;
                return Ok(Claim::Ours(NightLock { dir }));
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                if is_owner_alive(&dir) || being_claimed(&dir) {
                    let label = read_owner(&dir).map_or_else(|| "a night".into(), |o| o.2);
                    return Ok(Claim::Held(label));
                }
                std::fs::remove_dir_all(&dir)?;
            }
            Err(e) => return Err(e),
        }
    }
}

/// A claim whose owner is not written yet: another start is between creating
/// it and writing its owner, so it is not a dead night's.
fn being_claimed(dir: &Path) -> bool {
    read_owner(dir).is_none()
        && std::fs::metadata(dir)
            .and_then(|m| m.modified())
            .is_ok_and(|t| t.elapsed().is_ok_and(|e| e.as_secs() < 5))
}

impl Drop for NightLock {
    fn drop(&mut self) {
        if read_owner(&self.dir).is_some_and(|o| o.0 == std::process::id().to_string()) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }
}
