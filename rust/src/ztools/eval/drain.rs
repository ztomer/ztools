//! Drain mode for Ctrl-C (B4 deferral, closed 2026-09-19).
//!
//! The eval used to die on SIGINT mid-task: the GPU lock's dead-owner reclaim
//! kept the safety property, but the in-flight task's answer was lost and the
//! history got a run that stopped between two tasks with nothing saying why.
//! Now the first Ctrl-C asks the runner to finish the task it is on and stop,
//! so the outcomes so far are recorded, the completeness verdict says
//! "truncated", the lock is released by its guard, and the command exits
//! non-zero so a sweep files the run as FAILED and `--resume` re-runs it.
//! A second Ctrl-C is the old behaviour: leave now.

use std::sync::atomic::{AtomicU8, Ordering};

static SIGNALS: AtomicU8 = AtomicU8::new(0);

/// Install the handler once per process; a second install is a no-op.
pub fn install() {
    static INSTALLED: std::sync::Once = std::sync::Once::new();
    INSTALLED.call_once(|| {
        let _ = ctrlc::set_handler(|| {
            let n = SIGNALS.fetch_add(1, Ordering::SeqCst) + 1;
            if n == 1 {
                eprintln!(
                    "\n\u{26a0} Ctrl-C: finishing the task in flight, then stopping (press again to quit now)"
                );
            } else {
                eprintln!("\n\u{2717} Ctrl-C again: quitting now");
                std::process::exit(130);
            }
        });
    });
}

/// True once the operator has asked to stop.
#[must_use]
pub fn requested() -> bool {
    SIGNALS.load(Ordering::SeqCst) > 0
}

/// For tests: pretend the operator pressed Ctrl-C once.
#[cfg(test)]
pub(crate) fn request_for_test() {
    SIGNALS.store(1, Ordering::SeqCst);
}

/// For tests: clear the request.
#[cfg(test)]
pub(crate) fn reset_for_test() {
    SIGNALS.store(0, Ordering::SeqCst);
}
