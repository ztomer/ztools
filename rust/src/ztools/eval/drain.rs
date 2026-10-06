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

/// For tests: pretend the operator pressed Ctrl-C once, for as long as the
/// returned value is alive.
///
/// SCOPED, not a setter. `SIGNALS` is process-global and the flag used to be
/// raised and cleared by hand, so a panic in between leaked it into every
/// later test in the binary -- and `requested()` returning true for the rest
/// of the run is a silent pass, not a failure: the eval loops under test would
/// simply stop after zero tasks. `raise` also refuses to start if the flag is
/// already set, which is what makes a leak LOUD instead of invisible.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct DrainRequest;

#[cfg(test)]
impl DrainRequest {
    /// # Panics
    ///
    /// If the flag is already raised, which can only mean a previous test
    /// leaked it. That is a test bug with no innocent explanation, so it is
    /// named rather than absorbed.
    pub(crate) fn raise() -> Self {
        assert!(
            !requested(),
            "the drain flag was already set when this test started: a previous \
             test leaked it, and every later test would silently stop after \
             zero tasks"
        );
        SIGNALS.store(1, Ordering::SeqCst);
        Self
    }
}

#[cfg(test)]
impl Drop for DrainRequest {
    fn drop(&mut self) {
        SIGNALS.store(0, Ordering::SeqCst);
    }
}
