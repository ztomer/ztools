//! Which threads are inside a [`super::TestEnv`]: the runtime half of the
//! sandbox contract, for state a textual audit cannot follow (the GPU lock is
//! reached through call chains, not named in test bodies).

use std::cell::Cell;
use std::sync::atomic::{AtomicU32, Ordering};

thread_local! {
    /// How many [`super::TestEnv`] sandboxes THIS thread built and still holds.
    static SANDBOXED: Cell<u32> = const { Cell::new(0) };
}

/// How many [`super::TestEnv`] sandboxes are live in the process (0 or 1: the lock).
static LIVE: AtomicU32 = AtomicU32::new(0);

/// Whether a read or write of process-global lock state from this thread is
/// covered by a sandbox.
///
/// A libtest TEST thread is named after its test, so a named thread must hold
/// a [`super::TestEnv`] itself: a test that reaches the GPU lock without one reads
/// whichever `ZTOOLS_GPU_LOCK_DIR` a peer's sandbox has set at that instant
/// and takes, or releases, THAT test's lock under it (the 3.28.0 release
/// suite: `a_calibrated_summarize_model_pairs_under_the_lock_and_is_released`
/// saw every request unlocked). An UNNAMED thread is one a test spawned; it
/// cannot carry the thread-local, so it passes while some sandbox is live.
/// The residual is an unnamed thread spawned by an unguarded test while a
/// peer's sandbox is live, which this cannot tell apart.
#[must_use]
pub fn in_sandbox() -> bool {
    if SANDBOXED.with(Cell::get) > 0 {
        return true;
    }
    let named = std::thread::current().name().is_some_and(|n| n != "main");
    !named && LIVE.load(Ordering::SeqCst) > 0
}

/// Panic, naming `what`, when the calling test is outside every sandbox.
///
/// # Panics
///
/// When [`in_sandbox`] is false.
pub fn assert_sandboxed(what: &str) {
    assert!(
        in_sandbox(),
        "{what} touched outside a TestEnv: construct `TestEnv::new()` in this test \
         (process-global GPU-lock state is shared with every concurrent test)"
    );
}

/// Called by the sandbox when it is built.
pub(super) fn enter() {
    SANDBOXED.with(|d| d.set(d.get() + 1));
    LIVE.fetch_add(1, Ordering::SeqCst);
}

/// Called by the sandbox once it has restored the environment.
pub(super) fn leave() {
    SANDBOXED.with(|d| d.set(d.get().saturating_sub(1)));
    LIVE.fetch_sub(1, Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::assert_sandboxed;

    #[test]
    fn a_test_thread_outside_a_sandbox_is_refused_and_inside_one_is_not() {
        let named = std::thread::Builder::new()
            .name("a_test_with_no_sandbox".into())
            .spawn(|| assert_sandboxed("probe"))
            .unwrap()
            .join();
        let msg = named.expect_err("a named thread without a TestEnv is refused");
        let text = msg.downcast_ref::<String>().cloned().unwrap_or_default();
        assert!(text.contains("probe touched outside a TestEnv"), "{text}");
        let env = crate::test_env::TestEnv::new();
        assert_sandboxed("probe");
        drop(env);
    }
}
