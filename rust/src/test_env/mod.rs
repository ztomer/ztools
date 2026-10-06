//! ONE sandbox for every environment variable this crate reads a path out of.
//!
//! WHY IT EXISTS. Three instances of one defect were found on 2026-10-04: a
//! test that set `EVAL_SIGNALS_DIR` but not `EVAL_OUTPUT_DIR` wrote
//! `~/.config/ztools/outputs/gone-model/t1.txt` on every run; four tests that
//! passed `ZtoolsConfig::default()` were safe only because a branch elsewhere
//! happened not to read the cache; and a cookie test walked the developer's
//! real Firefox profile tree. Each had hand-rolled its own list of one or two
//! variables, so the next `forget` was one line away. This module is that next
//! line, pre-paid: one list, one lock, one restore.
//!
//! WHAT IT SANDBOXES, and why each entry is here. Every variable below is
//! READ by production code to decide where on disk to look or write.
//!
//! | variable | read at | pointed at |
//! |---|---|---|
//! | `HOME` | `dirs::home_dir()`, everywhere | empty dir |
//! | `XDG_CONFIG_HOME` / `_CACHE_` / `_DATA_` / `_STATE_` | `dirs` (Linux); see note | empty dirs |
//! | `EVAL_SIGNALS_DIR` | `eval::signals::signals_path` | empty |
//! | `EVAL_OUTPUT_DIR` | `eval::outputs::outputs_dir` | empty |
//! | `ZTOOLS_CONF_DIR` | `eval::budgets::conf_root`, `model_resolve::disk` | empty |
//! | `ZTOOLS_GPU_LOCK_DIR` | `eval::gpu_lock::lock_dir` | absent (absent = free) |
//! | `ZTOOLS_GPU_LOCK_OWNER` | `eval::gpu_lock` | unset (an inherited owner) |
//! | `MLX_MODELS_DIR` | `model_resolve::disk::models_dir` | empty |
//! | `HF_HOME` | `model_resolve::disk::hf_cache_dir` | empty |
//! | `TWITTER_OUTPUT_DIR` | `store::twitter_latest`, `twitter_status` | empty |
//! | `WEEKEND_OUTPUT_DIR` | `store::weekend_output_dir` | empty |
//! | `TWITTER_PROFILE_DIR` | `twitter::session::profile_dir` | empty |
//! | `CAMOUFOX_BIN` | `twitter::browser_bin` | unset (no browser to find) |
//! | `ZTOOLS_EXE` | `manifest::running_exe` | absent path, no checkout above it |
//! | `ZTOOLS_TLS_PROBE`, `ZTOOLS_TLS_PROBE_URL` | `tests/tls_probe.rs` | unset (the opt-in is off) |
//! | `EVAL_SAVE_OUTPUTS`, `EVAL_MAX_SAVED_OUTPUT`, `EVAL_DEFAULT_TIMEOUT`, `EVAL_MAX_TIMEOUT`, `EVAL_MODEL_STALL_SECONDS`, `EVAL_ALLOW_OVERSIZE`, `TWITTER_MAX_RUNTIME_S`, `TWITTER_FOLLOWING_TAB`, `TWITTER_FALLBACK_MODELS`, `ZTOOLS_UPDATE_GOLDENS` | `eval::outputs`, `eval::signals`, `eval::watchdog`, `eval::oversize`, `twitter`, the golden writers | unset (documented defaults) |
//!
//! The last row are POLICY knobs, not paths, and they are cleared for the same
//! reason: an exported `EVAL_MAX_TIMEOUT=99999` in the operator's shell
//! decides a test's outcome exactly as a path does. A test that wants a value
//! sets it AFTER construction, so what it asserts is what it wrote.
//!
//! `$HOME` is the load-bearing row, and it was the last one to be redirected.
//! `dirs::home_dir()` is process-global, and while a few files' tests resolved
//! a `~`-bearing default through it WITHOUT taking this guard, redirecting
//! `$HOME` broke them on a schedule: measured, it turned `weekend_fetch_tests`
//! and three twitter tests red in the same run, which is a worse defect than
//! the leak it was added to fix. So it sat MANAGED-but-PRESERVED from
//! 2026-10-04 until 2026-10-05, with the files still outside the contract
//! named one by one in `audit.rs`. Two things had to be true before the flip:
//! every test that reads `~` takes this guard, and the tests that need real
//! shipped DATA (`conf/weekend.toml`) pass it through a `*_under` / `*_from`
//! seam rather than through `$HOME`. Both hold; `Point::Preserved` no longer
//! exists, because a variant with no member is a decision nobody can audit.
//!
//! A test that wants the operator's real files asks for them by PATH
//! (`env!("CARGO_MANIFEST_DIR")`), which is the only form a peer cannot change
//! underneath it. A test that wants a fake `~` gets this sandbox. There is no
//! third answer, and that is the point: the previous policy had one, and it
//! was "whichever test ran last won".
//!
//! The `XDG_*` row is a hedge rather than a current read: this crate spells its
//! `~/.config` paths from `dirs::home_dir()` rather than from `dirs::config_dir()`,
//! so `HOME` alone covers macOS. `XDG_*` is what `dirs` uses on Linux x86-64
//! and aarch64 (both still supported by standing policy) and what the vendored
//! `camoufox` crate resolves its cache through, so they are redirected here
//! rather than left as a fourth thing to remember.
//!
//! WHY A LOCK AND NOT `#[serial]`. `serial_test::serial` is ONE-SIDED: it
//! excludes serial tests from each other, not from non-serial siblings, and
//! cargo runs every non-serial test in the binary concurrently. A guarded test
//! that reads a variable another guarded test just wrote is a flake with a
//! schedule. So the guard takes a process-wide lock for its WHOLE lifetime and
//! a test is serialised by constructing it, whether or not it is annotated.
//!
//! HOW THE LIST STAYS HONEST. `audit.rs` re-derives the set of variables the
//! crate reads out of the sources on every test run and fails if one is not
//! managed here, so a newly-read variable cannot be added without this file
//! knowing about it. That gate is what makes "impossible to forget" a property
//! of the system rather than of whoever writes the next test.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

#[cfg(test)]
mod audit;
#[cfg(test)]
mod restore_tests;
mod table;

use table::{CLEARED, MANAGED, Point, REDIRECTED};

/// THE CRATE'S ONLY ENVIRONMENT WRITER. Edition 2024 makes `set_var` and
/// `remove_var` `unsafe` (they call POSIX `setenv`, which is not thread-safe
/// against a concurrent `getenv`), and the alternative — an `unsafe` block at
/// each of the sites that need one — puts the soundness argument in N places,
/// where it holds in at most one.
///
/// Routing every TEST-side write through these two functions is what makes the
/// argument single: apart from the two production writes in `eval::gpu_lock`,
/// which carry their own, there is no other way to touch the environment from a
/// test in this crate -- so "no other writer" is a fact about the module rather
/// than a habit.
///
/// # Safety of the invariant
///
/// The precondition is that no OTHER thread reads the environment while one of
/// these runs. `ENV_LOCK` establishes it against every other sandbox, for this
/// thread's whole sandbox lifetime, because the lock is released after
/// [`TestEnv::drop`] has finished restoring. What `ENV_LOCK` does NOT establish
/// is exclusion against a thread that reads the environment without taking it.
/// That set used to be non-empty -- the audit gate in `audit.rs` named nine
/// files whose tests resolved a `~/…`-bearing default through
/// `dirs::home_dir()` on purpose -- and it is now empty: gate 2 fails if any
/// test that can reach the operator's home does not construct this guard, and
/// there are no allowlisted files left to hide one behind. That residual
/// window is now closed rather than recorded, which is the only reason the
/// `$HOME` row above can be a redirect.
fn put(key: &str, value: &OsStr) {
    // SAFETY: `ENV_LOCK` is held for the whole sandbox lifetime (see the
    // invariant above and `TestEnv::new`), so no other sandbox in this process
    // is mutating the environment, and `put`/`clear` are the only writers any
    // test can reach.
    unsafe { std::env::set_var(key, value) }
}

/// Remove a variable. The counterpart of [`put`]; see its invariant.
///
/// # Panics
///
/// Never. (Documented because it sits beside the panicking constructors.)
fn clear(key: &str) {
    // SAFETY: as `put`, and the same lock hold.
    unsafe { std::env::remove_var(key) }
}

/// Held for the whole lifetime of every [`TestEnv`].
///
/// Deliberately a plain `Mutex<()>` and not the env guard: this is a
/// cross-test exclusion, and the value is never read.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// A per-test sandbox: every managed variable redirected, the previous values
/// restored on drop.
///
/// # Panics
///
/// [`Self::new`] panics if the per-test temp dir cannot be created, which on
/// this platform means the machine has no usable temporary directory at all.
#[derive(Debug)]
pub struct TestEnv {
    /// Captured in the order the variables were changed, restored in reverse.
    saved: Vec<(&'static str, Option<OsString>)>,
    dir: tempfile::TempDir,
    /// Declared LAST so it is released after `drop` has restored the
    /// environment and after `dir` has been removed: no other test can observe
    /// a half-torn-down sandbox, and no test can read a variable the previous
    /// test has already restored.
    ///
    /// `None` only for a caller already inside [`with_env_lock`], which owns
    /// the lock for the whole closure -- including the assertions that come
    /// after the sandbox is gone. Owning it here would release it halfway
    /// through such a test and leave the tail racing every other sandbox.
    lock: Option<MutexGuard<'static, ()>>,
}

impl TestEnv {
    /// Take the lock, capture every managed variable, and redirect it into a
    /// fresh temp dir.
    ///
    /// BLOCKS until any other sandboxed test has finished. Calling this twice
    /// on one thread deadlocks, which is the intended reading: one sandbox at
    /// a time.
    ///
    /// The lock is taken BEFORE anything is redirected, and the order is
    /// load-bearing rather than cosmetic. This used to build the sandbox first
    /// and take the lock after, which meant a thread that was waiting for the
    /// lock had ALREADY overwritten the environment the lock holder was relying
    /// on: the holder's `~`-bearing defaults resolved into a temp dir it did
    /// not own, and its restore put the *waiter's* values back. Nothing failed
    /// loudly — `outputs.rs`'s "the guard is what redirects it" test simply
    /// started comparing one peer's sandbox against another's. `put` documents
    /// that the lock is what makes its write sound; a write taken before the
    /// lock is the one case where that is untrue.
    ///
    /// # Panics
    ///
    /// If the per-test temp dir cannot be created.
    #[must_use]
    pub fn new() -> Self {
        let taken = env_lock();
        let mut env = Self::redirected();
        env.lock = Some(taken);
        env
    }

    /// What [`Self::new`] does, minus taking the lock.
    ///
    /// For a caller already inside [`with_env_lock`], which holds the lock for
    /// its whole body: a test that has to set a managed variable's AMBIENT
    /// value -- to prove the restore -- has to hold the lock before it does
    /// that, and a non-reentrant `Mutex` taken twice on one thread deadlocks.
    ///
    /// # Panics
    ///
    /// If the per-test temp dir cannot be created.
    pub(crate) fn redirected() -> Self {
        let dir = tempfile::tempdir().expect("a per-test temp dir is always creatable");
        let mut saved = Vec::with_capacity(REDIRECTED.len() + CLEARED.len());
        for (key, point) in REDIRECTED {
            saved.push((*key, std::env::var_os(key)));
            match point {
                Point::Empty(sub) => {
                    let path = dir.path().join(sub);
                    std::fs::create_dir_all(&path).expect("a sandbox subdir is creatable");
                    put(key, path.as_os_str());
                }
                Point::Absent(sub) => {
                    // Set, NOT created: `create_dir` here would be a lock
                    // somebody else holds.
                    put(key, dir.path().join(sub).as_os_str());
                }
                Point::Unset => clear(key),
            }
        }
        for key in CLEARED {
            saved.push((key, std::env::var_os(key)));
            clear(key);
        }
        Self {
            saved,
            dir,
            lock: None,
        }
    }

    /// The sandbox root: unique per test, removed on drop.
    #[must_use]
    pub fn root(&self) -> &Path {
        self.dir.path()
    }

    /// The directory a sandboxed path variable points at.
    ///
    /// # Panics
    ///
    /// If `key` is not one of [`REDIRECTED`]. A key this guard does not manage
    /// is a variable nothing will restore, which is the leak this type exists
    /// to make impossible, so it is a hard failure rather than a silent pass.
    #[must_use]
    pub fn path(&self, key: &str) -> PathBuf {
        let sub = Self::subdir(key);
        self.dir.path().join(sub)
    }

    /// Write `contents` to `name` inside the directory `key` points at, so the
    /// code under test reads a fixture instead of the operator's store.
    ///
    /// # Panics
    ///
    /// As [`Self::path`], or if the file cannot be written.
    pub fn write(&self, key: &str, name: &str, contents: &str) {
        let path = self.path(key).join(name);
        std::fs::write(&path, contents).expect("a fixture inside the sandbox is writable");
    }

    /// Point a managed path variable at a DIFFERENT subdirectory of the
    /// sandbox -- a fixture conf root, a fixture models tree -- creating it.
    ///
    /// # Panics
    ///
    /// As [`Self::path`].
    #[must_use]
    pub fn set_path(&self, key: &str, subdir: &str) -> PathBuf {
        Self::subdir(key);
        let path = self.dir.path().join(subdir);
        std::fs::create_dir_all(&path).expect("a sandbox subdir is creatable");
        put(key, path.as_os_str());
        path
    }

    /// Set a managed policy knob. It stays managed -- and restored -- because
    /// the sandbox cleared it.
    ///
    /// # Panics
    ///
    /// If `key` is not one of [`CLEARED`].
    pub fn set(&self, key: &str, value: &str) {
        assert!(
            CLEARED.contains(&key),
            "{key} is not a sandboxed policy knob"
        );
        put(key, OsStr::new(value));
    }

    /// Remove a managed variable so the code under test reaches its own
    /// documented fallback -- the only way to pin a default is to make the
    /// override genuinely absent.
    ///
    /// # Panics
    ///
    /// If `key` is not managed.
    pub fn unset(&self, key: &str) {
        Self::assert_managed(key);
        clear(key);
    }

    /// Set a managed variable to a value the redirect tables cannot express —
    /// a fixture whose name is not one of the table's fixed subdirectories.
    ///
    /// [`Self::set`] is for the policy knobs the sandbox clears,
    /// [`Self::set_path`] for the paths it redirects at a named subdirectory,
    /// and [`Self::fixture`] for a path variable pointed at a fresh
    /// subdirectory. Between them they cover every row of [`REDIRECTED`], so
    /// this is for the remaining case: a value the tables cannot name. It
    /// used to exist for exactly one variable — `$HOME`, which was
    /// [`Point::Preserved`] and therefore had no sandbox directory — and
    /// `$HOME` is a redirect now, so the case it was written for is closed.
    /// It stays because a test may still need to point a managed variable at a
    /// path of its own choosing, and the alternative is a raw `set_var`, which
    /// is the one thing this module exists to prevent.
    ///
    /// The `MANAGED` assertion is what keeps this a sandbox method rather than
    /// an escape hatch: the restore contract is unchanged, so an unmanaged key
    /// is still a hard failure rather than a value nothing will put back.
    ///
    /// # Panics
    ///
    /// If `key` is not managed.
    pub fn set_managed(&self, key: &str, value: &OsStr) {
        Self::assert_managed(key);
        put(key, value);
    }

    // --- builder forms ------------------------------------------------------
    //
    // These four exist so that the COMMON test shape -- take the lock, point one
    // variable at a fixture, assert -- is a single `let` whose guard is never
    // used again. That is not only tidier: `clippy::significant_drop_tightening`
    // (a `nursery` lint this crate runs) reports a scope guard whose last use is
    // early, and its suggested fix -- fold the guard into the one call that uses
    // it -- DROPS THE GUARD AT THE END OF THAT STATEMENT, which would release the
    // lock and restore the environment before the assertions ran. The lint is
    // wrong for every RAII scope guard, so the way to satisfy it without
    // suppressing it is to shape the code so there is no early last use to
    // report: the guard goes straight into the binding and the fixture comes out
    // beside it.

    /// Consuming [`Self::unset`], for a sandbox that only removes variables.
    ///
    /// # Panics
    ///
    /// If `key` is not managed.
    #[must_use]
    pub fn without(self, key: &str) -> Self {
        self.unset(key);
        self
    }

    /// Consuming [`Self::set_managed`], for a sandbox that only sets variables.
    ///
    /// # Panics
    ///
    /// If `key` is not managed.
    #[must_use]
    pub fn with(self, key: &str, value: &OsStr) -> Self {
        self.set_managed(key, value);
        self
    }

    /// Consuming [`Self::set`], for a sandbox that only turns a policy knob.
    ///
    /// # Panics
    ///
    /// If `key` is not a sandboxed policy knob.
    #[must_use]
    pub fn policy(self, key: &str, value: &str) -> Self {
        self.set(key, value);
        self
    }

    /// Consuming [`Self::set_path`], returning the fixture directory alongside
    /// the guard that will restore it.
    ///
    /// # Panics
    ///
    /// As [`Self::set_path`].
    #[must_use]
    pub fn fixture(self, key: &str, subdir: &str) -> (Self, PathBuf) {
        let path = self.set_path(key, subdir);
        (self, path)
    }

    /// Consuming [`Self::path`], for a sandbox whose fixture directory is the
    /// variable's OWN sandbox path -- [`Point::Absent`], which is deliberately
    /// not created because an existing lock directory IS a held lock.
    ///
    /// # Panics
    ///
    /// If `key` is not redirected to a sandbox directory.
    #[must_use]
    pub fn at(self, key: &str) -> (Self, PathBuf) {
        let path = self.path(key);
        (self, path)
    }

    /// The sandbox-relative directory a managed variable points at.
    fn subdir(key: &str) -> &'static str {
        Self::assert_managed(key);
        match REDIRECTED
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, point)| *point)
        {
            Some(Point::Empty(sub) | Point::Absent(sub)) => sub,
            // `Unset` has no directory: the variable has no value at all, so
            // there is nothing for `path()` to hand back.
            Some(Point::Unset) => {
                panic!("{key} is cleared rather than redirected to a sandbox directory")
            }
            None => unreachable!("assert_managed ran"),
        }
    }

    fn assert_managed(key: &str) {
        assert!(
            MANAGED.contains(&key),
            "{key} is not managed by TestEnv; a variable this guard does not know \
             about is one nothing will restore"
        );
    }
}

impl Default for TestEnv {
    fn default() -> Self {
        Self::new()
    }
}

/// Run `f` holding the sandbox lock for its whole body.
///
/// For a test that must set a managed variable's AMBIENT value to prove the
/// restore: that write is itself an environment mutation, so it has to happen
/// under the same lock as the redirect -- and under the same lock as the
/// assertions that come after the sandbox is dropped, which is why the sandbox
/// inside is built with [`TestEnv::redirected`] and the lock stays here.
pub fn with_env_lock<T>(f: impl FnOnce(MutexGuard<'static, ()>) -> T) -> T {
    // A test that panicked while holding the lock must not turn one failure
    // into a cascade of "the mutex is poisoned" panics: the environment is
    // restored by the unwinding, so what the lock guards is sound.
    let lock = ENV_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    f(lock)
}

/// The sandbox lock on its own, for a caller that needs it before it can do
/// anything else. Prefer [`with_env_lock`], which cannot be forgotten.
pub(crate) fn env_lock() -> MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Drop for TestEnv {
    fn drop(&mut self) {
        // Runs BEFORE the fields: the environment is whole again before the
        // directory it pointed at is deleted and before the lock is released.
        // Restored in reverse of capture, the usual unwind order.
        for (key, prev) in self.saved.drain(..).rev() {
            match prev {
                Some(value) => put(key, &value),
                None => clear(key),
            }
        }
    }
}
