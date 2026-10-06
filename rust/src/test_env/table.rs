//! WHAT THE SANDBOX OWNS: the variables, where each one lands, and why `$HOME`
//! is a redirect like any other.
//!
//! Split out of `mod.rs` for the house 500-line cap, along the seam the file
//! already had: this is the DECLARATION (what is managed and why) and
//! `mod.rs` is the BEHAVIOUR (capture it, redirect it, restore it). The two
//! change for different reasons -- a table row is added when production code
//! starts reading a new variable, a method when a test needs a new way to point
//! one -- and `audit.rs` re-derives its expectations from the sources either
//! way, so a split cannot hide a disagreement.

/// What a managed variable is pointed at, relative to the sandbox root.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Point {
    /// An empty directory created inside the sandbox.
    Empty(&'static str),
    /// A path inside the sandbox that is deliberately NOT created: an existing
    /// GPU-lock directory IS a held lock, so the sandbox must start free.
    Absent(&'static str),
    /// Removed outright -- nothing in the sandbox stands in for it.
    Unset,
}

/// Every PATH variable the sandbox redirects, with where it lands.
pub(super) const REDIRECTED: &[(&str, Point)] = &[
    // `dirs::home_dir()` reads this first, and every `~/…` default in
    // `config.rs` and `store.rs` hangs off it. It is redirected for the same
    // reason every other path variable is: a test that resolves a `~`-bearing
    // default must land in the sandbox, not on the operator's disk.
    //
    // It was `Preserved` until 2026-10-05, because redirecting it broke tests
    // OUTSIDE this guard -- `dirs::home_dir()` is process-global, and those
    // tests read it without taking the lock, so a peer's redirect decided what
    // they saw. That was never a reason to leave `$HOME` unmanaged: it was a
    // reason to bring those tests under the guard, which is what
    // `OUTSIDE_THE_CONTRACT` tracked and what is now empty. Redirecting is the
    // only version in which a test can assert that a `~` default resolved to
    // the sandbox. `Point::Preserved` is gone rather than left unused: a
    // variant with no member is a decision nobody can audit.
    ("HOME", Point::Empty("home")),
    ("XDG_CONFIG_HOME", Point::Empty("xdg/config")),
    ("XDG_CACHE_HOME", Point::Empty("xdg/cache")),
    ("XDG_DATA_HOME", Point::Empty("xdg/data")),
    ("XDG_STATE_HOME", Point::Empty("xdg/state")),
    // eval::signals::signals_path
    ("EVAL_SIGNALS_DIR", Point::Empty("signals")),
    // eval::outputs::outputs_dir -- the variable whose absence wrote
    // `~/.config/ztools/outputs/gone-model/t1.txt` on every suite run.
    ("EVAL_OUTPUT_DIR", Point::Empty("outputs")),
    // eval::budgets::conf_root / model_resolve::disk::conf_models_root. Empty,
    // so the checkout's real `conf/config.toml` cannot decide a timeout.
    ("ZTOOLS_CONF_DIR", Point::Empty("conf")),
    // eval::gpu_lock::lock_dir. `Absent`, because falling back to
    // `/tmp/mac-osaurus-gpu.lock` -- what an unset variable does -- is the real
    // machine-wide lock.
    ("ZTOOLS_GPU_LOCK_DIR", Point::Absent("gpu-lock")),
    ("ZTOOLS_GPU_LOCK_OWNER", Point::Unset),
    // model_resolve::disk::{models_dir, hf_cache_dir}
    ("MLX_MODELS_DIR", Point::Empty("mlx")),
    ("HF_HOME", Point::Empty("hf")),
    // store::{twitter_latest, weekend_output_dir}, twitter_status
    ("TWITTER_OUTPUT_DIR", Point::Empty("twitter-out")),
    ("WEEKEND_OUTPUT_DIR", Point::Empty("weekend-out")),
    // twitter::session::profile_dir
    ("TWITTER_PROFILE_DIR", Point::Empty("twitter-profile")),
    // twitter::browser_bin: no operator override, and an empty sandbox home,
    // so resolution finds nothing to launch.
    ("CAMOUFOX_BIN", Point::Unset),
    // manifest::running_exe: the executable every checkout-derived default
    // starts from. Without it the test binary's BUILD LAYOUT decided the
    // defaults -- under `cargo llvm-cov` it lives inside the checkout, so they
    // named the operator's real `conf/`. A path with no checkout above it.
    ("ZTOOLS_EXE", Point::Absent("bin/ztools")),
];

/// Policy knobs the sandbox clears, so an exported value cannot decide a test.
pub(super) const CLEARED: &[&str] = &[
    "EVAL_SAVE_OUTPUTS",
    "EVAL_MAX_SAVED_OUTPUT",
    "EVAL_DEFAULT_TIMEOUT",
    "EVAL_MAX_TIMEOUT",
    "EVAL_MODEL_STALL_SECONDS",
    "EVAL_ALLOW_OVERSIZE",
    "TWITTER_MAX_RUNTIME_S",
    "TWITTER_FOLLOWING_TAB",
    "TWITTER_FALLBACK_MODELS",
    "ZTOOLS_TLS_PROBE",
    "ZTOOLS_TLS_PROBE_URL",
    "ZTOOLS_UPDATE_GOLDENS",
];

/// Variables read by code OUTSIDE the crate — the house gate's checkout
/// locator, reached only from `rust/tests/gpu_lock_shell_parity.rs`, which
/// spawns `tools/gpu_lock.sh`. The sandbox cannot recreate another repository,
/// and clearing `GOH_DIR` would only break that one test, so these are
/// declared rather than redirected: the gate still fails if a NEW external
/// variable appears.
///
/// Test-only because it exists solely to be checked by the gate in `audit`.
#[cfg(test)]
pub(super) const EXTERNAL: &[&str] = &["GOH_DIR"];

/// Every managed variable, spelled out.
///
/// Kept as a literal rather than derived from [`REDIRECTED`] + [`CLEARED`] so
/// it is greppable and diffable, and `restore_tests` proves the two agree -- a
/// variable added to one table and not the other is a test failure, not a
/// variable quietly left unrestored.
pub(super) const MANAGED: &[&str] = &[
    "CAMOUFOX_BIN",
    "EVAL_ALLOW_OVERSIZE",
    "EVAL_DEFAULT_TIMEOUT",
    "EVAL_MAX_SAVED_OUTPUT",
    "EVAL_MAX_TIMEOUT",
    "EVAL_MODEL_STALL_SECONDS",
    "EVAL_OUTPUT_DIR",
    "EVAL_SAVE_OUTPUTS",
    "EVAL_SIGNALS_DIR",
    "HF_HOME",
    "HOME",
    "MLX_MODELS_DIR",
    "TWITTER_FALLBACK_MODELS",
    "TWITTER_FOLLOWING_TAB",
    "TWITTER_MAX_RUNTIME_S",
    "TWITTER_OUTPUT_DIR",
    "TWITTER_PROFILE_DIR",
    "WEEKEND_OUTPUT_DIR",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
    "ZTOOLS_CONF_DIR",
    "ZTOOLS_EXE",
    "ZTOOLS_GPU_LOCK_DIR",
    "ZTOOLS_GPU_LOCK_OWNER",
    "ZTOOLS_TLS_PROBE",
    "ZTOOLS_TLS_PROBE_URL",
    "ZTOOLS_UPDATE_GOLDENS",
];
