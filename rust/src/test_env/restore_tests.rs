//! What the sandbox promises about the variables it takes over.
//!
//! These are the tests that make "panic-safe" and "restores the was-set AND
//! the never-set case" claims into facts rather than intentions. They also
//! pin the managed list against the tables that drive it.
//!
//! The three tests that touch a variable's AMBIENT value run inside
//! [`with_env_lock`] and build their sandbox with [`TestEnv::redirected`] from the
//! same acquisition: that write is itself an environment mutation, and the lock
//! is not reentrant, so ambient and redirect have to share one hold. Those
//! writes go through the module's [`put`]/[`clear`] seam rather than
//! `std::env` directly, which is the only reason the seam can claim to be the
//! crate's only environment writer.

use std::cell::Cell;
use std::rc::Rc;

use serial_test::serial;

use super::*;

/// Restore is the whole contract: whatever the ambient state was, the process
/// sees it again afterwards.
#[test]
#[serial]
fn every_managed_variable_is_sandboxed_and_restored() {
    with_env_lock(|_lock| {
        // Ambient state chosen so both restore arms are exercised by ONE run: the
        // odd-numbered keys are pre-set to a sentinel, the even-numbered ones are
        // removed first so the never-set arm is the real thing.
        let was_set = ["EVAL_SIGNALS_DIR", "MLX_MODELS_DIR", "TWITTER_OUTPUT_DIR"];
        let was_absent = ["EVAL_OUTPUT_DIR", "HF_HOME", "ZTOOLS_CONF_DIR"];
        let home_before = std::env::var_os("HOME");
        let ambient: Vec<(String, Option<std::ffi::OsString>)> = was_set
            .iter()
            .chain(was_absent.iter())
            .map(|k| ((*k).to_string(), std::env::var_os(k)))
            .collect();
        for key in was_set {
            put(key, OsStr::new(&format!("/sentinel-{key}")));
        }
        for key in was_absent {
            clear(key);
        }

        let inside = TestEnv::redirected();
        for key in was_set.iter().chain(was_absent.iter()) {
            let value = std::env::var_os(key).expect("a redirected variable is set");
            assert!(
                !value.to_string_lossy().starts_with("/sentinel-"),
                "{key} was not redirected into the sandbox"
            );
        }
        // Spelled out rather than `inside.path("HOME")`: the subdirectory name
        // IS part of the policy, so reading it back through the accessor that
        // resolves it would make this assertion agree with the table by
        // construction -- and the regression it exists to catch is the table
        // disagreeing with the table.
        let inside_home = inside.root().join("home");
        // `$HOME` is redirected like every other path variable, and the
        // redirect is asserted from BREADTH the guard manages: the variable,
        // the library that reads it, and the emptiness of the directory it now
        // names. A sandbox that restored without redirecting — the policy this
        // crate spent a day moving off — passes every other assertion here and
        // fails this one, which is the whole reason it is written twice.
        assert_eq!(
            std::env::var_os("HOME").as_deref(),
            Some(inside_home.as_os_str()),
            "HOME must point into the sandbox for the whole sandbox lifetime"
        );
        assert_eq!(
            dirs::home_dir().as_deref(),
            Some(inside_home.as_path()),
            "production reads the home through `dirs::home_dir()`, so redirecting \
             the variable is only the point if THAT follows it: a sandbox whose \
             `$HOME` points into the temp dir while `dirs` still answers with the \
             operator's home would leave every `~`-bearing default untouched"
        );
        let in_home = std::fs::read_dir(&inside_home)
            .expect("the sandbox home exists")
            .collect::<Vec<_>>();
        assert_empty!(
            in_home.as_slice(),
            "the sandbox home must be EMPTY: a `~`-bearing default that finds a \
             file here is reading the operator's data with a sandbox path on it"
        );
        drop(inside);

        for key in was_set {
            assert_eq!(
                std::env::var_os(key).as_deref(),
                Some(std::ffi::OsStr::new(&format!("/sentinel-{key}"))),
                "{key} must come back exactly as the guard found it"
            );
        }
        for key in was_absent {
            assert!(
                std::env::var_os(key).is_none(),
                "{key} must come back absent, not as a stale sandbox value"
            );
        }
        // The restore half, asserted after the sandbox is gone. Together with
        // the redirect asserted above this is both halves of the `$HOME`
        // contract: redirected for the sandbox's lifetime, exact afterwards.
        assert_eq!(
            std::env::var_os("HOME"),
            home_before,
            "HOME must come back exactly as the guard found it, or the sandbox \
             would leave the NEXT test in this binary reading a deleted temp dir \
             as its home"
        );
        // Whatever the shell handed this process is handed back, so a sandbox that
        // clobbered `$HOME` for a LATER test in the binary would be caught here.
        for (key, value) in &ambient {
            match value {
                Some(v) => put(key, v),
                None => clear(key),
            }
        }
        for (key, value) in ambient {
            assert_eq!(
                std::env::var_os(&key),
                value,
                "{key} must be exactly as the process found it"
            );
        }
    });
}

/// The GPU lock is the one variable where "redirect" and "create" are opposite
/// answers: `acquire` succeeds only when the path is free, so a sandbox that
/// CREATED its lock directory would be advertising a held lock to every other
/// test in the binary.
#[test]
#[serial]
fn the_gpu_lock_directory_is_redirected_but_not_created() {
    let env = TestEnv::new();
    let lock_dir = env.path("ZTOOLS_GPU_LOCK_DIR");
    assert!(!lock_dir.exists(), "an existing lock dir IS a held lock");
    assert_eq!(
        std::env::var("ZTOOLS_GPU_LOCK_DIR").as_deref(),
        Ok(lock_dir.to_str().expect("a temp path is utf-8"))
    );
    // The real machine-wide lock is not merely unused, it is unreachable.
    assert_ne!(
        lock_dir,
        std::path::PathBuf::from(crate::ztools::eval::gpu_lock::DEFAULT_LOCK_DIR)
    );
    assert!(
        std::env::var_os(crate::ztools::eval::gpu_lock::OWNER_ENV).is_none(),
        "an inherited owner would make foreign_holder() lie"
    );
    drop(env);
}

/// Unwinding out of the sandbox must not leave the process reading a deleted
/// temp dir -- `Drop` is what makes that true, and a `Drop`-less guard would
/// fail here.
#[test]
#[serial]
fn the_environment_is_restored_even_when_the_test_unwinds_through_a_panic() {
    with_env_lock(|_lock| {
        put("EVAL_SIGNALS_DIR", OsStr::new("/sentinel-signals"));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let env = TestEnv::redirected();
            assert_ne!(
                std::env::var("EVAL_SIGNALS_DIR").as_deref(),
                Ok("/sentinel-signals")
            );
            assert!(env.root().is_dir());
            drop(env);
            panic!("unwind through the sandbox");
        }));
        assert!(result.is_err(), "the closure must have panicked");
        assert_eq!(
            std::env::var_os("EVAL_SIGNALS_DIR").as_deref(),
            Some(std::ffi::OsStr::new("/sentinel-signals")),
            "a panic inside the sandbox must not leak the sandboxed value"
        );
        clear("EVAL_SIGNALS_DIR");
    });
}

/// `unset` is the only route to a documented default, so it has to reach the
/// managed set -- including the path variables, not only the policy knobs.
#[test]
#[serial]
fn unset_reaches_a_managed_path_variable_but_a_foreign_key_is_a_hard_failure() {
    let env = TestEnv::new();
    env.set("EVAL_MAX_TIMEOUT", "1234");
    assert_eq!(std::env::var("EVAL_MAX_TIMEOUT").as_deref(), Ok("1234"));
    env.unset("EVAL_MAX_TIMEOUT");
    assert!(std::env::var_os("EVAL_MAX_TIMEOUT").is_none());

    let redirected = std::env::var_os("EVAL_SIGNALS_DIR").expect("redirected");
    env.unset("EVAL_SIGNALS_DIR");
    assert!(std::env::var_os("EVAL_SIGNALS_DIR").is_none());
    put("EVAL_SIGNALS_DIR", &redirected);
    assert_eq!(
        std::env::var_os("EVAL_SIGNALS_DIR").as_deref(),
        Some(redirected.as_os_str())
    );

    // A key the guard does not manage must fail loudly rather than be set with
    // nothing to restore it: that is the leak, named.
    let foreign = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        env.set("NOT_A_ZTOOLS_VARIABLE", "x");
    }));
    assert!(foreign.is_err(), "an unmanaged key must not be settable");
    assert!(
        std::env::var_os("NOT_A_ZTOOLS_VARIABLE").is_none(),
        "the rejected set must not have happened"
    );
    drop(env);
}

/// The literal list, the redirect table and the cleared table are three
/// statements of one fact; they must agree.
#[test]
fn the_managed_list_is_exactly_the_two_tables() {
    let derived: Vec<&str> = REDIRECTED
        .iter()
        .map(|(key, _)| *key)
        .chain(CLEARED.iter().copied())
        .collect();
    let mut derived_sorted = derived.clone();
    derived_sorted.sort_unstable();
    let mut literal_sorted = MANAGED.to_vec();
    literal_sorted.sort_unstable();
    assert_eq!(
        literal_sorted, derived_sorted,
        "MANAGED must list exactly the variables the tables manage"
    );
    assert_eq!(
        derived_sorted.len(),
        MANAGED.len(),
        "a variable appears in the tables more than once"
    );
}

/// `set_path` re-points a managed variable at a fixture directory without
/// giving up the restore -- the property that lets one guard carry a whole
/// per-test filesystem.
#[test]
#[serial]
fn set_path_redirects_inside_the_sandbox_and_still_restores() {
    with_env_lock(|_lock| {
        put("ZTOOLS_CONF_DIR", OsStr::new("/sentinel-conf"));
        let env = TestEnv::redirected();
        let fixture = env.set_path("ZTOOLS_CONF_DIR", "fixture-conf");
        assert!(fixture.is_dir());
        assert_eq!(
            std::env::var_os("ZTOOLS_CONF_DIR").as_deref(),
            Some(fixture.as_os_str())
        );
        drop(env);
        assert_eq!(
            std::env::var_os("ZTOOLS_CONF_DIR").as_deref(),
            Some(std::ffi::OsStr::new("/sentinel-conf"))
        );
        clear("ZTOOLS_CONF_DIR");
    });
}

/// Records, at drop time, whether the variable it names is still SANDBOXED --
/// that is, whether the guard that redirected it is still installed.
struct SandboxStillInstalled(&'static str, Rc<Cell<bool>>);

impl Drop for SandboxStillInstalled {
    fn drop(&mut self) {
        self.1
            .set(std::env::var(self.0).is_ok_and(|v| !v.starts_with("/sentinel-")));
    }
}

/// Edition 2024 drops a tail expression's temporaries BEFORE the block's local
/// variables; 2021 dropped them after. For a crate whose entire isolation
/// strategy is a `Drop`, that difference is load-bearing, and nothing else in
/// the suite can see it -- every sandbox here is `let`-bound, so no ordinary
/// test has a tail temporary to get wrong. This is the test that notices.
///
/// The construction is deliberate in one specific way: the probe is the block's
/// tail expression and its value is CONSUMED by `.len()`, which is what makes it
/// a temporary in tail position rather than a value moved out of the block.
/// Written as `let probe = ...; probe.len()` both editions agree and this test
/// proves nothing at all.
///
/// Calibration: this test passes under `edition = "2024"` and FAILS under
/// `edition = "2021"`, because there the sandbox restores first and the probe
/// observes the sentinel.
#[test]
#[serial]
fn a_tail_expression_temporary_still_sees_the_sandbox_installed() {
    let still_sandboxed = Rc::new(Cell::new(false));
    with_env_lock(|_lock| {
        put("EVAL_SIGNALS_DIR", OsStr::new("/sentinel-tail-order"));
        // The `observed` binding is not decoration: it is what keeps the tail
        // expression a CONSUMING one. Written `let _ = ...;` the probe would be
        // a statement temporary, which both editions drop in the same order, and
        // the assertion below would stop testing the tail-expression rule at all.
        let observed = {
            let _env = TestEnv::redirected();
            assert_ne!(
                std::env::var("EVAL_SIGNALS_DIR").as_deref(),
                Ok("/sentinel-tail-order"),
                "the sandbox must be installed before the tail expression runs, or \
                 this test is not testing what it claims"
            );
            SandboxStillInstalled("EVAL_SIGNALS_DIR", Rc::clone(&still_sandboxed))
                .0
                .len()
        };
        assert_eq!(
            observed,
            "EVAL_SIGNALS_DIR".len(),
            "the tail expression must have produced the probe's name"
        );
        clear("EVAL_SIGNALS_DIR");
    });
    assert!(
        still_sandboxed.get(),
        "a temporary in a tail expression was dropped AFTER the sandbox restored, so \
         it observed the ambient value; under edition 2024 it must be dropped first, \
         while every variable it reads is still redirected"
    );
}
