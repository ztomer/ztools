//! Filesystem probing, generative-ness, and disk corroboration -- the `disk`
//! submodule's tests.

use std::io::Write as _;
use std::path::PathBuf;

use crate::ztools::eval::model_resolve::disk::{
    conf_models_root, documented_context_window, walk_configs,
};
use crate::ztools::eval::model_resolve::*;

use super::support::DiskGuard;
use serial_test::serial;

/// Is the kernel enforcing unix file modes for THIS process?
///
/// Both permission cases in this file are stated as "the OS refuses" -- a
/// mode-000 directory yields nothing, a mode-000 file keeps probing -- and
/// neither is checked in code. That makes them depend on WHO is running:
///
///   * as root (or with `CAP_DAC_OVERRIDE`) a mode-000 file is readable, so the
///     unreadable-file case PASSES while proving nothing at all: the file is
///     read, `{}` parses, and the verdict is generative for an unrelated reason;
///   * and a mode-000 root is traversable, so the unreadable-root case does not
///     merely go vacuous -- it FAILS, with a message that reads like a bug in
///     the walker rather than like a privilege nobody thought about.
///
/// So the enforcement is PROBED rather than assumed, and a probe that comes back
/// "not enforced" skips the assertion LOUDLY (see [`skip`]) instead of passing
/// quietly or failing confusingly. The coverage is kept either way: the point
/// is to run it wherever it can actually mean something, and to say out loud
/// where it cannot.
#[cfg(unix)]
fn file_modes_enforced() -> bool {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let probe = dir.path().join("mode-000");
    std::fs::write(&probe, b"x").unwrap();
    std::fs::set_permissions(&probe, std::fs::Permissions::from_mode(0o000)).unwrap();
    // The only evidence that matters: can THIS process still read it? Restoring
    // the mode afterwards keeps the temp dir removable on the way out.
    let enforced = std::fs::read(&probe).is_err();
    let _ = std::fs::set_permissions(&probe, std::fs::Permissions::from_mode(0o644));
    enforced
}

/// Nothing below is compiled without `unix`, so there is nothing to skip.
#[cfg(not(unix))]
fn file_modes_enforced() -> bool {
    true
}

/// Announce a skip no reader can mistake for a pass.
///
/// `eprintln!` alone is NOT enough: libtest captures it, so a default
/// `cargo test` prints `ok` for a test that asserted nothing -- the exact
/// failure this module is being fixed for, repeated inside the fix. Writing the
/// stderr HANDLE bypasses that capture, so the line is visible in a plain run.
/// Same idiom as `tests/tls_probe.rs`.
fn skip(reason: &str) {
    let mut err = std::io::stderr();
    let _ = writeln!(err, "SKIP: {reason}");
}

#[test]
#[serial_test::serial]
fn hf_snapshot_layout_three_levels_deep_is_found() {
    // REGRESSION: the walk used to stop at two directory levels, so the
    // real HF layout hub/models--org--model/snapshots/<sha>/config.json
    // was invisible -- corroboration dropped servable HF-cache models.
    // The guard points HF_HOME at `<root>/hf`; the snapshot goes where the
    // real HF cache keeps it, `<hf>/hub/<repo>/snapshots/<sha>/config.json`.
    let guard = DiskGuard::new();
    let snap = guard
        .dir()
        .join("hf/hub/models--org--model/snapshots/abc123");
    std::fs::create_dir_all(&snap).unwrap();
    std::fs::write(snap.join("config.json"), "{}").unwrap();
    // MLX is pointed at the guard's own empty tree, so the hit is the HF
    // layout and nothing else.
    let _ = guard.env().set_path("MLX_MODELS_DIR", "empty-mlx");
    let found = model_config_path("model");
    assert_eq!(
        found,
        Some(snap.join("config.json")),
        "three-level snapshot must be found"
    );
    drop(guard);
}

#[test]
fn walk_configs_finds_all_three_nesting_levels_and_skips_the_rest() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    std::fs::write(root.join("config.json"), "{}").unwrap();
    std::fs::create_dir_all(root.join("a/b")).unwrap();
    std::fs::write(root.join("a/config.json"), "{}").unwrap();
    std::fs::write(root.join("a/b/config.json"), "{}").unwrap();
    std::fs::write(root.join("loose.txt"), "not a config").unwrap();
    std::fs::create_dir_all(root.join("empty-dir")).unwrap();

    let mut found = walk_configs(root).unwrap();
    found.sort();
    assert_eq!(found.len(), 3, "{found:?}");
    assert_eq!(found[0], root.join("a/b/config.json"));
    assert_eq!(found[1], root.join("a/config.json"));
    assert_eq!(found[2], root.join("config.json"));
}

#[test]
#[serial]
fn empty_model_names_have_no_config_path() {
    let guard = DiskGuard::new();
    assert_eq!(model_config_path(""), None);
    drop(guard);
}

#[test]
#[serial]
fn mlx_layout_matches_case_insensitively_on_the_directory_name() {
    let guard = DiskGuard::new();
    let model_dir = guard.dir().join("mlx/TestOrg/LiveModel");
    std::fs::create_dir_all(&model_dir).unwrap();
    std::fs::write(model_dir.join("config.json"), "{}").unwrap();

    // Served ids are lowercased; directories keep their case.
    let found = model_config_path("livemodel")
        .expect("case-insensitive directory match must find the config");
    assert_eq!(found, model_dir.join("config.json"));

    // A name nothing on disk backs up.
    assert_eq!(model_config_path("ghost-model"), None);
    drop(guard);
}

#[test]
#[serial]
fn missing_roots_are_skipped_not_errors() {
    let guard = DiskGuard::new();
    // Both roots NAMED but neither present. `set_path` creates the directory it
    // points at, which is the opposite of what this case needs, so each is
    // removed again immediately -- the variable still names a path, and the
    // path is not there, which is the state under test. `unset` is not the
    // alternative: the documented fallback is `~/MLXModels` and
    // `~/.cache/huggingface`, so unsetting would walk the operator's real tree.
    for (key, sub) in [("MLX_MODELS_DIR", "nope"), ("HF_HOME", "also-nope")] {
        let named = guard.env().set_path(key, sub);
        std::fs::remove_dir(&named).unwrap();
    }
    assert_eq!(
        model_config_path("anything"),
        None,
        "absent roots mean not-found, never a panic"
    );

    // A root that exists but cannot be read is skipped the same way --
    // an unreadable probe is not evidence of absence.
    #[cfg(unix)]
    {
        if file_modes_enforced() {
            use std::os::unix::fs::PermissionsExt;
            let locked_root = guard.env().set_path("MLX_MODELS_DIR", "locked-root");
            let locked = locked_root.join("Org/LiveModel");
            std::fs::create_dir_all(&locked).unwrap();
            std::fs::write(locked.join("config.json"), "{}").unwrap();
            std::fs::set_permissions(&locked_root, std::fs::Permissions::from_mode(0o000)).unwrap();
            let found = model_config_path("livemodel");
            let _ = std::fs::set_permissions(&locked_root, std::fs::Permissions::from_mode(0o755));
            assert_eq!(
                found, None,
                "an unreadable root yields nothing, not a panic"
            );
        } else {
            skip(
                "missing_roots_are_skipped_not_errors: the unreadable-ROOT half. \
                 This process can read a mode-000 file, so the kernel is not \
                 enforcing unix modes here (root, or CAP_DAC_OVERRIDE) and a \
                 mode-000 directory would be walked anyway -- the assertion \
                 above would pass without proving anything.",
            );
        }
    }
    drop(guard);
}

/// The three branches of the conf-root resolver, in order.
///
/// The HOME branch is the only one that needs `$HOME` itself, and `$HOME` is
/// the one variable the sandbox PRESERVES rather than redirects (see
/// `Point::Preserved`): it is process-global, and tests outside this contract
/// read it without taking the guard's lock. So this test is the reason the
/// guard's lock is process-wide -- one test in the binary mutating `$HOME`
/// needs every other `~`-reader serialised against it.
#[test]
#[serial]
fn conf_models_root_prefers_env_then_the_checkout_then_a_relative_path() {
    {
        // `TestEnv::new()` already holds the process-wide lock for this whole
        // test, so the `$HOME` write needs no `with_env_lock` of its own, and
        // `set_managed` gets the operator's `$HOME` back on drop -- which is
        // what this test mutating a process-global variable owes the rest of the
        // binary.
        let env = crate::test_env::TestEnv::new();
        let home = env.root().join("fake-home");
        env.set_managed("HOME", home.as_os_str());
        std::fs::create_dir_all(home.join("Projects/ztools/conf/models")).unwrap();
        // A checkout under the fake home, and an env seam that wins over it.
        let conf = env.set_path("ZTOOLS_CONF_DIR", "fixture-conf");

        assert_eq!(conf_models_root(), conf.join("models"), "the env seam wins");

        // Without the env seam, a checkout under HOME is the next branch -- and
        // it is the branch a real developer hits, because every home checkout
        // layout has one.
        env.unset("ZTOOLS_CONF_DIR");
        assert_eq!(conf_models_root(), home.join("Projects/ztools/conf/models"));

        // An EMPTY home: no checkout there, so the documented relative
        // fallback stands. (A truly absent home is not simulatable: home_dir
        // falls back to the passwd entry when HOME is empty or unset.)
        std::fs::remove_dir_all(home.join("Projects")).unwrap();
        assert_eq!(conf_models_root(), PathBuf::from("conf/models"));
        drop(env);
    }
}

#[test]
#[serial]
fn hf_cache_layout_is_recognised_by_its_models_directory_component() {
    let guard = DiskGuard::new();
    let hub = guard.dir().join("hf/hub");
    let model_dir = hub.join("models--TestOrg--TestModel");
    std::fs::create_dir_all(&model_dir).unwrap();
    std::fs::write(model_dir.join("config.json"), "{}").unwrap();
    // MLX removed, so the hit can only be the HF cache.
    guard.env().unset("MLX_MODELS_DIR");
    let found = model_config_path("testmodel")
        .expect("the models-- component must be matched case-insensitively");
    assert_eq!(found, model_dir.join("config.json"));
    assert_eq!(model_config_path("othermodel"), None);
    drop(guard);
}

#[test]
#[serial]
fn documented_context_window_found_not_found_malformed_or_nonpositive() {
    let guard = DiskGuard::new();

    // Unknown family: no file is ever consulted.
    assert_eq!(documented_context_window("ghost-model"), None);

    guard.write_family_toml("foundation", "context_window = 4096\n");
    assert_eq!(
        documented_context_window("foundation-something"),
        Some(4096)
    );

    // Known family but no file for it.
    assert_eq!(documented_context_window("qwen3.8-27b"), None);

    guard.write_family_toml("gemma", "{{{ not toml");
    assert_eq!(
        documented_context_window("gemma-4-e2b"),
        None,
        "malformed toml"
    );

    for content in ["context_window = 0\n", "context_window = -5\n"] {
        guard.write_family_toml("nemotron", content);
        assert_eq!(documented_context_window("nemotron-x"), None, "{content}");
    }

    guard.write_family_toml("laguna", "context_window = \"4096\"\n");
    assert_eq!(
        documented_context_window("laguna-x"),
        None,
        "string is no window"
    );
    drop(guard);
}

#[test]
#[serial]
fn generative_verdict_comes_from_the_config_not_the_name() {
    let guard = DiskGuard::new();
    let put = |name: &str, content: &str| {
        let d = guard.dir().join("mlx/Org").join(name);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("config.json"), content).unwrap();
    };

    // Nothing on disk: assume generative rather than silently skipping a
    // model the user installed.
    assert!(is_generative_model("ghost-model"));

    put("Embedder", r#"{"model_type": "Model2Vec"}"#);
    assert!(
        !is_generative_model("embedder"),
        "type check is case-insensitive"
    );

    put("StaticArch", r#"{"architectures": ["StaticModel"]}"#);
    assert!(!is_generative_model("staticarch"));

    put("SentTrans", r#"{"architectures": ["SentenceTransformer"]}"#);
    assert!(!is_generative_model("senttrans"));

    put(
        "RealModel",
        r#"{"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}"#,
    );
    assert!(is_generative_model("realmodel"));

    put("NoArch", r#"{"model_type": "whatever"}"#);
    assert!(is_generative_model("noarch"));

    put("BrokenJson", "{not json");
    assert!(
        is_generative_model("brokenjson"),
        "unreadable-as-json keeps probing"
    );

    // An unreadable FILE also keeps probing: same verdict as missing.
    if file_modes_enforced() {
        let locked = guard.dir().join("mlx/Org/Locked/config.json");
        std::fs::create_dir_all(locked.parent().unwrap()).unwrap();
        std::fs::write(&locked, "{}").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
            assert!(is_generative_model("locked"));
            let _ = std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o644));
        }
        #[cfg(not(unix))]
        assert!(is_generative_model("locked"));
    } else {
        skip(
            "generative_verdict_comes_from_the_config_not_the_name: the \
             unreadable-FILE half. This process can read a mode-000 file, so \
             the kernel is not enforcing unix modes here (root, or \
             CAP_DAC_OVERRIDE) and the file would be parsed as ordinary `{}` -- \
             generative for a reason that has nothing to do with the unreadable \
             path the assertion is about.",
        );
    }
    drop(guard);
}

#[test]
#[serial]
fn corroboration_accepts_disk_configs_or_documented_windows_only() {
    let guard = DiskGuard::new();
    assert!(
        !disk_corroborated("ghost-model"),
        "nothing on disk backs it"
    );

    guard.write_family_toml("foundation", "context_window = 4096\n");
    assert!(
        disk_corroborated("foundation-x"),
        "a documented window corroborates without any disk config"
    );

    let model_dir = guard.dir().join("mlx/Org/DiskModel");
    std::fs::create_dir_all(&model_dir).unwrap();
    std::fs::write(model_dir.join("config.json"), "{}").unwrap();
    assert!(disk_corroborated("diskmodel"));
    drop(guard);
}
