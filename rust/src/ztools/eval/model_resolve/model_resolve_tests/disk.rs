//! Filesystem probing, generative-ness, and disk corroboration -- the `disk`
//! submodule's tests.

use std::path::PathBuf;

use crate::ztools::eval::model_resolve::disk::{
    conf_models_root, documented_context_window, walk_configs,
};
use crate::ztools::eval::model_resolve::*;

use super::support::DiskGuard;
use serial_test::serial;

#[test]
#[serial_test::serial]
fn hf_snapshot_layout_three_levels_deep_is_found() {
    // REGRESSION: the walk used to stop at two directory levels, so the
    // real HF layout hub/models--org--model/snapshots/<sha>/config.json
    // was invisible -- corroboration dropped servable HF-cache models.
    let dir = tempfile::tempdir().unwrap();
    let snap = dir.path().join("hub/models--org--model/snapshots/abc123");
    std::fs::create_dir_all(&snap).unwrap();
    std::fs::write(snap.join("config.json"), "{}").unwrap();
    let prev = std::env::var_os("MLX_MODELS_DIR");
    let prev_hf = std::env::var_os("HF_HOME");
    std::env::set_var("MLX_MODELS_DIR", dir.path().join("empty-mlx"));
    std::env::set_var("HF_HOME", dir.path());
    let found = model_config_path("model");
    match prev {
        Some(v) => std::env::set_var("MLX_MODELS_DIR", v),
        None => std::env::remove_var("MLX_MODELS_DIR"),
    }
    match prev_hf {
        Some(v) => std::env::set_var("HF_HOME", v),
        None => std::env::remove_var("HF_HOME"),
    }
    assert!(found.is_some(), "three-level snapshot must be found");
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
    let _guard = DiskGuard::new();
    assert_eq!(model_config_path(""), None);
}

#[test]
#[serial]
fn mlx_layout_matches_case_insensitively_on_the_directory_name() {
    let guard = DiskGuard::new();
    let model_dir = guard._dir.path().join("mlx/TestOrg/LiveModel");
    std::fs::create_dir_all(&model_dir).unwrap();
    std::fs::write(model_dir.join("config.json"), "{}").unwrap();

    // Served ids are lowercased; directories keep their case.
    let found = model_config_path("livemodel")
        .expect("case-insensitive directory match must find the config");
    assert_eq!(found, model_dir.join("config.json"));

    // A name nothing on disk backs up.
    assert_eq!(model_config_path("ghost-model"), None);
}

#[test]
#[serial]
fn missing_roots_are_skipped_not_errors() {
    let dir = tempfile::tempdir().unwrap();
    let keys = ["MLX_MODELS_DIR", "HF_HOME"];
    // MLX pre-set (sentinel -> was-set restore arm), HF removed before
    // capture (never-set restore arm): both arms covered deterministically.
    std::env::set_var("MLX_MODELS_DIR", "/nonexistent-sentinel");
    std::env::remove_var("HF_HOME");
    let saved: Vec<(&'static str, Option<std::ffi::OsString>)> =
        keys.iter().map(|k| (*k, std::env::var_os(k))).collect();
    struct Restore(Vec<(&'static str, Option<std::ffi::OsString>)>);
    impl Drop for Restore {
        fn drop(&mut self) {
            for (k, v) in self.0.drain(..) {
                match v {
                    Some(v) => std::env::set_var(k, v),
                    None => std::env::remove_var(k),
                }
            }
        }
    }
    let _restore = Restore(saved);
    std::env::set_var("MLX_MODELS_DIR", dir.path().join("nope"));
    std::env::set_var("HF_HOME", dir.path().join("also-nope"));
    assert_eq!(
        model_config_path("anything"),
        None,
        "absent roots mean not-found, never a panic"
    );

    // A root that exists but cannot be read is skipped the same way --
    // an unreadable probe is not evidence of absence.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let locked = dir.path().join("locked-root/Org/LiveModel");
        std::fs::create_dir_all(&locked).unwrap();
        std::fs::write(locked.join("config.json"), "{}").unwrap();
        std::fs::set_permissions(
            dir.path().join("locked-root"),
            std::fs::Permissions::from_mode(0o000),
        )
        .unwrap();
        std::env::set_var("MLX_MODELS_DIR", dir.path().join("locked-root"));
        let found = model_config_path("livemodel");
        let _ = std::fs::set_permissions(
            dir.path().join("locked-root"),
            std::fs::Permissions::from_mode(0o755),
        );
        assert_eq!(found, None, "an unreadable root yields nothing, not a panic");
    }
}

#[test]
#[serial]
fn conf_models_root_prefers_env_then_the_checkout_then_a_relative_path() {
    let keys = ["ZTOOLS_CONF_DIR", "HOME"];
    // ZTOOLS pre-set (was-set restore arm); HOME removed first
    // (never-set restore arm).
    std::env::set_var("ZTOOLS_CONF_DIR", "/nonexistent-sentinel");
    std::env::remove_var("HOME");
    let saved: Vec<(&'static str, Option<std::ffi::OsString>)> =
        keys.iter().map(|k| (*k, std::env::var_os(k))).collect();
    struct Restore(Vec<(&'static str, Option<std::ffi::OsString>)>);
    impl Drop for Restore {
        fn drop(&mut self) {
            for (k, v) in self.0.drain(..) {
                match v {
                    Some(v) => std::env::set_var(k, v),
                    None => std::env::remove_var(k),
                }
            }
        }
    }
    let _restore = Restore(saved);

    std::env::set_var("ZTOOLS_CONF_DIR", "/fixture-conf");
    assert_eq!(
        conf_models_root(),
        PathBuf::from("/fixture-conf").join("models"),
        "the env seam wins"
    );

    std::env::remove_var("ZTOOLS_CONF_DIR");
    // A checkout under HOME with a conf/models tree: the home branch.
    let fake_home = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(
        fake_home.path().join("Projects/ztools/conf/models"),
    )
    .unwrap();
    std::env::set_var("HOME", fake_home.path());
    assert_eq!(
        conf_models_root(),
        fake_home.path().join("Projects/ztools/conf/models")
    );

    // An EMPTY HOME: no checkout there, so the documented relative
    // fallback stands. (A truly absent home is not simulatable: home_dir
    // falls back to the passwd entry when HOME is empty or unset.)
    let empty_home = tempfile::tempdir().unwrap();
    std::env::set_var("HOME", empty_home.path());
    assert_eq!(conf_models_root(), PathBuf::from("conf/models"));
}

#[test]
#[serial]
fn hf_cache_layout_is_recognised_by_its_models_directory_component() {
    let guard = DiskGuard::new();
    let hub = guard._dir.path().join("hf/hub");
    let model_dir = hub.join("models--TestOrg--TestModel");
    std::fs::create_dir_all(&model_dir).unwrap();
    std::fs::write(model_dir.join("config.json"), "{}").unwrap();
    std::env::remove_var("MLX_MODELS_DIR");

    let found = model_config_path("testmodel")
        .expect("the models-- component must be matched case-insensitively");
    assert_eq!(found, model_dir.join("config.json"));
    assert_eq!(model_config_path("othermodel"), None);
}

#[test]
#[serial]
fn documented_context_window_found_not_found_malformed_or_nonpositive() {
    let guard = DiskGuard::new();

    // Unknown family: no file is ever consulted.
    assert_eq!(documented_context_window("ghost-model"), None);

    guard.write_family_toml("foundation", "context_window = 4096\n");
    assert_eq!(documented_context_window("foundation-something"), Some(4096));

    // Known family but no file for it.
    assert_eq!(documented_context_window("qwen3.8-27b"), None);

    guard.write_family_toml("gemma", "{{{ not toml");
    assert_eq!(documented_context_window("gemma-4-e2b"), None, "malformed toml");

    for content in ["context_window = 0\n", "context_window = -5\n"] {
        guard.write_family_toml("nemotron", content);
        assert_eq!(documented_context_window("nemotron-x"), None, "{content}");
    }

    guard.write_family_toml("laguna", "context_window = \"4096\"\n");
    assert_eq!(documented_context_window("laguna-x"), None, "string is no window");
}

#[test]
#[serial]
fn generative_verdict_comes_from_the_config_not_the_name() {
    let guard = DiskGuard::new();
    let put = |name: &str, content: &str| {
        let d = guard._dir.path().join("mlx/Org").join(name);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("config.json"), content).unwrap();
    };

    // Nothing on disk: assume generative rather than silently skipping a
    // model the user installed.
    assert!(is_generative_model("ghost-model"));

    put("Embedder", r#"{"model_type": "Model2Vec"}"#);
    assert!(!is_generative_model("embedder"), "type check is case-insensitive");

    put("StaticArch", r#"{"architectures": ["StaticModel"]}"#);
    assert!(!is_generative_model("staticarch"));

    put("SentTrans", r#"{"architectures": ["SentenceTransformer"]}"#);
    assert!(!is_generative_model("senttrans"));

    put("RealModel", r#"{"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}"#);
    assert!(is_generative_model("realmodel"));

    put("NoArch", r#"{"model_type": "whatever"}"#);
    assert!(is_generative_model("noarch"));

    put("BrokenJson", "{not json");
    assert!(is_generative_model("brokenjson"), "unreadable-as-json keeps probing");

    // An unreadable FILE also keeps probing: same verdict as missing.
    let locked = guard._dir.path().join("mlx/Org/Locked/config.json");
    std::fs::create_dir_all(locked.parent().unwrap()).unwrap();
    std::fs::write(&locked, "{}").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
    }
    assert!(is_generative_model("locked"));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o644));
    }
}

#[test]
#[serial]
fn corroboration_accepts_disk_configs_or_documented_windows_only() {
    let guard = DiskGuard::new();
    assert!(!disk_corroborated("ghost-model"), "nothing on disk backs it");

    guard.write_family_toml("foundation", "context_window = 4096\n");
    assert!(
        disk_corroborated("foundation-x"),
        "a documented window corroborates without any disk config"
    );

    let model_dir = guard._dir.path().join("mlx/Org/DiskModel");
    std::fs::create_dir_all(&model_dir).unwrap();
    std::fs::write(model_dir.join("config.json"), "{}").unwrap();
    assert!(disk_corroborated("diskmodel"));
}
