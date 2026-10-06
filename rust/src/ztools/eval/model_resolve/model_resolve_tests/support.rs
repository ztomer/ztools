//! Shared test scaffolding for `model_resolve`'s disk- and fetch-domain tests.
//!
//! `DiskGuard` used to capture and restore `MLX_MODELS_DIR`, `HF_HOME` and
//! `ZTOOLS_CONF_DIR` under a lock nothing else held, so it raced every test
//! using the crate's one `TestEnv` and the losing side silently probed the
//! other's models directory. It is now a THIN VIEW over that guard: the list,
//! the lock and the restore live in `crate::test_env`, and this adds only the
//! fixture-file helper the disk tests build.
//!
//! `~/MLXModels` and `~/Projects/ztools/conf` BOTH EXIST on a real machine, so
//! a test that skips the redirect measures whatever the developer has installed
//! rather than what it set up -- which is what made this pair of guards worth
//! consolidating rather than keeping as a second implementation.

use std::path::{Path, PathBuf};

use crate::test_env::TestEnv;

/// The one guard, plus the fixture-family files only these tests write.
pub(super) struct DiskGuard {
    env: TestEnv,
}

impl DiskGuard {
    pub(super) fn new() -> Self {
        Self {
            env: TestEnv::new(),
        }
    }

    /// The sandbox root. `TestEnv` already points `MLX_MODELS_DIR` at
    /// `<root>/mlx`, `HF_HOME` at `<root>/hf` and `ZTOOLS_CONF_DIR` at
    /// `<root>/conf`, all empty and all created, so a fixture tree written
    /// under `dir()` reads back through the same seams a real install uses.
    pub(super) fn dir(&self) -> &Path {
        self.env.root()
    }

    pub(super) fn conf_dir(&self) -> PathBuf {
        self.dir().join("conf")
    }

    pub(super) fn write_family_toml(&self, family: &str, content: &str) {
        let path = self
            .conf_dir()
            .join("models")
            .join(format!("{family}.toml"));
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }

    /// The underlying guard, for the handful of tests that must reach past the
    /// fixture files and remove one of the redirected variables.
    pub(super) fn env(&self) -> &TestEnv {
        &self.env
    }
}
