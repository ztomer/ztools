//! Shared test scaffolding for model_resolve's disk- and fetch-domain tests.

/// Isolates every disk/config seam from the operator's real machine
/// (~/MLXModels and the checkout's conf/ both exist here) and restores
/// whatever was set before.
pub(super) struct DiskGuard {
    saved: Vec<(&'static str, Option<std::ffi::OsString>)>,
    pub(super) _dir: tempfile::TempDir,
}

impl DiskGuard {
    pub(super) fn new() -> Self {
        let keys = ["MLX_MODELS_DIR", "HF_HOME", "ZTOOLS_CONF_DIR"];
        let saved = keys.iter().map(|k| (*k, std::env::var_os(k))).collect();
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("mlx")).unwrap();
        std::fs::create_dir_all(dir.path().join("conf")).unwrap();
        std::env::set_var("MLX_MODELS_DIR", dir.path().join("mlx"));
        std::env::set_var("HF_HOME", dir.path().join("hf"));
        std::env::set_var("ZTOOLS_CONF_DIR", dir.path().join("conf"));
        Self { saved, _dir: dir }
    }

    pub(super) fn conf_dir(&self) -> std::path::PathBuf {
        self._dir.path().join("conf")
    }

    pub(super) fn write_family_toml(&self, family: &str, content: &str) {
        let path = self.conf_dir().join("models").join(format!("{family}.toml"));
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }
}

impl Drop for DiskGuard {
    fn drop(&mut self) {
        for (key, prev) in self.saved.drain(..) {
            match prev {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }
}
