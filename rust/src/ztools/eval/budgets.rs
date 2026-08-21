//! Per-task output budgets, resolved from config like the Python eval does.
//!
//! Ported from `lib/config_getters.py::get_max_tokens_for_task` and its
//! `get_model_config` cap chain. The budget is the `[max_tokens]` task table
//! in `conf/config.toml` (fallback [`DEFAULT_MAX_TOKENS`]), then NARROWED --
//! never widened -- by the model family's `max_tokens`, which is the remedy
//! for a specific failure: a reasoning model given a large budget on a hard
//! prompt thinks past it and returns `finish_reason=length` with nothing to
//! score. A per-model entry that WIDENED the budget would silently override a
//! task's own limit.

use std::path::PathBuf;

/// The documented fallback for any task absent from the `[max_tokens]` table
/// (`lib/llm/constants.py::DEFAULT_MAX_TOKENS`).
pub const DEFAULT_MAX_TOKENS: u32 = 32_000;

/// Where `conf/` lives. `ZTOOLS_CONF_DIR` exists so tests can point the
/// resolver at fixture files without touching the operator's real config --
/// the same seam `model_resolve.rs` uses for `conf/models/`.
pub(crate) fn conf_root() -> PathBuf {
    if let Ok(dir) = std::env::var("ZTOOLS_CONF_DIR") {
        return PathBuf::from(dir);
    }
    if let Some(home) = dirs::home_dir() {
        let p = home.join("Projects/ztools/conf");
        if p.is_dir() {
            return p;
        }
    }
    PathBuf::from("conf")
}

fn parse(path: PathBuf) -> Option<toml::Value> {
    let content = std::fs::read_to_string(path).ok()?;
    toml::from_str(&content).ok()
}

fn family_toml_exists(candidate: &str) -> bool {
    for suffix in ["", "_versions"] {
        if conf_root().join("models").join(format!("{candidate}{suffix}.toml")).is_file() {
            return true;
        }
    }
    false
}

/// The conf/models/<family>.toml that serves an architecture, or None.
///
/// Architectures carry version and variant suffixes ("<fam>3_5_moe",
/// "<fam>4_unified", "<fam>_h") while the config files are named for the bare
/// family, so the two are reconciled by trimming one trailing segment at a
/// time and taking the first name that has a file -- rather than by a
/// hand-written architecture-to-family table, which would need editing every
/// time a vendor ships a new suffix and would silently mis-serve until someone
/// noticed.
fn config_family_for(architecture: &str) -> Option<String> {
    let mut candidate = architecture.to_lowercase();
    while !candidate.is_empty() {
        if family_toml_exists(&candidate) {
            return Some(candidate);
        }
        // Strip one trailing segment: `[_-.]<segment>$`, else trailing digits.
        let trimmed = match candidate.rfind(['_', '.', '-']) {
            Some(idx) if idx > 0 => candidate[..idx].to_string(),
            _ => candidate
                .trim_end_matches(|c: char| c.is_ascii_digit())
                .to_string(),
        };
        if trimmed == candidate {
            return None;
        }
        candidate = trimmed;
    }
    None
}

/// The architecture `ev` probed and wrote to eval_signals.json, or None.
///
/// Read from DISK, never from the server: production paths run constantly and
/// must not do network I/O.
fn recorded_architecture(model: &str) -> Option<String> {
    let signals = crate::ztools::eval::signals::load_signals();
    signals
        .get(model)?
        .get("_capabilities")?
        .get("family")?
        .as_str()
        .map(String::from)
}

/// Which conf/models/<family>.toml drives this model's config.
///
/// Prefers the architecture recorded in eval_signals (the NAME does not
/// reliably encode it: vendors ship models under brand names sharing an
/// architecture with a differently-named family). Falls back to name matching
/// when nothing has been recorded, so this never depends on the eval having
/// been run.
pub(crate) fn config_family(model: &str) -> Option<String> {
    let _ = recorded_architecture;
    if let Some(architecture) = recorded_architecture(model) {
        if let Some(mapped) = config_family_for(&architecture) {
            return Some(mapped);
        }
    }
    let family = crate::ztools::eval::quirks::get_model_family(model);
    if family == "default" {
        None
    } else {
        Some(family.to_string())
    }
}

/// The family config for `model`: `<family>_versions.toml` when it exists,
/// else `<family>.toml`. Mirrors `get_model_config`'s preference -- a versions
/// file REPLACES the family file wholesale there, so top-level keys like
/// `max_tokens` must be read from whichever file actually won.
fn family_config(model: &str) -> Option<toml::Value> {
    let family = config_family(model)?;
    let root = conf_root().join("models");
    let versions = root.join(format!("{family}_versions.toml"));
    if versions.is_file() {
        return parse(versions);
    }
    parse(root.join(format!("{family}.toml")))
}

/// The narrowing cap for one model: the family config's top-level
/// `max_tokens`, overridden by its `[models."<id>"]` section when present.
fn model_cap(model: &str) -> Option<u32> {
    let cfg = family_config(model)?;
    let mut cap = cfg.get("max_tokens").and_then(|v| v.as_integer());
    if let Some(section) = cfg.get("models").and_then(|m| m.get(model)) {
        if let Some(per_model) = section.get("max_tokens").and_then(|v| v.as_integer()) {
            cap = Some(per_model);
        }
    }
    cap.filter(|c| *c > 0).map(|c| c as u32)
}

/// Output budget for one task and model: the `[max_tokens]` table entry,
/// fallback [`DEFAULT_MAX_TOKENS`], narrowed by the model's configured cap.
///
/// An unreadable or missing config degrades to the fallback budget rather than
/// failing: the eval must still run, and 32000 is what the Python eval sends
/// for untabled tasks with no per-model cap.
pub fn max_tokens_for_task(task: &str, model: &str) -> u32 {
    let budget = parse(conf_root().join("config.toml"))
        .and_then(|cfg| {
            cfg.get("max_tokens")
                .and_then(|t| t.get(task))
                .and_then(|v| v.as_integer())
        })
        .filter(|b| *b > 0)
        .map(|b| b as u32)
        .unwrap_or(DEFAULT_MAX_TOKENS);
    match model_cap(model) {
        Some(cap) => budget.min(cap),
        None => budget,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::fs;

    struct ConfDir(tempfile::TempDir);

    impl ConfDir {
        fn new() -> Self {
            Self(tempfile::tempdir().unwrap())
        }
        fn write(&self, rel: &str, content: &str) {
            let path = self.0.path().join(rel);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, content).unwrap();
        }
        fn guard(&self) -> ConfEnvGuard {
            let prev_conf = std::env::var_os("ZTOOLS_CONF_DIR");
            std::env::set_var("ZTOOLS_CONF_DIR", self.0.path());
            // Isolate the signals store too: the recorded-architecture family
            // resolution reads it, and the operator's real file must not
            // decide a test's outcome.
            let prev_signals = std::env::var_os("EVAL_SIGNALS_DIR");
            std::env::set_var("EVAL_SIGNALS_DIR", self.0.path());
            ConfEnvGuard { prev_conf, prev_signals }
        }
    }

    struct ConfEnvGuard {
        prev_conf: Option<std::ffi::OsString>,
        prev_signals: Option<std::ffi::OsString>,
    }

    impl Drop for ConfEnvGuard {
        fn drop(&mut self) {
            match self.prev_conf.take() {
                Some(v) => std::env::set_var("ZTOOLS_CONF_DIR", v),
                None => std::env::remove_var("ZTOOLS_CONF_DIR"),
            }
            match self.prev_signals.take() {
                Some(v) => std::env::set_var("EVAL_SIGNALS_DIR", v),
                None => std::env::remove_var("EVAL_SIGNALS_DIR"),
            }
        }
    }

    #[test]
    #[serial]
    fn untabled_task_and_uncapped_model_get_the_documented_fallback() {
        let dir = ConfDir::new();
        let _g = dir.guard();
        assert_eq!(max_tokens_for_task("taxes_slip_qa", "gemma-4-e2b-it-8bit"), DEFAULT_MAX_TOKENS);
    }

    #[test]
    #[serial]
    fn the_task_table_beats_the_fallback_and_only_narrows() {
        let dir = ConfDir::new();
        dir.write("config.toml", "[max_tokens]\nsummarize = 8000\n");
        let _g = dir.guard();
        assert_eq!(max_tokens_for_task("summarize", "gemma-4-e2b-it-8bit"), 8000);
    }

    #[test]
    #[serial]
    fn a_family_top_level_cap_narrows_the_budget() {
        // foundation.toml carries max_tokens = 3000 at top level: the whole
        // point of the mechanism (its window covers prompt + OUTPUT).
        let dir = ConfDir::new();
        dir.write(
            "models/foundation.toml",
            "name = \"foundation\"\ncontext_window = 4096\nmax_tokens = 3000\n",
        );
        let _g = dir.guard();
        assert_eq!(max_tokens_for_task("think", "foundation"), 3000);
        assert_eq!(max_tokens_for_task("think", "foundation-something-else"), 3000);
    }

    #[test]
    #[serial]
    fn a_per_model_section_narrows_below_the_family() {
        let dir = ConfDir::new();
        dir.write(
            "models/gemma_versions.toml",
            "name = \"gemma\"\nmax_tokens = 16000\n\n[models.\"gemma-4-tiny-test\"]\nmax_tokens = 512\n",
        );
        let _g = dir.guard();
        assert_eq!(max_tokens_for_task("json", "gemma-4-e2b-it-8bit"), 16000);
        assert_eq!(max_tokens_for_task("json", "gemma-4-tiny-test"), 512);
    }

    #[test]
    #[serial]
    fn a_widening_cap_is_never_applied() {
        // Only ever NARROWS: a per-model entry larger than the task's own
        // limit must not silently override it.
        let dir = ConfDir::new();
        dir.write("config.toml", "[max_tokens]\nfilename = 1000\n");
        dir.write(
            "models/qwen.toml",
            "name = \"qwen\"\nmax_tokens = 32000\n",
        );
        let _g = dir.guard();
        assert_eq!(max_tokens_for_task("filename", "qwen3.8-27b-8bit"), 1000);
    }

    #[test]
    #[serial]
    fn a_brand_name_model_resolves_its_family_from_the_recorded_architecture() {
        // The NAME does not reliably encode the family: a "bonsai-*" model
        // carries no family substring, so name matching sends it nowhere --
        // but `ev` recorded its architecture, and trimming qwen3_5_moe ->
        // qwen3_5 -> qwen lands on the file written for it.
        let dir = ConfDir::new();
        dir.write(
            "models/qwen.toml",
            "name = \"qwen\"\nmax_tokens = 8000\n",
        );
        dir.write(
            "eval_signals.json",
            r#"{"bonsai-27b": {"_capabilities": {"family": "qwen3_5_moe"}}}"#,
        );
        let _g = dir.guard();
        // Name matching alone would find no family ("bonsai" matches nothing);
        // the recorded architecture must.
        assert_eq!(
            max_tokens_for_task("json", "bonsai-27b"),
            8000,
            "architecture-based family resolution drives the cap"
        );
    }

    #[test]
    #[serial]
    fn default_family_models_get_the_plain_budget() {
        // A name containing no known family has no conf/models file to consult.
        let dir = ConfDir::new();
        let _g = dir.guard();
        assert_eq!(max_tokens_for_task("json", "totally-unknown-model"), DEFAULT_MAX_TOKENS);
    }
}
