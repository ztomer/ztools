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
        if conf_root()
            .join("models")
            .join(format!("{candidate}{suffix}.toml"))
            .is_file()
        {
            return true;
        }
    }
    false
}

/// The conf/models/<family>.toml that serves an architecture, or None.
///
/// Architectures carry version and variant suffixes ("<fam>`3_5_moe`",
/// "<fam>`4_unified`", "<fam>_h") while the config files are named for the bare
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

/// The architecture `ev` probed and wrote to `eval_signals.json`, or None.
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
/// Prefers the architecture recorded in `eval_signals` (the NAME does not
/// reliably encode it: vendors ship models under brand names sharing an
/// architecture with a differently-named family). Falls back to name matching
/// when nothing has been recorded, so this never depends on the eval having
/// been run.
#[must_use]
pub fn config_family(model: &str) -> Option<String> {
    let _ = recorded_architecture;
    if let Some(architecture) = recorded_architecture(model)
        && let Some(mapped) = config_family_for(&architecture)
    {
        return Some(mapped);
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
    // The per-model section WINS over the family cap, so it is the `or` operand's
    // left side; `None` from either is the same answer, and the positivity filter
    // below is what rejects a zero or negative cap whichever side it came from.
    let per_model = cfg
        .get("models")
        .and_then(|m| m.get(model))
        .and_then(|section| section.get("max_tokens"))
        .and_then(toml::Value::as_integer);
    per_model
        .or_else(|| cfg.get("max_tokens").and_then(toml::Value::as_integer))
        .filter(|c| *c > 0)
        .map(|c| u32::try_from(c).unwrap_or(u32::MAX))
}

/// Output budget for one task and model: the `[max_tokens]` table entry,
/// fallback [`DEFAULT_MAX_TOKENS`], narrowed by the model's configured cap.
///
/// An unreadable or missing config degrades to the fallback budget rather than
/// failing: the eval must still run, and 32000 is what the Python eval sends
/// for untabled tasks with no per-model cap.
#[must_use]
pub fn max_tokens_for_task(task: &str, model: &str) -> u32 {
    let budget = parse(conf_root().join("config.toml"))
        .and_then(|cfg| {
            cfg.get("max_tokens")
                .and_then(|t| t.get(task))
                .and_then(toml::Value::as_integer)
        })
        .filter(|b| *b > 0)
        .map_or(DEFAULT_MAX_TOKENS, |b| u32::try_from(b).unwrap_or(u32::MAX));
    model_cap(model).map_or(budget, |cap| budget.min(cap))
}

/// The `best_for` tags for one model: its `[models."<id>"].best_for` section when
/// present, else the family config's top-level `best_for`.
#[must_use]
pub fn model_best_for(model: &str) -> Vec<String> {
    let Some(cfg) = family_config(model) else {
        return Vec::new();
    };
    let per_model = cfg
        .get("models")
        .and_then(|m| m.get(model))
        .and_then(|section| section.get("best_for"))
        .and_then(toml::Value::as_array);

    let arr = per_model.or_else(|| cfg.get("best_for").and_then(toml::Value::as_array));
    arr.map_or_else(Vec::new, |items| {
        items
            .iter()
            .filter_map(toml::Value::as_str)
            .map(str::to_string)
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_env::TestEnv;
    use serial_test::serial;
    use std::fs;

    /// A conf root and signals store inside one sandbox, holding `files`.
    ///
    /// This replaces a hand-rolled `ConfEnvGuard` that captured and restored
    /// `ZTOOLS_CONF_DIR` and `EVAL_SIGNALS_DIR` itself, under `#[serial]` and
    /// no shared lock -- so it excluded itself from the other `#[serial]` tests
    /// and from nothing else, and a panic between the set and the restore left
    /// both variables pointing at a deleted temp dir. `TestEnv` is the same
    /// guarantee with the lock included, and it holds for the whole test rather
    /// than for the window between the two writes.
    ///
    /// The fixtures are written HERE rather than through a `write` method called
    /// afterwards, and that is not a style choice. A guard whose last use is a
    /// `write` is one clippy's `significant_drop_tightening` asks to fold into
    /// that call -- which would drop the sandbox, releasing the lock and
    /// restoring the operator's paths, before the test had asserted anything.
    /// Passing `&[]` is the empty conf root: no `config.toml`, no `models/`, so
    /// a model with no entry finds nothing HERE rather than in the operator's.
    fn conf_sandbox(files: &[(&str, &str)]) -> TestEnv {
        // Both variables at ONE fixture path, as before: recorded-architecture
        // family resolution reads the signals store out of the same directory the
        // conf files live in, and a test that writes both expects one root.
        let (env, root) = TestEnv::new().fixture("ZTOOLS_CONF_DIR", "conf-fixture");
        let _signals = env.set_path("EVAL_SIGNALS_DIR", "conf-fixture");
        for (rel, content) in files {
            let path = root.join(rel);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, content).unwrap();
        }
        env
    }

    #[test]
    #[serial]
    fn untabled_task_and_uncapped_model_get_the_documented_fallback() {
        // The empty conf root IS the fixture: a model with no entry must find
        // nothing, and the sandbox is what guarantees it finds nothing HERE.
        let _env = conf_sandbox(&[]);
        assert_eq!(
            max_tokens_for_task("taxes_slip_qa", "gemma-4-e2b-it-8bit"),
            DEFAULT_MAX_TOKENS
        );
    }

    #[test]
    #[serial]
    fn the_task_table_beats_the_fallback_and_only_narrows() {
        let _dir = conf_sandbox(&[("config.toml", "[max_tokens]\nsummarize = 8000\n")]);
        assert_eq!(
            max_tokens_for_task("summarize", "gemma-4-e2b-it-8bit"),
            8000
        );
    }

    #[test]
    #[serial]
    fn a_family_top_level_cap_narrows_the_budget() {
        // foundation.toml carries max_tokens = 3000 at top level: the whole
        // point of the mechanism (its window covers prompt + OUTPUT).
        let _dir = conf_sandbox(&[(
            "models/foundation.toml",
            "name = \"foundation\"\ncontext_window = 4096\nmax_tokens = 3000\n",
        )]);
        assert_eq!(max_tokens_for_task("think", "foundation"), 3000);
        assert_eq!(
            max_tokens_for_task("think", "foundation-something-else"),
            3000
        );
    }

    #[test]
    #[serial]
    fn a_per_model_section_narrows_below_the_family() {
        let _dir = conf_sandbox(&[(
            "models/gemma_versions.toml",
            "name = \"gemma\"\nmax_tokens = 16000\n\n[models.\"gemma-4-tiny-test\"]\nmax_tokens = 512\n",
        )]);
        assert_eq!(max_tokens_for_task("json", "gemma-4-e2b-it-8bit"), 16000);
        assert_eq!(max_tokens_for_task("json", "gemma-4-tiny-test"), 512);
    }

    #[test]
    #[serial]
    fn a_widening_cap_is_never_applied() {
        // Only ever NARROWS: a per-model entry larger than the task's own
        // limit must not silently override it.
        let _dir = conf_sandbox(&[
            ("config.toml", "[max_tokens]\nfilename = 1000\n"),
            ("models/qwen.toml", "name = \"qwen\"\nmax_tokens = 32000\n"),
        ]);
        assert_eq!(max_tokens_for_task("filename", "qwen3.8-27b-8bit"), 1000);
    }

    #[test]
    #[serial]
    fn a_brand_name_model_resolves_its_family_from_the_recorded_architecture() {
        // The NAME does not reliably encode the family: a "bonsai-*" model
        // carries no family substring, so name matching sends it nowhere --
        // but `ev` recorded its architecture, and trimming qwen3_5_moe ->
        // qwen3_5 -> qwen lands on the file written for it.
        let _dir = conf_sandbox(&[
            ("models/qwen.toml", "name = \"qwen\"\nmax_tokens = 8000\n"),
            (
                "eval_signals.json",
                r#"{"bonsai-27b": {"_capabilities": {"family": "qwen3_5_moe"}}}"#,
            ),
        ]);
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
        // The empty conf root IS the fixture: a model with no entry must find
        // nothing, and the sandbox is what guarantees it finds nothing HERE.
        let _env = conf_sandbox(&[]);
        assert_eq!(
            max_tokens_for_task("json", "totally-unknown-model"),
            DEFAULT_MAX_TOKENS
        );
    }

    #[test]
    #[serial]
    fn model_best_for_prefers_specific_model_tags_over_family_tags() {
        let _env = conf_sandbox(&[(
            "models/qwen.toml",
            "name = \"qwen\"\nbest_for = [\"family_tag\"]\n\n[models.\"qwen-special\"]\nbest_for = [\"specific_tag\"]\n",
        )]);
        assert_eq!(
            model_best_for("qwen-special"),
            vec!["specific_tag".to_string()]
        );
        assert_eq!(model_best_for("qwen-other"), vec!["family_tag".to_string()]);
        assert_eq!(model_best_for("unknown-model"), Vec::<String>::new());
    }
}
