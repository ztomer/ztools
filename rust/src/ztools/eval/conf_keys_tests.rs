//! Every key the shipped conf holds is a key some reader consumes.
//!
//! The class is inert config that reads like live config (Rust rule 6): an
//! operator edits it, nothing changes, and nothing says so. W5 found per-model
//! `[prompts]` tables in `conf/models/*.toml` that nothing had loaded since the
//! Python runtime died, and deleted them behind a test that named `prompts`.
//! M18 found the rest of that file's dead weight -- `[field_mapping]`,
//! `[key_mappings]`, `[top_keys]`, `[[quirks]]`, `internal_name`, `name` --
//! and, in `conf/config.toml`, `[eval_inputs]` and a second `[prompts]`. A
//! test that names each deleted table only stops THOSE coming back, so this
//! one is an ALLOWLIST of what is read, checked from both sides:
//!
//! * every key in the files matches a pattern in [`MODEL_KEYS_READ`] /
//!   [`CONF_FILE_KEYS`], and every `conf/*.toml` is in that table;
//! * every allowlisted key is fetched by name somewhere in `src/` (an entry
//!   nothing reads is the same lie in the allowlist), and every ratchet entry
//!   still exists in the file (a stale entry fails rather than rots);
//! * no source fetches a key the files no longer hold.
//!
//! Where the key aliases, top-level keys and quirks actually live, so nobody
//! re-adds them here: `weekend::ALT_KEYS` (alias normalisation),
//! `weekend::parse_llm_events` (reads `transient_events` only), and the
//! constants in `eval/quirks.rs` (the qwen JSON trigger).

use std::fs;
use std::path::{Path, PathBuf};

/// Keys of a `conf/models/<family>.toml` file that `budgets.rs` and
/// `model_resolve/disk.rs` read. `*` is any one model id under `[models]`.
const MODEL_KEYS_READ: &[&str] = &[
    "best_for",
    "timeout",
    "max_tokens",
    "context_window",
    "models",
    "models.*",
    "models.*.best_for",
    "models.*.timeout",
    "models.*.max_tokens",
];

/// Every `conf/*.toml` and the keys some reader consumes in it. A conf file
/// not listed here fails the scan (a new file arrives checked or not at all).
/// `*` is any one slot under `[best_models]`, `[timeouts]` or `[max_tokens]`.
///
/// 2026-10-10 found the class outside the two files M18 scanned: `llm_url` in
/// three files, `twitter.toml`'s `state_file`/`output_dir`/`chrome_cookies_db`/
/// `max_scrolls`, all of `rename.toml`, a `[rename.relevance_check]` prompt for
/// a check never ported -- and `weekend.toml`'s coordinates, read by nothing
/// while the forecast URL hard-coded them (now wired, `weekend/place.rs`).
const CONF_FILE_KEYS: &[(&str, &[&str])] = &[
    (
        "config.toml",
        &[
            "best_models",
            "best_models.*",
            "timeouts",
            "timeouts.*",
            "max_tokens",
            "max_tokens.*",
        ],
    ),
    (
        "twitter.toml",
        &[
            "endpoints",
            "endpoints.timeline",
            "endpoints.following",
            "fallback",
            "fallback.models",
            "fallback.preferred",
        ],
    ),
    (
        "weekend.toml",
        &[
            "exclude_places",
            "location",
            "location.city",
            "location.region",
            "location.timezone",
            "location.province",
            "location.latitude",
            "location.longitude",
            "region",
            "region.in_region",
            "region.foreign",
            "children",
            "search",
            "search.demote_after_runs",
            "search.demote_wall_ratio",
            "llm",
            "llm.phase_retries",
        ],
    ),
    (
        "prompts.toml",
        &[
            "twitter",
            "twitter.summarize",
            "twitter.summarize.instructions",
            "weekend",
            "weekend.weather_condense",
            "weekend.weather_condense.instructions",
            "weekend.carry_fields",
            "weekend.carry_fields.instructions",
            "weekend.activity_rule",
            "weekend.activity_rule.instructions",
            "weekend.extract_events",
            "weekend.extract_events.instructions",
            "weekend.draft_transient",
            "weekend.draft_transient.instructions",
            "weekend.refine",
            "weekend.refine.instructions",
            "weekend.weather_rule",
            "weekend.weather_rule.instructions",
            "weekend.family_rule",
            "weekend.family_rule.instructions",
            "weekend.structure_transient_system",
            "weekend.structure_transient_system.instructions",
            "weekend.structure_user",
            "weekend.structure_user.instructions",
            "rename",
            "rename.filename",
            "rename.filename.instructions",
            "rename.task_restatement",
            "rename.task_restatement.instructions",
            "rename.image_to_filename",
            "rename.image_to_filename.instructions",
        ],
    ),
    ("eval_inputs.toml", &["test_inputs", "test_inputs.filename"]),
    ("eval_vision.toml", &["size", "fixture"]),
];

fn conf_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("rust/ has a parent")
        .join("conf")
}

/// Every key path in `v`, dotted from the root, with a model id under
/// `models` (or a slot under a `config.toml` slot table) written as `*`.
fn key_paths(v: &toml::Value, prefix: &[String], out: &mut Vec<String>) {
    let Some(table) = v.as_table() else { return };
    for (k, child) in table {
        let mut here = prefix.to_vec();
        here.push(k.clone());
        out.push(normalise(&here));
        key_paths(child, &here, out);
    }
}

fn normalise(segments: &[String]) -> String {
    let wild = |i: usize, s: &String| -> String {
        if i == 1
            && ["models", "best_models", "timeouts", "max_tokens"].contains(&segments[0].as_str())
        {
            "*".to_string()
        } else {
            s.clone()
        }
    };
    segments
        .iter()
        .enumerate()
        .map(|(i, s)| wild(i, s))
        .collect::<Vec<_>>()
        .join(".")
}

fn file_keys(path: &Path) -> Vec<String> {
    let text = fs::read_to_string(path).expect("a shipped conf file reads");
    let val: toml::Value = toml::from_str(&text).expect("a shipped conf file parses");
    let mut keys = Vec::new();
    key_paths(&val, &[], &mut keys);
    keys
}

/// Every `.rs` file under `src/`, concatenated.
fn all_source() -> String {
    let mut stack = vec![Path::new(env!("CARGO_MANIFEST_DIR")).join("src")];
    let mut text = String::new();
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).expect("src is readable").flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs")
                && !path.ends_with("conf_keys_tests.rs")
            {
                text.push_str(&fs::read_to_string(&path).unwrap_or_default());
            }
        }
    }
    text
}

/// Whether `src` names `key` the way a reader does: as a string literal
/// (`get("key")`, `["city", "region"]`, `rename = "fixture"`), as the last
/// segment of a dotted one (`"weekend.family_rule"`), or as a serde field
/// (`pub size:`).
fn fetched(src: &str, key: &str) -> bool {
    src.contains(&format!("\"{key}\""))
        || src.contains(&format!(".{key}\""))
        || src.contains(&format!("\"{key} = "))
        || src.contains(&format!("pub {key}:"))
}

#[test]
fn every_family_config_key_has_a_reader() {
    let mut inert = Vec::new();
    let mut seen_best_for = false;
    for entry in fs::read_dir(conf_dir().join("models"))
        .expect("conf/models is readable")
        .flatten()
    {
        let name = entry.file_name().to_string_lossy().into_owned();
        for key in file_keys(&entry.path()) {
            seen_best_for |= key == "best_for";
            if !MODEL_KEYS_READ.contains(&key.as_str()) {
                inert.push(format!("{name}: {key}"));
            }
        }
    }
    assert!(
        seen_best_for,
        "control: the scan must see a key the readers DO consume"
    );
    assert!(inert.is_empty(), "keys nothing reads: {inert:#?}");
}

#[test]
fn every_conf_file_key_has_a_reader_and_every_conf_file_is_scanned() {
    let mut inert = Vec::new();
    let mut unscanned = Vec::new();
    for entry in fs::read_dir(conf_dir())
        .expect("conf is readable")
        .flatten()
    {
        let name = entry.file_name().to_string_lossy().into_owned();
        if entry.path().extension().is_none_or(|e| e != "toml") {
            continue;
        }
        let Some((_, read)) = CONF_FILE_KEYS.iter().find(|(f, _)| *f == name) else {
            unscanned.push(name);
            continue;
        };
        for key in file_keys(&entry.path()) {
            if !read.contains(&key.as_str()) {
                inert.push(format!("{name}: {key}"));
            }
        }
    }
    assert!(
        unscanned.is_empty(),
        "conf files no allowlist covers: {unscanned:?}"
    );
    assert!(inert.is_empty(), "keys nothing reads: {inert:#?}");
    let gone: Vec<&str> = CONF_FILE_KEYS
        .iter()
        .map(|(f, _)| *f)
        .filter(|f| !conf_dir().join(f).is_file())
        .collect();
    assert!(gone.is_empty(), "allowlisted files that are gone: {gone:?}");
}

#[test]
fn every_allowlisted_key_is_fetched_and_no_deleted_one_is() {
    let src = all_source();
    let named = MODEL_KEYS_READ
        .iter()
        .chain(CONF_FILE_KEYS.iter().flat_map(|(_, keys)| keys.iter()))
        .filter_map(|k| k.rsplit('.').next())
        .filter(|leaf| *leaf != "*");
    let unread: Vec<&str> = named.filter(|leaf| !fetched(&src, leaf)).collect();
    assert!(
        unread.is_empty(),
        "allowlisted but fetched nowhere: {unread:?}"
    );

    let deleted = [
        "prompts",
        "field_mapping",
        "key_mappings",
        "top_keys",
        "quirks",
        "eval_inputs",
        "internal_name",
        "llm_url",
        "default_model",
        "filename_models",
        "model_fallback_chain",
        "relevance_check",
        "chrome_cookies_db",
    ];
    let read_again: Vec<&str> = deleted.into_iter().filter(|k| fetched(&src, k)).collect();
    assert!(
        read_again.is_empty(),
        "a deleted table is read again: {read_again:?}"
    );
}
