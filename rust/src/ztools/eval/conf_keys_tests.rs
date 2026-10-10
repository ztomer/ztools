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
//!   [`CONFIG_KEYS_READ`], or is a [`CONFIG_KEYS_UNREAD`] ratchet entry;
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

/// Keys of `conf/config.toml` that a reader consumes: `[best_models]`
/// (`config.rs`), `[timeouts]` (`signals.rs`), `[max_tokens]` (`budgets.rs`).
const CONFIG_KEYS_READ: &[&str] = &[
    "best_models",
    "best_models.*",
    "timeouts",
    "timeouts.*",
    "max_tokens",
    "max_tokens.*",
];

/// Top-level keys of `conf/config.toml` that NO Rust code reads today, each
/// left in place because it records an owner decision that deleting would
/// silently drop, and wiring would change production behaviour. A RATCHET:
/// an entry is deleted when its key is wired or removed, and the test fails
/// on an entry whose key has gone.
const CONFIG_KEYS_UNREAD: &[(&str, &str)] = &[
    (
        "llm_url",
        "the server URL lives in the --config file (osaurus_url); this copy is stale prose",
    ),
    (
        "default_model",
        "records the think winner; production takes think_model from the --config file",
    ),
    (
        "filename_models",
        "records the owner's 2026-08-18 fallback-order trade for rn; rn has no fallback chain",
    ),
    (
        "model_fallback_chain",
        "Python's substitution order; Rust uses default_fallback_chain() (different order)",
    ),
];

fn conf_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("rust/ has a parent")
        .join("conf")
}

/// Every key path in `v`, dotted from the root, with a model id under
/// `models` (or a slot under a `CONFIG_KEYS_READ` table) written as `*`.
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

/// Whether `src` fetches `key` by name: `get("key")` or `["key"]`.
fn fetched(src: &str, key: &str) -> bool {
    src.contains(&format!("get(\"{key}\")")) || src.contains(&format!("[\"{key}\"]"))
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
fn every_config_toml_key_has_a_reader_or_a_ratchet_entry() {
    let keys = file_keys(&conf_dir().join("config.toml"));
    assert!(keys.iter().any(|k| k == "best_models"), "control: {keys:?}");
    let inert: Vec<&String> = keys
        .iter()
        .filter(|k| !CONFIG_KEYS_READ.contains(&k.as_str()))
        .filter(|k| !CONFIG_KEYS_UNREAD.iter().any(|(u, _)| u == k))
        .collect();
    assert!(inert.is_empty(), "keys nothing reads: {inert:#?}");
    let stale: Vec<&str> = CONFIG_KEYS_UNREAD
        .iter()
        .map(|(k, _)| *k)
        .filter(|k| !keys.iter().any(|have| have == k))
        .collect();
    assert!(
        stale.is_empty(),
        "ratchet entries whose key is gone: {stale:?}"
    );
}

#[test]
fn every_allowlisted_key_is_fetched_and_no_deleted_one_is() {
    let src = all_source();
    let named = MODEL_KEYS_READ
        .iter()
        .chain(CONFIG_KEYS_READ)
        .filter_map(|k| k.rsplit('.').next())
        .filter(|leaf| *leaf != "*");
    let unread: Vec<&str> = named.filter(|leaf| !fetched(&src, leaf)).collect();
    assert!(
        unread.is_empty(),
        "allowlisted but fetched nowhere: {unread:?}"
    );

    let ratchet_read: Vec<&str> = CONFIG_KEYS_UNREAD
        .iter()
        .map(|(k, _)| *k)
        .filter(|k| fetched(&src, k))
        .collect();
    assert!(
        ratchet_read.is_empty(),
        "now read -- move from CONFIG_KEYS_UNREAD to CONFIG_KEYS_READ: {ratchet_read:?}"
    );

    let deleted = [
        "prompts",
        "field_mapping",
        "key_mappings",
        "top_keys",
        "quirks",
        "eval_inputs",
        "internal_name",
    ];
    let read_again: Vec<&str> = deleted.into_iter().filter(|k| fetched(&src, k)).collect();
    assert!(
        read_again.is_empty(),
        "a deleted table is read again: {read_again:?}"
    );
}
