//! The machine-local selection file, `~/.config/ztools/config.toml [best_models]`.
//!
//! `with_ztools_best_models` reads the FIRST config.toml carrying a
//! `[best_models]` table and takes that table whole, so a file this job creates
//! is seeded with every slot's current model before the one slot changes:
//! otherwise the other four would silently fall back to the embedded defaults.
//! Every other key and comment in the file is kept (`toml_edit`).

use std::collections::BTreeMap;
use std::path::Path;

/// The file's text, or `None` when there is none.
///
/// # Errors
/// When it exists and cannot be read.
pub fn read(path: &Path) -> anyhow::Result<Option<String>> {
    match std::fs::read_to_string(path) {
        Ok(t) => Ok(Some(t)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(anyhow::anyhow!("{}: {e}", path.display())),
    }
}

/// The `[best_models]` table of `text`, if it has one.
#[must_use]
pub fn slots(text: &str) -> Option<BTreeMap<String, String>> {
    let doc: toml_edit::DocumentMut = text.parse().ok()?;
    let table = doc.get("best_models")?.as_table_like()?;
    Some(
        table
            .iter()
            .filter_map(|(k, v)| v.as_str().map(|s| (k.to_string(), s.to_string())))
            .collect(),
    )
}

/// `text` (or a new file) with `slot = model` in `[best_models]`, seeding a
/// missing table from `base`.
///
/// # Errors
/// When `text` is not TOML, or its `best_models` is not a table.
pub fn with_slot(
    text: Option<&str>,
    base: &BTreeMap<String, String>,
    slot: &str,
    model: &str,
) -> anyhow::Result<String> {
    let mut doc: toml_edit::DocumentMut = text.unwrap_or("").parse()?;
    if doc.get("best_models").is_none() {
        let mut table = toml_edit::Table::new();
        for (k, v) in base {
            table.insert(k, toml_edit::value(v.as_str()));
        }
        doc.insert("best_models", toml_edit::Item::Table(table));
    }
    let table = doc
        .get_mut("best_models")
        .and_then(toml_edit::Item::as_table_like_mut)
        .ok_or_else(|| anyhow::anyhow!("[best_models] is not a table"))?;
    table.insert(slot, toml_edit::value(model));
    Ok(doc.to_string())
}

/// Write `text` to `path` whole (temp file, then rename), or remove the file
/// when `text` is `None`.
///
/// # Errors
/// When the directory or the file cannot be written.
pub fn write(path: &Path, text: Option<&str>) -> anyhow::Result<()> {
    match text {
        None => match std::fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(anyhow::anyhow!("{}: {e}", path.display())),
        },
        Some(t) => {
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let tmp = path.with_extension("toml.tmp");
            std::fs::write(&tmp, t)?;
            std::fs::rename(&tmp, path)?;
            Ok(())
        }
    }
}
