//! `ztools weekend exclude <place>`: the ONE writer of the planner's
//! `exclude_places` list. Its CLI contract is in `docs/ROADMAP.md`, "Contracts
//! other repos build on".
//!
//! The `routines` dashboard row gets a button that calls this rather than
//! editing `conf/weekend.toml` itself, because the file has a trap that only its
//! owner should have to know about: a top-level key written BELOW a table
//! header parses as that table's key (C8 in `docs/REPORT_WEAKNESS_CLASSES.md`),
//! so an "append to the end of the file" writer silently produces a list the
//! planner never reads.
//!
//! What makes this writer safe, each property pinned in `exclude_tests.rs`:
//! - **Format-preserving**: `toml_edit` edits the list in place, so comments,
//!   table order and the list's own layout survive; a new list is inserted at
//!   the top level, never after a table.
//! - **The file the planner reads**: the target is the file [`source`] picks,
//!   which is what [`crate::ztools::weekend::load_exclusions`] reads.
//! - **Idempotent**: a place already listed (or already absent, for `--undo`)
//!   leaves the file byte-identical.
//! - **Validated by identity**: a place must be a row's bold name in the newest
//!   stored plan -- the planner's own spelling, which is what is stored -- never
//!   a row index, which churns between runs.
//! - **Atomic and serialised**: temp file + rename in the same directory, under
//!   an exclusive lock on the file, so a concurrent writer cannot lose an entry.

use std::fs::File;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use toml_edit::{Array, DocumentMut, Item, Value};

use super::enforce::normalize_for_match;

const KEY: &str = "exclude_places";
/// Longer than any venue name; a paragraph is not a place.
const MAX_PLACE_CHARS: usize = 200;
/// How long a writer waits for a peer before saying so.
const LOCK_WAIT: Duration = Duration::from_secs(10);

/// What to do with the place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Add it to the list.
    Exclude,
    /// Remove it from the list (`--undo`).
    Include,
}

/// What the edit did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Change {
    Added,
    Removed,
    /// Already in the requested state; the file was not rewritten.
    Unchanged,
}

/// A successful edit.
#[derive(Debug)]
pub struct Outcome {
    pub change: Change,
    /// The list's spelling of the place (the plan's, for a new entry).
    pub entry: String,
    /// The file the list lives in.
    pub file: PathBuf,
    /// How many entries the list holds afterwards.
    pub entries: usize,
}

/// Why nothing was written.
#[derive(Debug)]
pub enum EditError {
    /// The request was wrong: an unknown or malformed place, or a file whose
    /// list is misplaced. Retrying the same request cannot succeed.
    Refused(String),
    /// The machine failed: unreadable or unparseable file, lock, write.
    Failed(String),
}

fn same(a: &str, b: &str) -> bool {
    normalize_for_match(a) == normalize_for_match(b)
}

/// `exclude_places` at the TOP level of `text`, as TOML reads it.
fn top_level_list(text: &str) -> Option<Vec<String>> {
    let value: toml::Value = toml::from_str(text).ok()?;
    let list = value.get(KEY)?.as_array()?;
    Some(
        list.iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
    )
}

/// The first configured file with a non-empty top-level `exclude_places`, and
/// that list: the file the planner reads, and so the file this edits.
#[must_use]
pub fn source(paths: &[String]) -> Option<(PathBuf, Vec<String>)> {
    paths.iter().find_map(|raw| {
        let path = crate::manifest::expand_tilde(raw);
        let list = top_level_list(&std::fs::read_to_string(&path).ok()?)?;
        (!list.is_empty()).then_some((path, list))
    })
}

/// The row identities in a rendered plan: each row's bold name, in plan order.
#[must_use]
pub fn plan_places(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for (_, rows) in super::report::parse_tables(text) {
        for row in rows {
            let cell = row
                .iter()
                .find(|(column, _)| column.ends_with("& Location"))
                .map(|(_, v)| v.as_str());
            let name = cell
                .and_then(|c| c.split_once("**"))
                .and_then(|(_, rest)| rest.split_once("**"))
                .map(|(name, _)| name.trim());
            if let Some(name) = name.filter(|n| !n.is_empty()) {
                out.push(name.to_string());
            }
        }
    }
    out
}

fn validate(place: &str) -> Result<&str, EditError> {
    let place = place.trim();
    if place.is_empty() {
        return Err(EditError::Refused("the place is empty".into()));
    }
    if place.chars().any(char::is_control) {
        return Err(EditError::Refused(format!(
            "{place:?} contains a control character; a place is one line of text"
        )));
    }
    if place.chars().count() > MAX_PLACE_CHARS {
        return Err(EditError::Refused(format!(
            "the place is longer than {MAX_PLACE_CHARS} characters"
        )));
    }
    Ok(place)
}

/// The plan's own spelling of `place`, or why it is not a place the planner knows.
fn plan_identity(plan_dir: &Path, place: &str) -> Result<String, EditError> {
    let Some(plan) = crate::ztools::store::newest_dated_plan(plan_dir) else {
        return Err(EditError::Refused(format!(
            "no stored plan in {} to check {place:?} against; run the planner first",
            plan_dir.display()
        )));
    };
    let text = std::fs::read_to_string(&plan)
        .map_err(|e| EditError::Failed(format!("cannot read {}: {e}", plan.display())))?;
    plan_places(&text)
        .into_iter()
        .find(|name| same(name, place))
        .ok_or_else(|| {
            EditError::Refused(format!(
                "{place:?} is not a place in the newest plan ({}); send a row's bold name \
                 exactly as the plan prints it",
                plan.display()
            ))
        })
}

/// The file to edit: the planner's source, else the first configured file
/// that exists. Never a new file -- a list nobody configured is read by nobody.
fn target(paths: &[String]) -> Result<PathBuf, EditError> {
    let path = source(paths).map(|(p, _)| p).or_else(|| {
        paths
            .iter()
            .map(|p| crate::manifest::expand_tilde(p))
            .find(|p| p.is_file())
    });
    let path = path.ok_or_else(|| {
        EditError::Failed(format!(
            "no weekend config file exists among {paths:?} (weekend_exclusions_paths)"
        ))
    })?;
    // Through a symlink to the real file: a rename onto the link would replace
    // the link, and the dotfile it pointed at would keep the old list.
    std::fs::canonicalize(&path)
        .map_err(|e| EditError::Failed(format!("cannot resolve {}: {e}", path.display())))
}

/// An exclusive lock on `path` that is still `path` once held.
///
/// The lock is on the file itself (no sidecar left in the checkout). Because
/// the write renames a new file into place, a writer that waited may hold the
/// lock on a file no longer at `path`; it then retries on the new one.
fn lock(path: &Path) -> Result<File, EditError> {
    use std::os::unix::fs::MetadataExt as _;
    let fail = |e: std::io::Error| EditError::Failed(format!("{}: {e}", path.display()));
    let deadline = Instant::now() + LOCK_WAIT;
    loop {
        let file = File::open(path).map_err(fail)?;
        match file.try_lock() {
            Ok(()) => {
                let (held, now) = (
                    file.metadata().map_err(fail)?,
                    std::fs::metadata(path).map_err(fail)?,
                );
                if (held.dev(), held.ino()) == (now.dev(), now.ino()) {
                    return Ok(file);
                }
            }
            Err(std::fs::TryLockError::WouldBlock) => {}
            Err(std::fs::TryLockError::Error(e)) => return Err(fail(e)),
        }
        if Instant::now() >= deadline {
            return Err(EditError::Failed(format!(
                "another writer has held {} for {}s",
                path.display(),
                LOCK_WAIT.as_secs()
            )));
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn write_atomic(path: &Path, text: &str) -> Result<(), EditError> {
    let fail = |e: std::io::Error| EditError::Failed(format!("writing {}: {e}", path.display()));
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let mut tmp = tempfile::NamedTempFile::new_in(dir).map_err(fail)?;
    tmp.write_all(text.as_bytes()).map_err(fail)?;
    tmp.as_file().sync_all().map_err(fail)?;
    let permissions = std::fs::metadata(path).map_err(fail)?.permissions();
    tmp.as_file().set_permissions(permissions).map_err(fail)?;
    tmp.persist(path).map_err(|e| fail(e.error))?;
    Ok(())
}

/// Refuse a file whose list sits under a table (C8): adding a top-level one
/// beside it would leave two lists, one of them dead.
fn refuse_misplaced(doc: &DocumentMut, file: &Path) -> Result<(), EditError> {
    for (name, item) in doc.iter() {
        let nested = match item {
            Item::Table(t) => t.contains_key(KEY),
            Item::ArrayOfTables(a) => a.iter().any(|t| t.contains_key(KEY)),
            _ => false,
        };
        if nested {
            return Err(EditError::Refused(format!(
                "{} has `{KEY}` under [{name}], where TOML reads it as that table's key and \
                 the planner never sees it (C8); move it above the first table",
                file.display()
            )));
        }
    }
    Ok(())
}

/// The list's strings, or why it is not a list of strings.
fn strings(list: &Array, file: &Path) -> Result<Vec<String>, EditError> {
    list.iter()
        .map(|v| v.as_str().map(str::to_string))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            EditError::Failed(format!(
                "`{KEY}` in {} is not a list of strings",
                file.display()
            ))
        })
}

/// A new multi-line list holding `entries`, laid out like the shipped config.
fn new_list(entries: &[String]) -> Array {
    let mut list = Array::new();
    for entry in entries {
        push(&mut list, entry);
    }
    list.set_trailing("\n");
    list
}

/// Append `entry` on its own line, at the indent of the existing entries.
fn push(list: &mut Array, entry: &str) {
    let indent = list
        .iter()
        .last()
        .and_then(|v| v.decor().prefix())
        .and_then(|p| p.as_str())
        .and_then(|p| p.rsplit_once('\n'))
        .map_or_else(|| "  ".to_string(), |(_, indent)| indent.to_string());
    let mut value = Value::from(entry);
    value.decor_mut().set_prefix(format!("\n{indent}"));
    list.push_formatted(value);
    list.set_trailing_comma(true);
}

/// Remove entry `index`, keeping any comment that sat above it.
fn remove(list: &mut Array, index: usize) {
    let removed = list.remove(index);
    let prefix = removed
        .decor()
        .prefix()
        .and_then(|p| p.as_str())
        .unwrap_or("")
        .trim_end_matches([' ', '\t'])
        .to_string();
    if !prefix.trim().is_empty() {
        if let Some(next) = list.get_mut(index) {
            let own = next
                .decor()
                .prefix()
                .and_then(|p| p.as_str())
                .unwrap_or("")
                .to_string();
            let indent = own.rsplit_once('\n').map_or("", |(_, i)| i);
            next.decor_mut().set_prefix(format!("{prefix}\n{indent}"));
        } else {
            let trailing = list.trailing().as_str().unwrap_or("").to_string();
            list.set_trailing(format!("{prefix}{trailing}"));
        }
    }
}

/// Exclude or re-include `place`, under the lock, atomically.
///
/// # Errors
///
/// [`EditError::Refused`] for a request that cannot succeed as sent;
/// [`EditError::Failed`] when the file, the lock or the write fails.
pub fn edit(
    paths: &[String],
    plan_dir: &Path,
    action: Action,
    place: &str,
) -> Result<Outcome, EditError> {
    let place = validate(place)?;
    let file = target(paths)?;
    let _lock = lock(&file)?;
    let text = std::fs::read_to_string(&file)
        .map_err(|e| EditError::Failed(format!("cannot read {}: {e}", file.display())))?;
    let mut doc: DocumentMut = text
        .parse()
        .map_err(|e| EditError::Failed(format!("{} is not valid TOML: {e}", file.display())))?;
    refuse_misplaced(&doc, &file)?;

    let existing = match doc.get(KEY) {
        None => None,
        Some(item) => Some(strings(
            item.as_array().ok_or_else(|| {
                EditError::Failed(format!("`{KEY}` in {} is not a list", file.display()))
            })?,
            &file,
        )?),
    };
    // No list yet: the planner is running on its built-in one, so a new list
    // starts from that -- excluding one place must not re-include the rest.
    let current = existing
        .clone()
        .unwrap_or_else(crate::ztools::weekend_cache::builtin_exclusions);
    let hit = current.iter().position(|e| same(e, place));

    let (change, entry) = match (action, hit) {
        (Action::Exclude, Some(i)) => (Change::Unchanged, current[i].clone()),
        (Action::Exclude, None) => (Change::Added, plan_identity(plan_dir, place)?),
        (Action::Include, Some(i)) => (Change::Removed, current[i].clone()),
        (Action::Include, None) => (Change::Unchanged, place.to_string()),
    };
    let mut entries = current.len();
    if change != Change::Unchanged {
        if existing.is_none() {
            doc.insert(KEY, Item::Value(Value::Array(new_list(&current))));
        }
        let list = doc
            .get_mut(KEY)
            .and_then(Item::as_array_mut)
            .ok_or_else(|| EditError::Failed("the list vanished mid-edit".into()))?;
        match (change, hit) {
            (Change::Removed, Some(i)) => remove(list, i),
            _ => push(list, &entry),
        }
        entries = list.len();
        write_atomic(&file, &doc.to_string())?;
    }
    Ok(Outcome {
        change,
        entry,
        file,
        entries,
    })
}

/// The machine-readable result and the process exit code (the contract the
/// `routines` button parses; documented in `docs/ROADMAP.md`).
#[must_use]
pub fn report(
    action: Action,
    place: &str,
    result: &Result<Outcome, EditError>,
) -> (serde_json::Value, u8) {
    let action = match action {
        Action::Exclude => "exclude",
        Action::Include => "include",
    };
    let base = |result: &str, code: u8, reason: Option<&str>| {
        (
            serde_json::json!({
                "schema": 1,
                "action": action,
                "place": place,
                "result": result,
                "entry": null,
                "file": null,
                "entries": null,
                "reason": reason,
            }),
            code,
        )
    };
    match result {
        Ok(out) => {
            let word = match out.change {
                Change::Added => "added",
                Change::Removed => "removed",
                Change::Unchanged => "unchanged",
            };
            let (mut json, code) = base(word, 0, None);
            json["entry"] = out.entry.clone().into();
            json["file"] = out.file.to_string_lossy().into_owned().into();
            json["entries"] = out.entries.into();
            (json, code)
        }
        Err(EditError::Refused(reason)) => base("refused", 3, Some(reason)),
        Err(EditError::Failed(reason)) => base("failed", 1, Some(reason)),
    }
}

#[cfg(test)]
#[path = "exclude_tests.rs"]
mod tests;
