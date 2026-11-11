//! The minimal one-section document a repair applies, and the check that its
//! `osaurus config plan` diff touches nothing but the one reference.
//!
//! A full export is never fed back: it is not re-appliable (plan rejects an
//! automation in it), and a document that restates a section re-sets what the
//! app has since changed. The document holds `version` and the one entity.

use super::CmdOut;
use super::refs::Slot;
use serde_json::{Value, json};

/// Where in the plan output the one permitted change must appear.
#[derive(Debug, PartialEq, Eq)]
pub struct Edit {
    pub section: &'static str,
    pub item: String,
    pub field: &'static str,
}

/// The plan lines a repair of `slot` must produce. `None` for a slot that is
/// not an Osaurus setting.
#[must_use]
pub fn edit_for(slot: &Slot) -> Option<Edit> {
    let (section, item, field) = match slot {
        Slot::CoreModel => ("model_defaults", "model_defaults".into(), "core_model"),
        Slot::CompactionModel => (
            "model_defaults",
            "model_defaults".into(),
            "compaction_model",
        ),
        Slot::DefaultAgent => ("default_agent", "default_agent".into(), "model"),
        Slot::Configurator => ("configurator", "configurator".into(), "model"),
        Slot::Agent(n) => ("agents", n.clone(), "model"),
        Slot::Automation(n) => ("automations", n.clone(), "model"),
        Slot::Best(_) => return None,
    };
    Some(Edit {
        section,
        item,
        field,
    })
}

/// The one-section document that sets `slot` to `new`.
///
/// # Errors
///
/// When the slot is not an Osaurus setting, or an automation cannot be found
/// in the export (the document restates the whole entity).
pub fn minimal_doc(export: &Value, slot: &Slot, new: &str) -> Result<Value, String> {
    let version = export.get("version").cloned().unwrap_or_else(|| json!(1));
    let body = match slot {
        Slot::CoreModel => json!({"model_defaults": {"core_model": new}}),
        Slot::CompactionModel => json!({"model_defaults": {"compaction_model": new}}),
        Slot::DefaultAgent => json!({"default_agent": {"model": new}}),
        Slot::Configurator => json!({"configurator": {"model": new}}),
        Slot::Agent(name) => json!({"agents": [{"name": name, "model": new}]}),
        Slot::Automation(name) => {
            // Osaurus replaces an automation whole: a key the document leaves
            // out is unset (`fresh_session: true -> (unset)` in the plan), so
            // the document is the export's own entity with the one field
            // changed and every other key kept.
            let mut entity = export
                .get("automations")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .find(|a| a.get("name").and_then(Value::as_str) == Some(name))
                .cloned()
                .ok_or_else(|| format!("automation {name} is not in the export"))?;
            entity["model"] = json!(new);
            json!({"automations": [entity]})
        }
        Slot::Best(_) => return Err("[best_models] is not an Osaurus setting".into()),
    };
    let mut doc = json!({ "version": version });
    if let (Some(d), Some(b)) = (doc.as_object_mut(), body.as_object()) {
        d.extend(b.clone());
    }
    Ok(doc)
}

/// Accept the plan only if it is exactly the one change wanted.
///
/// # Errors
///
/// The reason, naming what the plan did instead: a non-zero exit, a second
/// section, item or field, or a different new value.
pub fn verify_plan(edit: &Edit, new: &str, out: &CmdOut) -> Result<(), String> {
    let first = |t: &str| t.lines().take(6).collect::<Vec<_>>().join(" | ");
    if out.code != 0 {
        return Err(format!("plan exited {}: {}", out.code, first(&out.text)));
    }
    let lines: Vec<&str> = out
        .text
        .lines()
        .map(str::trim_end)
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with("note:"))
        .collect();
    let head = format!("      {}: ", edit.field);
    let exact = lines.len() == 3
        && lines[0] == format!("{}:", edit.section)
        && lines[1] == format!("  ~ {}", edit.item)
        && lines[2].starts_with(&head)
        && lines[2].ends_with(&format!(" -> {new}"));
    if exact {
        return Ok(());
    }
    Err(format!(
        "the plan touches more than {}.{}: {}",
        edit.item,
        edit.field,
        first(&out.text)
    ))
}
