//! Where an Osaurus export names a model, and which names are installed.
//!
//! A reference is a path into the export (`agents[Disk Audit].model`) plus the
//! model it holds. `null`, an absent key, and `foundation` are not references
//! to anything installable; everything else must resolve (`Installed`).

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

/// Which setting a reference lives in. It decides the minimal repair document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Slot {
    CoreModel,
    CompactionModel,
    DefaultAgent,
    Configurator,
    Agent(String),
    Automation(String),
    /// A slot of the ztools `[best_models]` table. Not an Osaurus setting.
    Best(String),
}

/// One place a model is named.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reference {
    /// The exact path, as the error line prints it.
    pub path: String,
    pub slot: Slot,
    pub model: String,
}

fn model_of(v: &Value) -> Option<&str> {
    v.as_str().map(str::trim).filter(|s| !s.is_empty())
}

fn named<'a>(list: &'a Value, key: &str) -> impl Iterator<Item = (&'a str, &'a Value)> {
    list.as_array()
        .into_iter()
        .flatten()
        .filter_map(move |item| Some((item.get("name")?.as_str()?, item.get(key)?)))
}

/// Every model reference in an export, then the `[best_models]` table.
#[must_use]
pub fn collect(export: &Value, best: &BTreeMap<String, String>) -> Vec<Reference> {
    let mut refs = Vec::new();
    let mut push = |path: String, slot: Slot, v: Option<&Value>| {
        if let Some(model) = v.and_then(model_of) {
            refs.push(Reference {
                path,
                slot,
                model: model.to_string(),
            });
        }
    };
    let defaults = export.get("model_defaults");
    push(
        "model_defaults.core_model".into(),
        Slot::CoreModel,
        defaults.and_then(|d| d.get("core_model")),
    );
    push(
        "model_defaults.compaction_model".into(),
        Slot::CompactionModel,
        defaults.and_then(|d| d.get("compaction_model")),
    );
    push(
        "default_agent.model".into(),
        Slot::DefaultAgent,
        export.get("default_agent").and_then(|d| d.get("model")),
    );
    push(
        "configurator.model".into(),
        Slot::Configurator,
        export.get("configurator").and_then(|d| d.get("model")),
    );
    for (name, v) in named(export.get("agents").unwrap_or(&Value::Null), "model") {
        push(
            format!("agents[{name}].model"),
            Slot::Agent(name.to_string()),
            Some(v),
        );
    }
    for (name, v) in named(export.get("automations").unwrap_or(&Value::Null), "model") {
        push(
            format!("automations[{name}].model"),
            Slot::Automation(name.to_string()),
            Some(v),
        );
    }
    for (slot, model) in best {
        refs.push(Reference {
            path: format!("best_models.{slot}"),
            slot: Slot::Best(slot.clone()),
            model: model.clone(),
        });
    }
    refs
}

/// The last path component, lowercased: the name `osaurus list` and the
/// eval history key a model by.
#[must_use]
pub fn short_name(model: &str) -> String {
    model.rsplit('/').next().unwrap_or(model).to_lowercase()
}

/// What is installed, and which cloud models are offered.
#[derive(Debug, Default)]
pub struct Installed {
    /// `org/name`, case-sensitive: the export's `models` plus the disk.
    ids: BTreeSet<String>,
    /// Lowercased last components, plus whatever `osaurus list` printed.
    short: BTreeSet<String>,
    /// `<provider>/` prefixes of enabled remote providers (lowercased).
    remote_prefixes: Vec<String>,
    /// Pinned ids of enabled providers.
    remote_ids: BTreeSet<String>,
}

impl Installed {
    /// Union the three sources of "installed".
    #[must_use]
    pub fn gather(export: &Value, listed: &[String], disk: &[String]) -> Self {
        let mut me = Self::default();
        let from_export = export
            .get("models")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str);
        for id in from_export.chain(disk.iter().map(String::as_str)) {
            me.ids.insert(id.to_string());
            me.short.insert(short_name(id));
        }
        for name in listed {
            me.short.insert(name.trim().to_lowercase());
        }
        for p in export
            .get("providers")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|p| p.get("enabled").and_then(Value::as_bool) == Some(true))
        {
            for key in ["provider", "name"] {
                if let Some(s) = p.get(key).and_then(Value::as_str) {
                    me.remote_prefixes.push(format!("{}/", s.to_lowercase()));
                }
            }
            for id in p
                .get("manual_model_ids")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
            {
                me.remote_ids.insert(id.to_string());
            }
        }
        me
    }

    /// The installed `org/name` ids, for choosing a repair target.
    #[must_use]
    pub fn local_ids(&self) -> Vec<String> {
        self.ids.iter().cloned().collect()
    }

    /// Whether `r` names something that exists. A local id is matched exactly
    /// (case-sensitive); a `[best_models]` entry is a short name.
    #[must_use]
    pub fn is_valid(&self, r: &Reference) -> bool {
        let m = r.model.as_str();
        if m == "foundation" || self.ids.contains(m) || self.remote_ids.contains(m) {
            return true;
        }
        let lower = m.to_lowercase();
        if self.remote_prefixes.iter().any(|p| lower.starts_with(p)) {
            return true;
        }
        matches!(r.slot, Slot::Best(_)) && self.short.contains(&short_name(m))
    }
}
