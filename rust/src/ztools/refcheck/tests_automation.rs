//! The fake's `osaurus config plan` and `apply`, and the automation repair they
//! prove: an `automations[].model` reference is repaired in place, every other
//! key of the entity kept (ZT.REFCHECKGAPS a).
//!
//! The two entity kinds merge differently, and the simulation says so because
//! the repair depends on it. An AGENT merges by key: a document naming only
//! `name` and `model` leaves the rest of the agent alone (the Disk Audit repair
//! of LM.REFCHECK planned as one `model:` line). An AUTOMATION is replaced
//! whole: a key the document leaves out is unset, which is the measured plan
//! `fresh_session: true -> (unset)` that made every automation repair abort.

use super::tests::{BROKEN, CRACK, Fake, export_with};
use super::*;
use serde_json::{Map, Value, json};

/// How the plan prints one value: a string bare, `null` as `(unset)`.
fn shown(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) => "(unset)".into(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
    }
}

fn entity<'a>(export: &'a Value, section: &str, name: &Value) -> Option<&'a Map<String, Value>> {
    export[section]
        .as_array()?
        .iter()
        .find(|e| e["name"] == *name)?
        .as_object()
}

/// The plan text osaurus prints for `doc` against `export`.
pub fn simulated_plan(export: &Value, doc: &Value) -> String {
    let mut lines: Vec<String> = Vec::new();
    for section in ["agents", "automations"] {
        let Some(list) = doc[section].as_array() else {
            continue;
        };
        lines.push(format!("{section}:"));
        for new in list.iter().filter_map(Value::as_object) {
            let name = new.get("name").cloned().unwrap_or(Value::Null);
            lines.push(format!("  ~ {}", shown(Some(&name))));
            let empty = Map::new();
            let old = entity(export, section, &name).unwrap_or(&empty);
            let mut keys: Vec<&String> = old.keys().chain(new.keys()).collect();
            keys.sort();
            keys.dedup();
            for key in keys.into_iter().filter(|k| *k != "name") {
                let kept = if section == "automations" {
                    new.get(key)
                } else {
                    new.get(key).or_else(|| old.get(key))
                };
                if kept != old.get(key) {
                    lines.push(format!(
                        "      {key}: {} -> {}",
                        shown(old.get(key)),
                        shown(kept)
                    ));
                }
            }
        }
    }
    lines.iter().flat_map(|l| [l.as_str(), "\n"]).collect()
}

/// `apply` of `doc` to `export`, merging as [`simulated_plan`] describes.
pub fn simulated_apply(export: &mut Value, doc: &Value) {
    for section in ["agents", "automations"] {
        for new in doc[section].as_array().into_iter().flatten() {
            for old in export[section].as_array_mut().into_iter().flatten() {
                if old["name"] != new["name"] {
                    continue;
                }
                if section == "automations" {
                    *old = new.clone();
                } else if let (Some(o), Some(n)) = (old.as_object_mut(), new.as_object()) {
                    o.extend(n.clone());
                }
            }
        }
    }
}

#[test]
fn an_automation_naming_an_uninstalled_model_is_repaired_keeping_every_other_key() {
    let mut export = export_with(CRACK);
    export["automations"][0]["model"] = json!(BROKEN);
    export["automations"][0]["source"] = json!({"kind": "schedule", "frequency": "cron"});
    let mut want = export["automations"][0].clone();
    want["model"] = json!(CRACK);
    let env = Fake::new(export);
    let r = run(&env, true).unwrap();
    let doc = json!({"version": 1, "automations": [want.clone()]});
    assert_eq!(*env.applies.borrow(), vec![doc], "{}", r.lines.join("\n"));
    assert_eq!((r.repaired, r.unresolved, r.exit_code()), (1, 0, 3));
    assert_eq!(env.export.borrow()["automations"][0], want);
}

#[test]
fn the_simulated_plan_resets_an_automation_key_the_document_leaves_out() {
    // The instrument, checked against the measured plan line before it is
    // trusted to pass the repair above.
    let export = export_with(CRACK);
    let doc = json!({"version": 1, "automations": [{
        "name": "code-simplifier:Code Simplifier", "instructions": "Simplify.", "model": CRACK}]});
    let text = simulated_plan(&export, &doc);
    assert!(text.contains("fresh_session: true -> (unset)"), "{text}");
    assert!(
        text.contains(&format!("model: (unset) -> {CRACK}")),
        "{text}"
    );
    let agent = json!({"version": 1, "agents": [{"name": "Disk Audit", "model": BROKEN}]});
    assert_eq!(
        simulated_plan(&export, &agent),
        format!("agents:\n  ~ Disk Audit\n      model: {CRACK} -> {BROKEN}\n")
    );
}
