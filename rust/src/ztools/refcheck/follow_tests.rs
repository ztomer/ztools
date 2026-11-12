//! `follow_think`: Osaurus's core model and default agent follow the think
//! slot through minimal one-section documents, each planned to exactly one key,
//! with a rollback that plans clean before anything is applied (LM.CORE).

use super::follow::follow_think;
use super::tests::{CRACK, Fake, export_with};
use serde_json::json;

const GEMMA: &str = "OsaurusAI/gemma-4-12B-it-MXFP8";

fn on_gemma() -> Fake {
    let mut env = Fake::new(export_with(CRACK));
    env.best
        .insert("think".into(), "gemma-4-12b-it-mxfp8".into());
    env
}

#[test]
fn both_keys_move_to_the_think_model_with_a_clean_rollback() {
    let env = on_gemma();
    let r = follow_think(&env, true).unwrap();
    assert_eq!(r.exit_code(), 0, "{}", r.lines.join("\n"));
    assert_eq!(
        *env.applies.borrow(),
        vec![
            json!({"version": 1, "model_defaults": {"core_model": GEMMA}}),
            json!({"version": 1, "default_agent": {"model": GEMMA}}),
        ]
    );
    assert_eq!(
        r.rollback,
        vec![
            json!({"version": 1, "model_defaults": {"core_model": CRACK}}),
            json!({"version": 1, "default_agent": {"model": CRACK}}),
        ]
    );
    let e = env.export.borrow();
    assert_eq!(e["model_defaults"]["core_model"], json!(GEMMA));
    assert_eq!(e["default_agent"]["model"], json!(GEMMA));
    assert_eq!(
        e["default_agent"]["name"],
        json!("Default"),
        "other keys kept"
    );
    // Rollbacks planned before the first apply; the re-export planned clean after.
    let plans = env.plans.borrow();
    assert_eq!(
        plans.len(),
        6,
        "rollback+forward per slot, then a re-plan per slot"
    );
    assert_eq!(plans[0], r.rollback[0]);
}

#[test]
fn a_plan_touching_another_key_refuses_and_applies_nothing() {
    let mut env = on_gemma();
    env.plan_text = Some(format!(
        "model_defaults:\n  ~ model_defaults\n      core_model: {CRACK} -> {GEMMA}\n      compaction_model: (unset) -> {GEMMA}\n"
    ));
    let r = follow_think(&env, true).unwrap();
    assert_eq!(r.exit_code(), 1);
    assert_empty!(env.applies.borrow());
}

#[test]
fn a_rollback_that_does_not_plan_clean_refuses() {
    let mut env = on_gemma();
    env.plan_code = 1;
    env.plan_text = Some("Document is invalid".into());
    let r = follow_think(&env, true).unwrap();
    assert_eq!(r.exit_code(), 1);
    assert!(
        r.lines.join("\n").contains("rollback"),
        "{}",
        r.lines.join("\n")
    );
    assert_empty!(env.applies.borrow());
}

#[test]
fn an_uninstalled_think_model_refuses() {
    let mut env = Fake::new(export_with(CRACK));
    env.best.insert("think".into(), "nope-1b".into());
    let r = follow_think(&env, true).unwrap();
    assert_eq!(r.exit_code(), 1);
    assert_empty!(env.plans.borrow());
}

#[test]
fn already_following_plans_clean_and_applies_nothing() {
    let env = Fake::new(export_with(CRACK));
    let r = follow_think(&env, true).unwrap();
    assert_eq!(r.exit_code(), 0, "{}", r.lines.join("\n"));
    assert_empty!(env.applies.borrow());
}

#[test]
fn a_dry_run_plans_but_never_applies() {
    let env = on_gemma();
    let r = follow_think(&env, false).unwrap();
    assert_eq!(r.exit_code(), 0, "{}", r.lines.join("\n"));
    assert_empty!(env.applies.borrow());
}
