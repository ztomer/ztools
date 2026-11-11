//! Refcheck on fixtures: a broken export must fail, a clean one must pass, and
//! a repair must be exactly one agent's model.

use super::*;
use serde_json::json;
use std::cell::RefCell;

pub const BROKEN: &str = "mlx-community/Qwen3.8-27B-8bit";
pub const CRACK: &str = "dealignai/Qwen3.8-27B-JANG_6D-CRACK";
const GEMMA: &str = "OsaurusAI/gemma-4-12B-it-MXFP8";

/// The shape of a real `osaurus config export`, cut to what refcheck reads.
pub fn export_with(disk_audit: &str) -> Value {
    json!({
        "version": 1,
        "models": [
            "OsaurusAI/Raptor-v0.5-8B-A1B-JANG_6M", GEMMA, CRACK,
            "mlx-community/LFM2.5-350M-OptiQ-4bit"
        ],
        "model_defaults": {"core_model": CRACK, "compaction_model": null},
        "default_agent": {"name": "Default", "model": CRACK},
        "configurator": {"model": null},
        "agents": [
            {"name": "Code Reader", "model": CRACK},
            {"name": "Consult Kit", "model": null},
            {"name": "Disk Audit", "model": disk_audit}
        ],
        "automations": [{
            "name": "code-simplifier:Code Simplifier", "enabled": false,
            "fresh_session": true, "instructions": "Simplify.", "model": null
        }],
        "providers": [
            {"name": "Google", "provider": "google", "enabled": false, "manual_model_ids": []},
            {"name": "Anthropic", "provider": "anthropic", "enabled": true, "manual_model_ids": []}
        ]
    })
}

pub struct Fake {
    pub export: RefCell<Value>,
    pub best: BTreeMap<String, String>,
    pub listed: Vec<String>,
    pub disk: Vec<String>,
    pub bases: BTreeMap<String, String>,
    pub scores: ThinkScores,
    /// When set, `plan` prints this whatever the document is.
    pub plan_text: Option<String>,
    pub plan_code: i32,
    pub plans: RefCell<Vec<Value>>,
    pub applies: RefCell<Vec<Value>>,
}

impl Fake {
    pub fn new(export: Value) -> Self {
        let disk = vec![
            "OsaurusAI/Raptor-v0.5-8B-A1B-JANG_6M".into(),
            GEMMA.into(),
            CRACK.into(),
            "mlx-community/LFM2.5-350M-OptiQ-4bit".into(),
        ];
        Self {
            export: RefCell::new(export),
            best: BTreeMap::from([("think".into(), "qwen3.8-27b-jang_6d-crack".into())]),
            listed: vec!["foundation".into(), "qwen3.8-27b-jang_6d-crack".into()],
            disk,
            bases: BTreeMap::from([
                (BROKEN.into(), "Qwen/Qwen3.8-27B".into()),
                (CRACK.into(), "Qwen/Qwen3.8-27B".into()),
            ]),
            scores: ThinkScores {
                scores: BTreeMap::from([("qwen3.8-27b-jang_6d-crack".into(), 82.5)]),
                stale: Vec::new(),
            },
            plan_text: None,
            plan_code: 0,
            plans: RefCell::default(),
            applies: RefCell::default(),
        }
    }

    fn agent_model(&self, name: &str) -> String {
        let e = self.export.borrow();
        e["agents"]
            .as_array()
            .and_then(|a| a.iter().find(|x| x["name"] == name))
            .and_then(|a| a["model"].as_str())
            .unwrap_or("(unset)")
            .to_string()
    }
}

impl Env for Fake {
    fn export(&self) -> Result<Value> {
        Ok(self.export.borrow().clone())
    }
    fn osaurus_list(&self) -> Result<Vec<String>> {
        Ok(self.listed.clone())
    }
    fn disk_models(&self) -> Vec<String> {
        self.disk.clone()
    }
    fn best_models(&self) -> BTreeMap<String, String> {
        self.best.clone()
    }
    fn base_model(&self, repo: &str) -> Option<String> {
        self.bases.get(repo).cloned()
    }
    fn think_scores(&self) -> ThinkScores {
        self.scores.clone()
    }
    fn plan(&self, doc: &Value) -> Result<CmdOut> {
        self.plans.borrow_mut().push(doc.clone());
        let text = self
            .plan_text
            .clone()
            .unwrap_or_else(|| super::tests_automation::simulated_plan(&self.export.borrow(), doc));
        Ok(CmdOut {
            code: self.plan_code,
            text,
        })
    }
    fn apply(&self, doc: &Value) -> Result<CmdOut> {
        self.applies.borrow_mut().push(doc.clone());
        super::tests_automation::simulated_apply(&mut self.export.borrow_mut(), doc);
        Ok(CmdOut {
            code: 0,
            text: "applied".into(),
        })
    }
}

fn text(r: &Report) -> String {
    r.lines.join("\n")
}

#[test]
fn a_clean_export_passes() {
    let env = Fake::new(export_with(CRACK));
    let r = run(&env, true).unwrap();
    assert_eq!(
        (r.errors, r.unresolved, r.exit_code()),
        (0, 0, 0),
        "{}",
        text(&r)
    );
    assert_empty!(env.plans.borrow());
}

#[test]
fn an_export_pinning_the_uninstalled_8bit_fails_naming_the_exact_path() {
    let env = Fake::new(export_with(BROKEN));
    let r = run(&env, false).unwrap();
    assert_eq!((r.errors, r.unresolved, r.exit_code()), (1, 1, 1));
    assert!(
        text(&r)
            .contains("agents[Disk Audit].model = mlx-community/Qwen3.8-27B-8bit: not installed"),
        "{}",
        text(&r)
    );
    assert_empty!(env.plans.borrow());
}

#[test]
fn every_reference_kind_is_checked() {
    let paths = [
        ("model_defaults", "core_model", "model_defaults.core_model"),
        (
            "model_defaults",
            "compaction_model",
            "model_defaults.compaction_model",
        ),
        ("default_agent", "model", "default_agent.model"),
        ("configurator", "model", "configurator.model"),
    ];
    for (section, key, path) in paths {
        let mut export = export_with(CRACK);
        export[section][key] = json!(BROKEN);
        let r = run(&Fake::new(export), false).unwrap();
        assert!(
            text(&r).contains(&format!("{path} = {BROKEN}: not installed")),
            "{path}"
        );
    }
    let mut export = export_with(CRACK);
    export["automations"][0]["model"] = json!(BROKEN);
    let r = run(&Fake::new(export), false).unwrap();
    assert!(text(&r).contains(&format!(
        "automations[code-simplifier:Code Simplifier].model = {BROKEN}: not installed"
    )));
}

#[test]
fn null_foundation_and_an_enabled_remote_model_are_valid() {
    for ok in ["foundation", "anthropic/claude-sonnet-5"] {
        let r = run(&Fake::new(export_with(ok)), false).unwrap();
        assert_eq!(r.errors, 0, "{ok}: {}", text(&r));
    }
    // A disabled provider offers nothing.
    let r = run(&Fake::new(export_with("google/gemini-x")), false).unwrap();
    assert_eq!(r.errors, 1);
}

#[test]
fn a_local_id_is_matched_case_sensitively() {
    let r = run(
        &Fake::new(export_with("dealignai/qwen3.8-27b-jang_6d-crack")),
        false,
    )
    .unwrap();
    assert_eq!(r.errors, 1, "{}", text(&r));
}

#[test]
fn a_best_models_entry_naming_an_uninstalled_model_is_an_error() {
    let mut env = Fake::new(export_with(CRACK));
    env.best
        .insert("summarize".into(), "qwen3.8-27b-8bit".into());
    let r = run(&env, true).unwrap();
    assert_eq!((r.errors, r.unresolved, r.exit_code()), (1, 1, 1));
    assert!(text(&r).contains("best_models.summarize = qwen3.8-27b-8bit: not installed"));
    assert_empty!(env.plans.borrow());
}

#[test]
fn repair_on_the_broken_fixture_is_a_one_agent_plan_to_the_6d_crack() {
    let env = Fake::new(export_with(BROKEN));
    let r = run(&env, true).unwrap();
    let want = json!({"version": 1, "agents": [{"name": "Disk Audit", "model": CRACK}]});
    assert_eq!(*env.plans.borrow(), vec![want.clone()]);
    assert_eq!(*env.applies.borrow(), vec![want]);
    assert_eq!(
        (r.errors, r.repaired, r.unresolved, r.exit_code()),
        (1, 1, 0, 3),
        "{}",
        text(&r)
    );
    assert!(text(&r).contains(&format!("agents[Disk Audit].model: {BROKEN} -> {CRACK}")));
    assert_eq!(env.agent_model("Disk Audit"), CRACK);
}

#[test]
fn a_plan_that_touches_anything_else_aborts_the_repair() {
    let mut env = Fake::new(export_with(BROKEN));
    env.plan_text = Some(format!(
        "agents:\n  ~ Disk Audit\n      model: {BROKEN} -> {CRACK}\n  ~ Code Reader\n      model: x -> y\n"
    ));
    let r = run(&env, true).unwrap();
    assert_empty!(env.applies.borrow());
    assert_eq!((r.repaired, r.unresolved, r.exit_code()), (0, 1, 1));
    assert!(text(&r).contains("touches more than"), "{}", text(&r));
    assert_eq!(env.agent_model("Disk Audit"), BROKEN);
}

#[test]
fn a_failing_plan_aborts_the_repair() {
    let mut env = Fake::new(export_with(BROKEN));
    env.plan_code = 1;
    env.plan_text = Some("Document is invalid".into());
    let r = run(&env, true).unwrap();
    assert_empty!(env.applies.borrow());
    assert_eq!(r.unresolved, 1);
    assert!(text(&r).contains("plan exited 1"));
}

#[test]
fn an_unconfirmed_repair_stays_an_error() {
    // The apply "succeeds" but the re-export still names the missing model.
    struct Stuck(Fake);
    impl Env for Stuck {
        fn export(&self) -> Result<Value> {
            self.0.export()
        }
        fn osaurus_list(&self) -> Result<Vec<String>> {
            self.0.osaurus_list()
        }
        fn disk_models(&self) -> Vec<String> {
            self.0.disk_models()
        }
        fn best_models(&self) -> BTreeMap<String, String> {
            self.0.best_models()
        }
        fn base_model(&self, r: &str) -> Option<String> {
            self.0.base_model(r)
        }
        fn think_scores(&self) -> ThinkScores {
            self.0.think_scores()
        }
        fn plan(&self, d: &Value) -> Result<CmdOut> {
            self.0.plan(d)
        }
        fn apply(&self, _: &Value) -> Result<CmdOut> {
            Ok(CmdOut {
                code: 0,
                text: String::new(),
            })
        }
    }
    let r = run(&Stuck(Fake::new(export_with(BROKEN))), true).unwrap();
    assert_eq!(
        (r.repaired, r.unresolved, r.exit_code()),
        (0, 1, 1),
        "{}",
        text(&r)
    );
}

#[test]
fn with_no_family_member_installed_it_goes_to_core_model_and_stays_red() {
    let mut env = Fake::new(export_with("mlx-community/Gemma-9-Q8"));
    env.bases.clear();
    let r = run(&env, true).unwrap();
    let doc = &env.applies.borrow()[0];
    assert_eq!(doc["agents"][0]["model"], CRACK);
    assert_eq!(
        (r.repaired, r.unresolved, r.exit_code()),
        (0, 1, 1),
        "{}",
        text(&r)
    );
    assert!(text(&r).contains("no installed member of the family"));
}

#[test]
fn a_missing_core_model_has_nowhere_to_fall_back_to() {
    let mut export = export_with(CRACK);
    export["model_defaults"]["core_model"] = json!("mlx-community/Nope-1B-4bit");
    let env = Fake::new(export);
    let r = run(&env, true).unwrap();
    assert_empty!(env.plans.borrow());
    assert_eq!((r.unresolved, r.exit_code()), (1, 1));
}

#[test]
fn a_name_stem_finds_the_family_when_no_base_model_is_cached() {
    let mut env = Fake::new(export_with(BROKEN));
    env.bases.clear();
    let r = run(&env, true).unwrap();
    assert_eq!(env.applies.borrow()[0]["agents"][0]["model"], CRACK);
    assert_eq!(r.repaired, 1, "{}", text(&r));
}

#[test]
fn the_best_think_score_wins_and_a_cracked_build_wins_a_tie() {
    let installed: Vec<String> = [
        "mlx-community/Qwen3.8-27B-4bit",
        "dealignai/Qwen3.8-27B-JANG_6D-CRACK",
        "dealignai/Qwen3.8-27B-JANG_6D",
        "OsaurusAI/gemma-4-12B-it-MXFP8",
    ]
    .map(String::from)
    .to_vec();
    let none = |_: &str| None;
    let score = |pairs: &[(&str, f64)]| -> BTreeMap<String, f64> {
        pairs.iter().map(|(k, v)| ((*k).to_string(), *v)).collect()
    };
    let pick = |s: &BTreeMap<String, f64>| choose(BROKEN, &installed, &none, s);
    assert_eq!(
        pick(&score(&[
            ("qwen3.8-27b-4bit", 90.0),
            ("qwen3.8-27b-jang_6d-crack", 82.5),
            ("qwen3.8-27b-jang_6d", 85.0),
        ]))
        .as_deref(),
        Some("mlx-community/Qwen3.8-27B-4bit")
    );
    assert_eq!(
        pick(&score(&[
            ("qwen3.8-27b-jang_6d-crack", 85.0),
            ("qwen3.8-27b-jang_6d", 85.0),
        ]))
        .as_deref(),
        Some(CRACK)
    );
    // A scored model beats an unscored one; a gemma is never in the family.
    assert_eq!(
        pick(&score(&[("qwen3.8-27b-jang_6d", 10.0)])).as_deref(),
        Some("dealignai/Qwen3.8-27B-JANG_6D")
    );
    assert_eq!(choose("x/Llama-3-8B", &installed, &none, &score(&[])), None);
}

#[test]
fn stems_drop_the_quantisation_and_keep_the_size() {
    for (name, want) in [
        ("mlx-community/Qwen3.8-27B-8bit", "qwen3.8-27b"),
        ("dealignai/Qwen3.8-27B-JANG_6D-CRACK", "qwen3.8-27b"),
        ("OsaurusAI/gemma-4-12B-it-MXFP8", "gemma-4-12b"),
        ("OsaurusAI/Raptor-v0.5-8B-A1B-JANG_6M", "raptor-v0.5-8b-a1b"),
        ("mlx-community/LFM2.5-350M-OptiQ-4bit", "lfm2.5-350m"),
        ("Qwen/Qwen3.8-27B", "qwen3.8-27b"),
        ("x/Mistral-7B-Q4_K_M", "mistral-7b"),
    ] {
        assert_eq!(stem(name), want, "{name}");
    }
    assert_eq!(
        family_key(Some("Qwen/Qwen3.8-27B"), "any/Thing"),
        "qwen3.8-27b"
    );
}

#[test]
fn the_minimal_document_holds_only_the_one_entity() {
    let export = export_with(BROKEN);
    let doc = |slot| plan::minimal_doc(&export, &slot, CRACK).unwrap();
    assert_eq!(
        doc(Slot::CoreModel),
        json!({"version": 1, "model_defaults": {"core_model": CRACK}})
    );
    assert_eq!(
        doc(Slot::CompactionModel),
        json!({"version": 1, "model_defaults": {"compaction_model": CRACK}})
    );
    assert_eq!(
        doc(Slot::DefaultAgent),
        json!({"version": 1, "default_agent": {"model": CRACK}})
    );
    assert_eq!(
        doc(Slot::Configurator),
        json!({"version": 1, "configurator": {"model": CRACK}})
    );
    assert_eq!(
        doc(Slot::Automation("code-simplifier:Code Simplifier".into())),
        json!({"version": 1, "automations": [{
            "name": "code-simplifier:Code Simplifier", "enabled": false,
            "fresh_session": true, "instructions": "Simplify.", "model": CRACK}]})
    );
    assert!(plan::minimal_doc(&export, &Slot::Automation("ghost".into()), CRACK).is_err());
    assert!(plan::minimal_doc(&export, &Slot::Best("think".into()), CRACK).is_err());
    assert_eq!(plan::edit_for(&Slot::Best("think".into())), None);
}

#[test]
fn an_automation_repair_whose_plan_resets_another_field_is_refused() {
    let mut export = export_with(CRACK);
    export["automations"][0]["model"] = json!(BROKEN);
    let mut env = Fake::new(export);
    env.plan_text = Some(format!(
        "automations:\n  ~ code-simplifier:Code Simplifier\n      fresh_session: true -> (unset)\n      model: {BROKEN} -> {CRACK}\n"
    ));
    let r = run(&env, true).unwrap();
    assert_empty!(env.applies.borrow());
    assert_eq!(r.unresolved, 1, "{}", text(&r));
}
