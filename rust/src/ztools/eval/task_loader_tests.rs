use super::*;

#[test]
fn test_run_check_primitives() {
    assert!(run_check(
        &Check::Contains("hello".to_string()),
        "hello world",
        None
    ));
    assert!(!run_check(
        &Check::Contains("goodbye".to_string()),
        "hello world",
        None
    ));

    assert!(run_check(
        &Check::ContainsLower("HELLO".to_string()),
        "hello world",
        None
    ));
    assert!(run_check(
        &Check::ContainsAny(vec!["cat".to_string(), "dog".to_string()]),
        "my dog is cute",
        None
    ));
    assert!(!run_check(
        &Check::ContainsAny(vec!["cat".to_string(), "bird".to_string()]),
        "my dog is cute",
        None
    ));

    assert!(run_check(
        &Check::NotContains("bad".to_string()),
        "good message",
        None
    ));
    assert!(!run_check(
        &Check::NotContains("good".to_string()),
        "good message",
        None
    ));

    assert!(run_check(
        &Check::NotContainsLower("BAD".to_string()),
        "good message",
        None
    ));
}

#[test]
fn test_run_check_json_and_summary() {
    let val: serde_json::Value = serde_json::json!({
        "items": [1, 2, 3],
        "details": {"title": "doc"}
    });
    assert!(run_check(
        &Check::JsonArrayLen("items".to_string(), 3),
        "",
        Some(&val)
    ));
    assert!(!run_check(
        &Check::JsonArrayLen("items".to_string(), 2),
        "",
        Some(&val)
    ));
    assert!(run_check(
        &Check::JsonKeyExists("details".to_string()),
        "",
        Some(&val)
    ));
    assert!(!run_check(
        &Check::JsonKeyExists("missing".to_string()),
        "",
        Some(&val)
    ));

    // FileSummary check
    let good_summary = r#"[{"path": "lib/parse.py", "desc": "parses incoming data stream and validates headers"}]"#;
    assert!(run_check(&Check::FileSummary(40), good_summary, None));
}

#[test]
fn test_run_check_taxes_grounding_and_sections() {
    let check = Check::TaxesGrounding {
        expected_signals: vec!["T1135".to_string(), "Box 38".to_string(), "RSU".to_string()],
        gt_forbidden: vec!["Filed (GT)".to_string()],
        min_hits: 2,
    };
    let text = "Here is the summary regarding t1135 and rsu income.";
    assert!(run_check(&check, text, None));

    let leaked = "Here is the summary with Filed (GT) and t1135 and rsu.";
    assert!(!run_check(&check, leaked, None));

    let insufficient = "Only t1135 mentioned.";
    assert!(!run_check(&check, insufficient, None));

    let section_check =
        Check::SectionHeaders(vec!["**1. Missing".to_string(), "**2. Impact".to_string()]);
    assert!(run_check(
        &section_check,
        "**1. Missing docs\n**2. Impact is $500",
        None
    ));
    assert!(!run_check(&section_check, "**1. Missing docs only", None));

    // Advanced validator checks
    let inject_check = Check::ResistsInjection {
        markers: vec!["PWNED".to_string()],
        keywords: vec!["receipt".to_string()],
    };
    assert!(run_check(&inject_check, "store_receipt.png", None));
    assert!(!run_check(&inject_check, "PWNED_BY_ATTACKER.png", None));
}

#[test]
fn test_load_taxes_tasks_from_real_data_dir() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let taxes_dir = repo_root.join("eval_tasks").join("data").join("taxes");
    if taxes_dir.is_dir() {
        let tasks = load_taxes_tasks_from_dir(&taxes_dir).unwrap();
        assert_eq!(
            tasks.len(),
            6,
            "expected 6 taxes tasks, got {}",
            tasks.len()
        );
        let names: Vec<_> = tasks.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"taxes_anomalies"));
        assert!(names.contains(&"taxes_audit_readiness"));
        assert!(names.contains(&"taxes_synthesis"));
        assert!(names.contains(&"taxes_qa"));
        assert!(names.contains(&"taxes_slip_qa"));
        assert!(names.contains(&"taxes_yoy_narrative"));
    }
}

#[test]
fn test_load_taxes_tasks_poisoned_file_fails() {
    let temp = tempfile::tempdir().unwrap();
    let bad_file = temp.path().join("taxes_bad.json");
    fs::write(&bad_file, b"{ invalid json").unwrap();

    let result = load_taxes_tasks_from_dir(temp.path());
    assert!(result.is_err(), "expected error on poisoned JSON file");
}

#[test]
fn load_taxes_tasks_from_missing_dir_is_empty() {
    let temp = tempfile::tempdir().unwrap();
    let missing = temp.path().join("does_not_exist");
    let tasks = load_taxes_tasks_from_dir(&missing).unwrap();
    assert!(tasks.is_empty());
}

#[test]
fn snapshot_shape_decides_messages_and_check_routing() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(
        temp.path().join("taxes_qa.json"),
        r#"{"task": "qa", "system": "You are a tax assistant.", "user": "What changed?", "grounding": {"known_amounts": []}}"#,
    )
    .unwrap();
    fs::write(
        temp.path().join("taxes_anomalies.json"),
        r#"{"task": "anomalies", "user": "List the anomalies"}"#,
    )
    .unwrap();
    fs::write(
        temp.path().join("taxes_mystery.json"),
        r#"{"task": "mystery", "system": "", "user": "Anything goes"}"#,
    )
    .unwrap();
    fs::write(
        temp.path().join("unrelated.json"),
        r#"{"task": "ignored", "user": "never loaded"}"#,
    )
    .unwrap();

    let tasks = load_taxes_tasks_from_dir(temp.path()).unwrap();
    assert_eq!(tasks.len(), 3, "non-taxes files must be filtered out");

    let qa = tasks.iter().find(|t| t.name == "taxes_qa").unwrap();
    assert_eq!(
        qa.checks,
        vec![Check::TaxesGrounded {
            task_name: "qa".to_string(),
            min_score: 90,
        }]
    );
    assert_eq!(qa.messages.len(), 2, "system prompt present");
    assert_eq!(qa.messages[0].role, "system");
    assert_eq!(qa.messages[0].content, "You are a tax assistant.");

    let anomalies = tasks.iter().find(|t| t.name == "taxes_anomalies").unwrap();
    assert_eq!(
        anomalies.checks,
        vec![Check::TaxesRubric {
            task_name: "anomalies".to_string(),
        }]
    );
    assert_eq!(anomalies.messages.len(), 1, "no system prompt");
    assert_eq!(anomalies.messages[0].role, "user");

    let mystery = tasks.iter().find(|t| t.name == "taxes_mystery").unwrap();
    assert_eq!(
        mystery.checks,
        vec![Check::ContainsLower("mystery".to_string())],
        "unknown task with neither grounding nor rubric falls back"
    );
    assert_eq!(
        mystery.messages.len(),
        1,
        "empty system string must not create a system message"
    );
}

#[test]
fn load_all_eval_tasks_without_dir_returns_smoke_only() {
    let tasks = load_all_eval_tasks(None);
    assert_eq!(tasks.len(), get_built_in_smoke_tasks().len());
    assert_eq!(tasks[0].name, "Weekend Planner (JSON Extraction)");
}

#[test]
fn load_all_eval_tasks_prefers_a_taxes_subdir() {
    let temp = tempfile::tempdir().unwrap();
    let taxes_dir = temp.path().join("taxes");
    fs::create_dir(&taxes_dir).unwrap();
    fs::write(
        taxes_dir.join("taxes_qa.json"),
        r#"{"task": "qa", "user": "What changed?", "grounding": {}}"#,
    )
    .unwrap();
    fs::write(
        temp.path().join("taxes_stray.json"),
        r#"{"task": "stray", "user": "must be ignored when taxes/ exists"}"#,
    )
    .unwrap();

    let tasks = load_all_eval_tasks(Some(temp.path()));
    assert_eq!(tasks.len(), get_built_in_smoke_tasks().len() + 1);
    assert!(
        tasks.iter().any(|t| t.name == "taxes_qa"),
        "subdir task loaded"
    );
    assert!(
        !tasks.iter().any(|t| t.name == "taxes_stray"),
        "sibling of the taxes/ dir must not be read"
    );
}

#[test]
fn load_all_eval_tasks_reads_a_flat_dir_when_no_subdir_exists() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(
        temp.path().join("taxes_yoy_narrative.json"),
        r#"{"task": "yoy_narrative", "user": "Narrate the year"}"#,
    )
    .unwrap();

    let tasks = load_all_eval_tasks(Some(temp.path()));
    assert_eq!(tasks.len(), get_built_in_smoke_tasks().len() + 1);
    assert!(tasks.iter().any(|t| t.name == "taxes_yoy_narrative"));
}

#[test]
fn graded_score_dispatches_every_graded_variant() {
    let parsed = serde_json::json!({});

    for name in ["yoy_narrative", "qa", "slip_qa"] {
        let check = Check::TaxesGrounded {
            task_name: name.to_string(),
            min_score: 0,
        };
        assert!(
            check_graded_score(&check, "answer text", Some(&parsed)).is_some(),
            "{name} is a graded grounded task"
        );
        assert!(
            check_graded_score(&check, "answer text", None).is_some(),
            "{name} grades unparsed text too"
        );
    }
    for name in ["anomalies", "audit_readiness", "synthesis"] {
        let check = Check::TaxesRubric {
            task_name: name.to_string(),
        };
        assert!(
            check_graded_score(&check, "rubric text", Some(&parsed)).is_some(),
            "{name} is a graded rubric task"
        );
        assert!(check_graded_score(&check, "rubric text", None).is_some());
    }

    assert_eq!(
        check_graded_score(
            &Check::TaxesGrounded {
                task_name: "unknown_task".to_string(),
                min_score: 0,
            },
            "",
            None
        ),
        None
    );
    assert_eq!(
        check_graded_score(
            &Check::TaxesRubric {
                task_name: "unknown_task".to_string(),
            },
            "",
            None
        ),
        None
    );
    assert_eq!(
        check_graded_score(&Check::Contains("x".to_string()), "", None),
        None,
        "boolean checks carry no grade"
    );
}

#[test]
fn run_check_validator_wrappers_pass_and_fail_on_threshold() {
    let with_items = serde_json::json!({"items": [{"name": "Alpha"}, {"name": "Beta"}]});
    assert!(run_check(&Check::JsonValidator(1), "", Some(&with_items)));
    assert!(!run_check(
        &Check::JsonValidator(101),
        "",
        Some(&with_items)
    ));
    assert!(
        !run_check(&Check::JsonValidator(1), "", Some(&serde_json::json!({}))),
        "no extractable items scores 0"
    );

    assert!(run_check(&Check::DetailedJson(1), "", Some(&with_items)));
    assert!(!run_check(&Check::DetailedJson(101), "", Some(&with_items)));
    assert!(!run_check(
        &Check::DetailedJson(1),
        "",
        Some(&serde_json::json!({}))
    ));
}

#[test]
fn run_check_no_fabrication_never_passes_against_an_empty_source() {
    let grounded = serde_json::json!({"activities": [{"name": "Alpha Park"}]});
    assert!(!run_check(
        &Check::NoFabrication { lures: vec![] },
        "Alpha Park",
        Some(&grounded)
    ));
    assert!(!run_check(
        &Check::NoFabrication {
            lures: vec!["Lure Hall".to_string()]
        },
        "",
        None
    ));
}

#[test]
fn run_check_attribution_against_no_source_always_fails() {
    assert!(!run_check(
        &Check::Attribution(50),
        "- claim text (@someone | Aug 10)",
        None
    ));
    assert!(!run_check(&Check::Attribution(1), "prose", None));
}

#[test]
fn run_check_taxes_grounded_compares_score_to_min() {
    for name in ["yoy_narrative", "qa", "slip_qa"] {
        let passing = Check::TaxesGrounded {
            task_name: name.to_string(),
            min_score: 0,
        };
        let failing = Check::TaxesGrounded {
            task_name: name.to_string(),
            min_score: 101,
        };
        assert!(run_check(&passing, "{}", Some(&serde_json::json!({}))));
        assert!(!run_check(&failing, "{}", Some(&serde_json::json!({}))));
    }
    let unknown = Check::TaxesGrounded {
        task_name: "nope".to_string(),
        min_score: 1,
    };
    assert!(
        !run_check(&unknown, "", None),
        "unknown grounded task scores 0"
    );
}

#[test]
fn run_check_taxes_rubric_rejects_unknown_names_and_hollow_output() {
    for name in ["anomalies", "audit_readiness", "synthesis"] {
        let check = Check::TaxesRubric {
            task_name: name.to_string(),
        };
        assert!(
            !run_check(&check, "", None),
            "empty output cannot reach the 50 mark for {name}"
        );
    }
    assert!(!run_check(
        &Check::TaxesRubric {
            task_name: "nope".to_string(),
        },
        "brilliant prose",
        None
    ));
}
