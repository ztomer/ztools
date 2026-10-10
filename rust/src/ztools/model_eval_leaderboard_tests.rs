use crate::ztools::eval::leaderboard::{
    check_regression, detect_model_family, format_leaderboard, format_leaderboard_by_family,
    format_leaderboard_csv, generate_leaderboard, generate_leaderboard_filtered,
    task_matches_category, task_slot,
};
use crate::ztools::eval::report::HistoryEntry;
use std::collections::BTreeMap;

fn model_a_prior_runs() -> Vec<HistoryEntry> {
    vec![
        // Older clean run (single task spot check, timestamp 100.0)
        HistoryEntry {
            date: "2026-09-01".to_string(),
            timestamp: 100.0,
            task: "json".to_string(),
            score: 50,
            time: Some(1.0),
            complete: true,
            fingerprint: None,
        },
        // Previous multi-task clean run (timestamp 150.0, 5 tasks at 80% = 80.0% mean)
        HistoryEntry {
            date: "2026-09-15".to_string(),
            timestamp: 150.0,
            task: "weekend_transient".to_string(),
            score: 80,
            time: Some(2.0),
            complete: true,
            fingerprint: Some("fp0".to_string()),
        },
        HistoryEntry {
            date: "2026-09-15".to_string(),
            timestamp: 150.0,
            task: "summarize".to_string(),
            score: 80,
            time: Some(4.0),
            complete: true,
            fingerprint: Some("fp0".to_string()),
        },
        HistoryEntry {
            date: "2026-09-15".to_string(),
            timestamp: 150.0,
            task: "filename".to_string(),
            score: 80,
            time: Some(0.5),
            complete: true,
            fingerprint: Some("fp0".to_string()),
        },
        HistoryEntry {
            date: "2026-09-15".to_string(),
            timestamp: 150.0,
            task: "file_summary".to_string(),
            score: 80,
            time: Some(6.0),
            complete: true,
            fingerprint: Some("fp0".to_string()),
        },
        HistoryEntry {
            date: "2026-09-15".to_string(),
            timestamp: 150.0,
            task: "image_real".to_string(),
            score: 80,
            time: Some(3.0),
            complete: true,
            fingerprint: Some("fp0".to_string()),
        },
    ]
}

fn model_a_latest_runs() -> Vec<HistoryEntry> {
    vec![
        // Latest clean run (timestamp 200.0, 5 tasks, mean = 94.0%)
        HistoryEntry {
            date: "2026-10-09".to_string(),
            timestamp: 200.0,
            task: "weekend_transient".to_string(),
            score: 100,
            time: Some(2.0),
            complete: true,
            fingerprint: Some("fp1".to_string()),
        },
        HistoryEntry {
            date: "2026-10-09".to_string(),
            timestamp: 200.0,
            task: "summarize".to_string(),
            score: 80,
            time: Some(5.0),
            complete: true,
            fingerprint: Some("fp2".to_string()),
        },
        HistoryEntry {
            date: "2026-10-09".to_string(),
            timestamp: 200.0,
            task: "filename".to_string(),
            score: 100,
            time: Some(0.5),
            complete: true,
            fingerprint: Some("fp3".to_string()),
        },
        HistoryEntry {
            date: "2026-10-09".to_string(),
            timestamp: 200.0,
            task: "file_summary".to_string(),
            score: 90,
            time: Some(8.0),
            complete: true,
            fingerprint: Some("fp4".to_string()),
        },
        HistoryEntry {
            date: "2026-10-09".to_string(),
            timestamp: 200.0,
            task: "image_real".to_string(),
            score: 100,
            time: Some(3.0),
            complete: true,
            fingerprint: Some("fp5".to_string()),
        },
        // Newer incomplete run (should NOT be picked)
        HistoryEntry {
            date: "2026-10-10".to_string(),
            timestamp: 300.0,
            task: "weekend_transient".to_string(),
            score: 0,
            time: None,
            complete: false,
            fingerprint: None,
        },
    ]
}

fn model_a_history() -> Vec<HistoryEntry> {
    let mut out = model_a_prior_runs();
    out.extend(model_a_latest_runs());
    out
}

fn sample_history() -> BTreeMap<String, Vec<HistoryEntry>> {
    let mut history: BTreeMap<String, Vec<HistoryEntry>> = BTreeMap::new();

    // Model A: older run, previous multi-task run, clean latest run, and newer incomplete run
    history.insert("model_a".to_string(), model_a_history());

    // Model B: latest clean run (timestamp 250.0, 4 tasks, no VLM)
    history.insert(
        "model_b".to_string(),
        vec![
            HistoryEntry {
                date: "2026-10-09".to_string(),
                timestamp: 250.0,
                task: "weekend_transient".to_string(),
                score: 80,
                time: Some(1.0),
                complete: true,
                fingerprint: Some("fp1".to_string()),
            },
            HistoryEntry {
                date: "2026-10-09".to_string(),
                timestamp: 250.0,
                task: "summarize".to_string(),
                score: 70,
                time: Some(2.0),
                complete: true,
                fingerprint: Some("fp2".to_string()),
            },
            HistoryEntry {
                date: "2026-10-09".to_string(),
                timestamp: 250.0,
                task: "filename".to_string(),
                score: 90,
                time: Some(0.5),
                complete: true,
                fingerprint: Some("fp3".to_string()),
            },
            HistoryEntry {
                date: "2026-10-09".to_string(),
                timestamp: 250.0,
                task: "file_summary".to_string(),
                score: 60,
                time: Some(3.0),
                complete: true,
                fingerprint: Some("fp4".to_string()),
            },
        ],
    );

    // Model C: only incomplete runs (should be excluded)
    history.insert(
        "model_c".to_string(),
        vec![HistoryEntry {
            date: "2026-10-09".to_string(),
            timestamp: 210.0,
            task: "json".to_string(),
            score: 0,
            time: None,
            complete: false,
            fingerprint: None,
        }],
    );

    history
}

pub fn verify_leaderboard_ranking() {
    let history = sample_history();
    let rows = generate_leaderboard(&history, None, None).unwrap();
    assert_eq!(rows.len(), 2, "only models with complete runs are ranked");

    // Rank 1: Model A (mean = (100+80+100+90+100)/5 = 94.0%, delta = +14.0%)
    let first = &rows[0];
    assert_eq!(first.model, "model_a");
    assert!((first.overall_mean - 94.0).abs() < 0.01);
    assert_eq!(first.delta, Some(14.0));
    assert_eq!(first.task_count, 5);
    assert_eq!(first.json_score, Some(100.0));
    assert_eq!(first.summarize_score, Some(80.0));
    assert_eq!(first.filename_score, Some(100.0));
    assert_eq!(first.think_score, Some(90.0));
    assert_eq!(first.vlm_score, Some(100.0));

    // Rank 2: Model B (mean = (80+70+90+60)/4 = 75.0%, delta = None)
    let second = &rows[1];
    assert_eq!(second.model, "model_b");
    assert!((second.overall_mean - 75.0).abs() < 0.01);
    assert_eq!(second.delta, None);
    assert_eq!(second.task_count, 4);
    assert_eq!(second.json_score, Some(80.0));
    assert_eq!(second.summarize_score, Some(70.0));
    assert_eq!(second.filename_score, Some(90.0));
    assert_eq!(second.think_score, Some(60.0));
    assert_eq!(second.vlm_score, None, "missing slot renders None");

    // Format table
    let table = format_leaderboard(&rows);
    assert!(table.contains(
        "| 1 | `model_a` | 94.0% | +14.0% | 90.0% | 100.0% | 80.0% | 100.0% | 100.0% | 5 | 2026-10-09 |"
    ));
    assert!(table.contains(
        "| 2 | `model_b` | 75.0% | — | 60.0% | 80.0% | 70.0% | 90.0% | — | 4 | 2026-10-09 |"
    ));
}

pub fn verify_leaderboard_slot_sorting() {
    let mut history = sample_history();
    history.insert(
        "model_f".to_string(),
        vec![
            HistoryEntry {
                date: "2026-10-09".to_string(),
                timestamp: 260.0,
                task: "file_summary".to_string(),
                score: 98,
                time: Some(2.0),
                complete: true,
                fingerprint: None,
            },
            HistoryEntry {
                date: "2026-10-09".to_string(),
                timestamp: 260.0,
                task: "taxes_anomalies".to_string(),
                score: 98,
                time: Some(2.0),
                complete: true,
                fingerprint: None,
            },
            HistoryEntry {
                date: "2026-10-09".to_string(),
                timestamp: 260.0,
                task: "weekend_transient".to_string(),
                score: 40,
                time: Some(1.0),
                complete: true,
                fingerprint: None,
            },
            HistoryEntry {
                date: "2026-10-09".to_string(),
                timestamp: 260.0,
                task: "summarize".to_string(),
                score: 40,
                time: Some(1.0),
                complete: true,
                fingerprint: None,
            },
            HistoryEntry {
                date: "2026-10-09".to_string(),
                timestamp: 260.0,
                task: "filename".to_string(),
                score: 49,
                time: Some(1.0),
                complete: true,
                fingerprint: None,
            },
        ],
    );

    let overall_rows = generate_leaderboard(&history, None, Some("overall")).unwrap();
    assert_eq!(overall_rows[0].model, "model_a");
    assert_eq!(overall_rows[1].model, "model_f");
    assert_eq!(overall_rows[2].model, "model_b");

    let think_rows = generate_leaderboard(&history, None, Some("think")).unwrap();
    assert_eq!(think_rows[0].model, "model_f");
    assert_eq!(think_rows[1].model, "model_a");
    assert_eq!(think_rows[2].model, "model_b");

    let vlm_rows = generate_leaderboard(&history, None, Some("vlm")).unwrap();
    assert_eq!(vlm_rows[0].model, "model_a");
    assert_eq!(vlm_rows[1].model, "model_f");

    assert!(generate_leaderboard(&history, None, Some("unsupported")).is_err());
}

pub fn verify_leaderboard_deltas() {
    let mut history = BTreeMap::new();
    history.insert(
        "model_regress".to_string(),
        vec![
            HistoryEntry {
                date: "2026-09-01".to_string(),
                timestamp: 100.0,
                task: "json".to_string(),
                score: 90,
                time: Some(1.0),
                complete: true,
                fingerprint: None,
            },
            HistoryEntry {
                date: "2026-10-01".to_string(),
                timestamp: 200.0,
                task: "json".to_string(),
                score: 80,
                time: Some(1.0),
                complete: true,
                fingerprint: None,
            },
        ],
    );
    let rows = generate_leaderboard(&history, Some(1), None).unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].delta, Some(-10.0));
    let table = format_leaderboard(&rows);
    assert!(
        table.contains("-10.0%"),
        "negative delta formatted as -X.X%: {table}"
    );
}

#[test]
fn test_task_slot_mapping() {
    assert_eq!(task_slot("weekend_transient"), Some("json"));
    assert_eq!(task_slot("detailed_json"), Some("json"));
    assert_eq!(task_slot("summarize_mixed"), Some("summarize"));
    assert_eq!(task_slot("filename_leak"), Some("filename"));
    assert_eq!(task_slot("file_summary_mixed"), Some("think"));
    assert_eq!(task_slot("taxes_anomalies"), Some("think"));
    assert_eq!(task_slot("image_real"), Some("vlm"));
    assert_eq!(task_slot("nonexistent_unknown"), None);
}

#[test]
fn test_empty_leaderboard() {
    let history = BTreeMap::new();
    let rows = generate_leaderboard(&history, None, None).unwrap();
    assert_eq!(rows.len(), 0);
    let table = format_leaderboard(&rows);
    assert!(table.contains("No complete model eval runs found in history"));
}

#[test]
fn test_min_tasks_filtering() {
    let history = sample_history();

    let default_rows = generate_leaderboard(&history, None, None).unwrap();
    assert_eq!(default_rows.len(), 2);

    let min5_rows = generate_leaderboard(&history, Some(5), None).unwrap();
    assert_eq!(min5_rows.len(), 1);
    assert_eq!(min5_rows[0].model, "model_a");
    assert_eq!(min5_rows[0].task_count, 5);

    let min6_rows = generate_leaderboard(&history, Some(6), None).unwrap();
    assert_eq!(min6_rows.len(), 0);
}

#[test]
fn test_leaderboard_json_serialization() {
    let history = sample_history();
    let rows = generate_leaderboard(&history, None, None).unwrap();
    let json_str = serde_json::to_string_pretty(&rows).unwrap();
    let parsed: Vec<serde_json::Value> = serde_json::from_str(&json_str).unwrap();
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0]["model"], "model_a");
    assert_eq!(parsed[0]["task_count"], 5);
    assert_eq!(parsed[0]["json_score"], 100.0);
    assert_eq!(parsed[0]["delta"], 14.0);
    assert_eq!(parsed[1]["model"], "model_b");
    assert_eq!(parsed[1]["task_count"], 4);
    assert!(parsed[1]["delta"].is_null());
    assert!(parsed[1]["vlm_score"].is_null());
}

#[test]
fn test_category_filtering_m13() {
    assert!(task_matches_category("weekend_transient", "weekend"));
    assert!(task_matches_category("summarize", "twitter"));
    assert!(task_matches_category("taxes_qa", "taxes"));
    assert!(!task_matches_category("filename", "taxes"));

    let history = sample_history();
    let rows_weekend =
        generate_leaderboard_filtered(&history, None, None, Some("weekend")).unwrap();
    assert_eq!(rows_weekend.len(), 2);
    assert_eq!(rows_weekend[0].task_count, 1);
    assert_eq!(rows_weekend[0].overall_mean, 100.0);

    let rows_twitter =
        generate_leaderboard_filtered(&history, None, None, Some("twitter")).unwrap();
    assert_eq!(rows_twitter.len(), 2);
}

#[test]
fn test_csv_output_m14() {
    let history = sample_history();
    let rows = generate_leaderboard(&history, None, None).unwrap();
    let csv = format_leaderboard_csv(&rows);
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(
        lines[0],
        "model,mean,delta,think,json,summarize,filename,vlm,tasks,date"
    );
    assert!(lines[1].starts_with("model_a,94.0,14.0,90.0,100.0,80.0,100.0,100.0,5,2026-10-09"));
}

#[test]
fn test_family_grouping_m15() {
    assert_eq!(detect_model_family("qwen3.8-27b-jang"), "qwen");
    assert_eq!(detect_model_family("gemma-4-26b"), "gemma");
    assert_eq!(detect_model_family("raptor-v0.5-8b"), "raptor");
    assert_eq!(detect_model_family("muse-glimmer-30b"), "muse");
    assert_eq!(detect_model_family("custom-model"), "other");

    let history = sample_history();
    let rows = generate_leaderboard(&history, None, None).unwrap();
    let grouped = format_leaderboard_by_family(&rows);
    assert!(grouped.contains("# Model Evaluation Leaderboard (Grouped by Family)"));
    assert!(grouped.contains("## Family: other"));
}

#[test]
fn test_fail_on_regression_m16() {
    let mut history = sample_history();
    let rows = generate_leaderboard(&history, None, None).unwrap();
    assert_eq!(check_regression(&rows, 5.0), None);

    let runs_b = history.get_mut("model_b").unwrap();
    runs_b[0].score = 100;
    runs_b.push(HistoryEntry {
        date: "2026-10-10".to_string(),
        timestamp: 300.0,
        task: "weekend_transient".to_string(),
        score: 70,
        time: Some(1.0),
        complete: true,
        fingerprint: None,
    });
    let rows_reg = generate_leaderboard(&history, None, None).unwrap();
    let reg = check_regression(&rows_reg, 5.0);
    assert!(reg.is_some());
    let (model, delta) = reg.unwrap();
    assert_eq!(model, "model_b");
    assert!(delta < -5.0);
}
