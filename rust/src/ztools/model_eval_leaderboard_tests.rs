//! Unit tests for multi-model leaderboard generation and ranking (`M8`).

use crate::ztools::eval::leaderboard::{format_leaderboard, generate_leaderboard, task_slot};
use crate::ztools::eval::report::HistoryEntry;
use std::collections::BTreeMap;

fn model_a_history() -> Vec<HistoryEntry> {
    vec![
        // Older clean run (should NOT be picked)
        HistoryEntry {
            date: "2026-09-01".to_string(),
            timestamp: 100.0,
            task: "json".to_string(),
            score: 50,
            time: Some(1.0),
            complete: true,
            fingerprint: None,
        },
        // Latest clean run (timestamp 200.0, 5 tasks)
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

fn sample_history() -> BTreeMap<String, Vec<HistoryEntry>> {
    let mut history: BTreeMap<String, Vec<HistoryEntry>> = BTreeMap::new();

    // Model A: older run, clean latest run, and newer incomplete run
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
    let rows = generate_leaderboard(&history, None);
    assert_eq!(rows.len(), 2, "only models with complete runs are ranked");

    // Rank 1: Model A (mean = (100+80+100+90+100)/5 = 94.0%)
    let first = &rows[0];
    assert_eq!(first.model, "model_a");
    assert!((first.overall_mean - 94.0).abs() < 0.01);
    assert_eq!(first.task_count, 5);
    assert_eq!(first.json_score, Some(100.0));
    assert_eq!(first.summarize_score, Some(80.0));
    assert_eq!(first.filename_score, Some(100.0));
    assert_eq!(first.think_score, Some(90.0));
    assert_eq!(first.vlm_score, Some(100.0));

    // Rank 2: Model B (mean = (80+70+90+60)/4 = 75.0%)
    let second = &rows[1];
    assert_eq!(second.model, "model_b");
    assert!((second.overall_mean - 75.0).abs() < 0.01);
    assert_eq!(second.task_count, 4);
    assert_eq!(second.json_score, Some(80.0));
    assert_eq!(second.summarize_score, Some(70.0));
    assert_eq!(second.filename_score, Some(90.0));
    assert_eq!(second.think_score, Some(60.0));
    assert_eq!(second.vlm_score, None, "missing slot renders None");

    // Format table
    let table = format_leaderboard(&rows);
    assert!(table.contains(
        "| 1 | `model_a` | 94.0% | 90.0% | 100.0% | 80.0% | 100.0% | 100.0% | 5 | 2026-10-09 |"
    ));
    assert!(table.contains(
        "| 2 | `model_b` | 75.0% | 60.0% | 80.0% | 70.0% | 90.0% | — | 4 | 2026-10-09 |"
    ));
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
    let rows = generate_leaderboard(&history, None);
    assert_eq!(rows.len(), 0);
    let table = format_leaderboard(&rows);
    assert!(table.contains("No complete model eval runs found in history"));
}

#[test]
fn test_min_tasks_filtering() {
    let history = sample_history();

    // Default (None): includes model_a (5 tasks) and model_b (4 tasks)
    let default_rows = generate_leaderboard(&history, None);
    assert_eq!(default_rows.len(), 2);

    // min_tasks = 5: includes model_a (5 tasks), excludes model_b (4 tasks)
    let min5_rows = generate_leaderboard(&history, Some(5));
    assert_eq!(min5_rows.len(), 1);
    assert_eq!(min5_rows[0].model, "model_a");
    assert_eq!(min5_rows[0].task_count, 5);

    // min_tasks = 6: neither model has 6 tasks
    let min6_rows = generate_leaderboard(&history, Some(6));
    assert_eq!(min6_rows.len(), 0);
}

#[test]
fn test_leaderboard_json_serialization() {
    let history = sample_history();
    let rows = generate_leaderboard(&history, None);
    let json_str = serde_json::to_string_pretty(&rows).unwrap();
    let parsed: Vec<serde_json::Value> = serde_json::from_str(&json_str).unwrap();
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0]["model"], "model_a");
    assert_eq!(parsed[0]["task_count"], 5);
    assert_eq!(parsed[0]["json_score"], 100.0);
    assert_eq!(parsed[1]["model"], "model_b");
    assert_eq!(parsed[1]["task_count"], 4);
    assert!(parsed[1]["vlm_score"].is_null());
}
