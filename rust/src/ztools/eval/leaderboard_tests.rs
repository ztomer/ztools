//! M17: a model whose latest result for a task predates the task's change is
//! not scored on it beside models measured on the task as it is now.

use super::*;

fn row(task: &str, score: i64, fingerprint: &str) -> HistoryEntry {
    HistoryEntry {
        date: "2026-10-10".to_string(),
        timestamp: 500.0,
        task: task.to_string(),
        score,
        time: Some(1.0),
        complete: true,
        fingerprint: Some(fingerprint.to_string()),
    }
}

/// The shape W5 left behind: `weekend_transient` changed fingerprint
/// (`old` → `new`); `summarize` did not. The stale model scored 100 on the old
/// weekend row, the current model 40 on the new one.
fn stale_and_current() -> (BTreeMap<String, Vec<HistoryEntry>>, TaskIdentities) {
    let mut history = BTreeMap::new();
    history.insert(
        "model_current".to_string(),
        vec![
            row("weekend_transient", 40, "new"),
            row("summarize", 80, "s"),
        ],
    );
    history.insert(
        "model_stale".to_string(),
        vec![
            row("weekend_transient", 100, "old"),
            row("summarize", 60, "s"),
        ],
    );
    let current: TaskIdentities = [("weekend_transient", "new"), ("summarize", "s")]
        .into_iter()
        .map(|(t, f)| (t.to_string(), f.to_string()))
        .collect();
    (history, current)
}

#[test]
fn test_a_stale_row_is_not_scored_beside_a_current_one() {
    let (history, current) = stale_and_current();
    let rows = generate_leaderboard(&history, &current, Some(1), Some("json")).unwrap();
    let get = |m: &str| rows.iter().find(|r| r.model == m).unwrap();

    let fresh = get("model_current");
    assert_eq!(fresh.json_score, Some(40.0));
    assert_eq!(fresh.not_current, Vec::<String>::new());

    let stale = get("model_stale");
    assert_eq!(
        stale.json_score, None,
        "a json score from the retired weekend row must not be ranked"
    );
    assert_eq!(stale.not_current, vec!["weekend_transient".to_string()]);
    assert_eq!(
        stale.summarize_score,
        Some(60.0),
        "the unchanged task still scores"
    );
    assert_eq!(
        stale.overall_mean,
        Some(60.0),
        "the mean covers only current rows"
    );
    assert_eq!(stale.task_count, 1);

    // Sorted by json, the model measured on the current task leads: before M17
    // the stale 100 outranked it.
    assert_eq!(rows[0].model, "model_current");

    let table = format_leaderboard(&rows);
    assert!(
        table.contains("| `model_stale` | 60.0% | — | — | not current | 60.0% |"),
        "{table}"
    );
    assert!(
        table.contains("not measured on the current task"),
        "{table}"
    );
    assert!(
        table.contains("- `model_stale`: weekend_transient"),
        "{table}"
    );
    assert!(format_leaderboard_csv(&rows).contains(",not current,"));
}

#[test]
fn test_a_run_with_no_current_row_has_no_mean() {
    let (mut history, current) = stale_and_current();
    history.insert(
        "model_gone".to_string(),
        vec![row("weekend_transient", 90, "old")],
    );
    let rows = generate_leaderboard(&history, &current, Some(1), None).unwrap();
    let gone = rows.iter().find(|r| r.model == "model_gone").unwrap();
    assert_eq!(gone.overall_mean, None);
    assert_eq!(gone.task_count, 0);
    assert_eq!(
        rows.last().unwrap().model,
        "model_gone",
        "unscored ranks last"
    );
    assert!(format_leaderboard(&rows).contains("| `model_gone` | not current |"));
}

#[test]
fn test_the_delta_compares_only_tasks_current_in_both_runs() {
    let current: TaskIdentities = [("weekend_transient", "new"), ("summarize", "s")]
        .into_iter()
        .map(|(t, f)| (t.to_string(), f.to_string()))
        .collect();
    let mut prev_weekend = row("weekend_transient", 100, "old");
    prev_weekend.timestamp = 100.0;
    let mut prev_summary = row("summarize", 50, "s");
    prev_summary.timestamp = 100.0;
    let mut history = BTreeMap::new();
    history.insert(
        "m".to_string(),
        vec![
            prev_weekend,
            prev_summary,
            row("weekend_transient", 20, "new"),
            row("summarize", 70, "s"),
        ],
    );
    let rows = generate_leaderboard(&history, &current, Some(1), None).unwrap();
    // summarize 50 -> 70. The weekend row changed task between the runs, so
    // 100 -> 20 is not a regression of this model; it is two questions.
    assert_eq!(rows[0].delta, Some(20.0));
}
