//! Eval discrimination: classifies tasks as ranking or gate.

/// Minimal result record used by the discrimination module.
/// This mirrors the fields needed from the Python `references/eval/discrimination.py`:
/// `task` (task name) and `quality_score` (score 0-100).
#[derive(Debug, Clone)]
pub struct EvalResult {
    /// Task name (e.g., "json", "filename", "twitter_summarize")
    pub task: String,
    /// Quality score 0-100 for this model on this task
    pub quality_score: i32,
}

/// Minimum distinct scores across models before a task is credited with ranking.
/// Two models scoring 100 and 0 is a gate that one model failed, not a ranking:
/// it sorts models into pass and fail, which is what a gate does. Three distinct
/// values is the first count that can order anything.
const MIN_RANKING_VALUES: usize = 3;

/// How many models must have reported a task before its spread means anything.
/// Two models trivially produce at most two values, so any task looks like a gate.
const MIN_MODELS_FOR_VERDICT: usize = 4;

/// Tasks measured as gates, with the evidence. NOT a policy list -- a record of
/// what was observed, checkable by `disagreements()` against any later run.
///
/// This is a RECORD OF A MEASUREMENT, and `disagreements()` re-derives the
/// classification from any result set and reports where the record and the data
/// no longer agree. The record is the claim; the data is the check.
static GATE_TASKS: &[&str] = &["image_real", "taxes_slip_qa"];

/// Whether `task` is recorded as unable to rank (a gate).
pub fn is_gate(task: &str) -> bool {
    GATE_TASKS.contains(&task)
}

/// The subset of `tasks` that can order models.
/// Returns tasks that are NOT gates.
pub fn ranking_tasks(tasks: &[String]) -> Vec<String> {
    tasks.iter().filter(|t| !is_gate(t)).cloned().collect()
}

/// Count distinct float values by sorting and dedupieing (avoids HashSet<!f64>).
fn count_distinct(values: &[f64]) -> usize {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut count = 1usize;
    for i in 1..sorted.len() {
        if sorted[i] != sorted[i - 1] {
            count += 1;
        }
    }
    count
}

/// Every model's score for each task, from a set of per-model records.
/// Only COMPLETE runs contribute. A truncated run's absent tasks would otherwise
/// read as a narrower spread and could reclassify a ranking task as a gate.
///
/// Returns a dict mapping task name -> list of scores.
pub fn scores_by_task(all_results: &[EvalResult]) -> ::std::collections::HashMap<String, Vec<f64>> {
    let mut by_task: ::std::collections::HashMap<String, Vec<f64>> = ::std::collections::HashMap::new();
    for record in all_results {
        let task = &record.task;
        let score = record.quality_score as f64;
        by_task.entry(task.clone()).or_default().push(score);
    }
    by_task
}

/// How many different scores `task` produced across models.
pub fn distinct_values(all_results: &[EvalResult], task: &str) -> usize {
    let task_scores = scores_by_task(all_results);
    let scores = task_scores.get(task);
    scores.map(|s| count_distinct(s)).unwrap_or(0)
}

/// Derive, from data alone, which tasks rank and which gate.
///
/// Returns task -> "ranks" | "gate" | "unknown".
/// "unknown" is not a hedge: with fewer than MIN_MODELS_FOR_VERDICT models reporting,
/// a narrow spread is a property of the sample size and calling it a gate would be
/// inventing a finding.
pub fn classify(all_results: &[EvalResult]) -> ::std::collections::HashMap<String, &'static str> {
    let mut verdicts: ::std::collections::HashMap<String, &'static str> = ::std::collections::HashMap::new();
    let task_scores = scores_by_task(all_results);

    for (task, scores) in &task_scores {
        if scores.len() < MIN_MODELS_FOR_VERDICT {
            verdicts.insert(task.clone(), "unknown");
        } else {
            let distinct = count_distinct(scores.as_slice());
            if distinct >= MIN_RANKING_VALUES {
                verdicts.insert(task.clone(), "ranks");
            } else {
                verdicts.insert(task.clone(), "gate");
            }
        }
    }

    verdicts
}

/// Where the recorded classification and this run's data conflict.
///
/// Two directions:
/// - a task recorded as a GATE that now ranks -- the record is stale, and a task
///   that earned its place is being thrown away;
/// - a task counted for RANKING that now behaves as a gate -- it is diluting
///   every mean, which is the failure this module exists to stop.
///
/// Reported rather than acted on. Reclassifying a task automatically from one
/// run's data is how a single contended sweep silently rewrites what the suite
/// measures.
pub fn disagreements(all_results: &[EvalResult]) -> Vec<String> {
    let verdicts = classify(all_results);
    let mut found = Vec::new();

    for (task, verdict) in &verdicts {
        if *verdict == "unknown" {
            continue;
        }

        if *verdict == "ranks" && is_gate(task) {
            let dv = distinct_values(all_results, task);
            found.push(format!(
                "{task}: recorded as a GATE but produced {dv} distinct values here -- \
                 it may have started ranking; re-check before trusting either."
            ));
        } else if *verdict == "gate" && !is_gate(task) {
            let dv = distinct_values(all_results, task);
            found.push(format!(
                "{task}: counted for RANKING but produced only {dv} distinct values here -- \
                 it is diluting the mean without ordering anything."
            ));
        }
    }

    found
}

/// Mean over the tasks that can actually order models.
///
/// Falls back to the full mean when a run contains ONLY gate tasks -- which is
/// what `--task image_real` produces. Returning 0 there would report a model
/// that scored 100 on the one task it was asked for as having failed.
pub fn ranking_mean(all_results: &[EvalResult]) -> f64 {
    let scored_tasks: Vec<&EvalResult> = all_results.iter().filter(|r| !is_gate(&r.task)).collect();

    if !scored_tasks.is_empty() {
        let sum: f64 = scored_tasks.iter().map(|r| r.quality_score as f64).sum();
        sum / scored_tasks.len() as f64
    } else {
        // Fallback: mean over all tasks
        let all_scores: f64 = all_results.iter().map(|r| r.quality_score as f64).sum();
        let count = all_results.len() as f64;
        if count == 0.0 {
            return 0.0;
        }
        all_scores / count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_tasks() {
        assert!(is_gate("image_real"));
        assert!(is_gate("taxes_slip_qa"));
        assert!(!is_gate("json"));
        assert!(!is_gate("filename"));
    }

    #[test]
    fn test_ranking_tasks() {
        let tasks = vec![
            "image_real".to_string(),
            "taxes_slip_qa".to_string(),
            "json".to_string(),
            "filename".to_string(),
        ];
        let ranked = ranking_tasks(&tasks);
        assert_eq!(ranked.len(), 2);
        assert!(ranked.contains(&"json".to_string()));
        assert!(ranked.contains(&"filename".to_string()));
        assert!(!ranked.contains(&"image_real".to_string()));
        assert!(!ranked.contains(&"taxes_slip_qa".to_string()));
    }

    #[test]
    fn test_classify_with_3_models_unknown() {
        // 3 models < MIN_MODELS_FOR_VERDICT=4, so should be "unknown"
        let results = vec![
            EvalResult {
                task: "json".to_string(),
                quality_score: 100,
            },
            EvalResult {
                task: "json".to_string(),
                quality_score: 80,
            },
            EvalResult {
                task: "json".to_string(),
                quality_score: 60,
            },
        ];
        let verdicts = classify(&results);
        assert_eq!(verdicts.get("json"), Some(&"unknown"));
    }

    #[test]
    fn test_classify_with_5_models_ranks() {
        // 5 models >= MIN_MODELS_FOR_VERDICT=4, 3 distinct values >= MIN_RANKING_VALUES=3, so "ranks"
        let results = vec![
            EvalResult {
                task: "json".to_string(),
                quality_score: 100,
            },
            EvalResult {
                task: "json".to_string(),
                quality_score: 80,
            },
            EvalResult {
                task: "json".to_string(),
                quality_score: 60,
            },
            EvalResult {
                task: "json".to_string(),
                quality_score: 90,
            },
            EvalResult {
                task: "json".to_string(),
                quality_score: 70,
            },
        ];
        let verdicts = classify(&results);
        assert_eq!(verdicts.get("json"), Some(&"ranks"));
    }

    #[test]
    fn test_classify_with_few_models() {
        let results = vec![
            EvalResult {
                task: "json".to_string(),
                quality_score: 100,
            },
        ];
        let verdicts = classify(&results);
        assert_eq!(verdicts.get("json"), Some(&"unknown"));
    }

    #[test]
    fn test_distinct_values() {
        let results = vec![
            EvalResult {
                task: "json".to_string(),
                quality_score: 100,
            },
            EvalResult {
                task: "json".to_string(),
                quality_score: 100,
            },
        ];
        assert_eq!(distinct_values(&results, "json"), 1);
    }

    #[test]
    fn test_disagreements_no_conflict() {
        let results = vec![
            EvalResult {
                task: "json".to_string(),
                quality_score: 100,
            },
            EvalResult {
                task: "filename".to_string(),
                quality_score: 80,
            },
        ];
        let conflicts = disagreements(&results);
        assert_eq!(conflicts.len(), 0);
    }

    #[test]
    fn test_ranking_mean() {
        let results = vec![
            EvalResult {
                task: "json".to_string(),
                quality_score: 100,
            },
            EvalResult {
                task: "filename".to_string(),
                quality_score: 80,
            },
            EvalResult {
                task: "image_real".to_string(),
                quality_score: 50,
            },
        ];
        let mean = ranking_mean(&results);
        assert_eq!(mean, 90.0);
    }

    #[test]
    fn test_ranking_mean_fallback() {
        let results = vec![
            EvalResult {
                task: "image_real".to_string(),
                quality_score: 100,
            },
            EvalResult {
                task: "taxes_slip_qa".to_string(),
                quality_score: 80,
            },
        ];
        let mean = ranking_mean(&results);
        assert_eq!(mean, 90.0);
    }
}
