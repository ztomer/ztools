//! The Hebrew tasks, in both directions: a perfect answer built from the planted facts
//! scores 100, an empty or wrong one scores low, and the prompt really carries every fact
//! the scorer asks for (an unsatisfiable item would cap every model below 100 for nothing).

use super::*;

fn graded_of(task: &EvalTask) -> &Graded {
    let Check::Graded(g) = &task.checks[0] else {
        panic!("{}: not graded", task.name);
    };
    g
}

#[test]
fn the_three_tasks_exist_and_only_the_json_one_parses_json() {
    let tasks = hebrew_tasks();
    let names: Vec<&str> = tasks.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, ["hebrew_summarize", "hebrew_mixed", "hebrew_json"]);
    let parse: Vec<bool> = tasks.iter().map(|t| t.parse_json).collect();
    assert_eq!(parse, [false, false, true]);
}

#[test]
fn every_planted_item_and_handle_is_in_the_prompt() {
    for task in hebrew_tasks() {
        let prompt = &task.messages[0].content;
        match graded_of(&task) {
            Graded::Recall {
                items,
                handles,
                forbidden,
            } => {
                for x in items.iter().chain(handles).chain(forbidden) {
                    assert!(prompt.contains(x.as_str()), "{}: {x}", task.name);
                }
            }
            Graded::JsonFields { expected } => {
                // Values are in the prompt, except the date, which the prompt asks to be
                // rewritten as YYYY-MM-DD and so is checked by its day and year.
                for (key, value) in expected {
                    if key == "date" {
                        assert!(prompt.contains(&value[..4]), "{}: {value}", task.name);
                    } else {
                        assert!(
                            prompt.contains(value.as_str()),
                            "{}: {key}={value}",
                            task.name
                        );
                    }
                }
            }
            other => panic!("{}: unexpected {other:?}", task.name),
        }
    }
}

#[test]
fn the_summarize_task_separates_a_good_a_sloppy_and_an_empty_answer() {
    let tasks = hebrew_tasks();
    let task = &tasks[0];
    let g = graded_of(task);
    let Graded::Recall { items, handles, .. } = g else {
        unreachable!()
    };
    let perfect = handles
        .iter()
        .zip(items.chunks(2))
        .map(|(h, facts)| format!("- {} (@{h} | 08:00)\n", facts.join(" ")))
        .collect::<Vec<String>>()
        .concat();
    assert_eq!(g.score(&perfect, None).0, 100);
    let sloppy = handles
        .iter()
        .map(|h| format!("- משהו קרה לפי {h}\n"))
        .collect::<Vec<String>>()
        .concat();
    assert_eq!(g.score(&sloppy, None).0, 0);
    let half: String = items[..6].join(" ");
    let mid = g.score(&half, None).0;
    assert!(mid > 0 && mid < 100, "partial credit exists: {mid}");
    assert_eq!(g.score("", None).0, 0);
}

#[test]
fn the_mixed_task_punishes_summarizing_the_english_tweets() {
    let tasks = hebrew_tasks();
    let task = &tasks[1];
    let g = graded_of(task);
    let Graded::Recall { items, handles, .. } = g else {
        unreachable!()
    };
    let hebrew_only = handles
        .iter()
        .zip(items.chunks(2))
        .map(|(h, facts)| format!("- {} (@{h} | 08:00)\n", facts.join(" ")))
        .collect::<Vec<String>>()
        .concat();
    assert_eq!(g.score(&hebrew_only, None).0, 100);
    let with_english =
        format!("{hebrew_only}- The Hartley Bridge reopens and Vantex earned revenue\n");
    assert_eq!(g.score(&with_english, None).0, 70);
}

#[test]
fn the_json_task_scores_a_perfect_answer_100_and_a_wrong_key_zero() {
    let tasks = hebrew_tasks();
    let task = &tasks[2];
    let g = graded_of(task);
    let good = serde_json::json!({"events": [
        {"title": "הגן של סבתא רוזה", "date": "2027-03-14", "venue": "התיאטרון הקאמרי",
         "price_nis": 120, "organizer": "אסנת ברק", "duration_min": 95},
        {"title": "לילות ים תיכוניים", "date": "2027-04-02", "venue": "היכל התרבות",
         "price_nis": 85, "organizer": "יובל שמעוני", "duration_min": 140}
    ]});
    assert_eq!(g.score(&good.to_string(), Some(&good)).0, 100);
    let wrong_keys = serde_json::json!({"events": [{"name": "הגן של סבתא רוזה", "cost": 120}]});
    assert_eq!(g.score(&wrong_keys.to_string(), Some(&wrong_keys)).0, 0);
    assert_eq!(g.score("not json at all", None).0, 0);
}
