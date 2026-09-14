//! The built-in smoke suite.
//!
//! Five offline, fixture-only tasks that prove a model answers at all, before
//! the full roster (`super::tasks`) is spent on it. Split out of
//! `task_loader.rs` for the 500-line cap.

use super::task_loader::{Check, EvalTask};

/// Built-in smoke tasks (offline fixtures).
#[must_use]
pub fn get_built_in_smoke_tasks() -> Vec<EvalTask> {
    vec![
        EvalTask::new(
            "Weekend Planner (JSON Extraction)",
            "You are an expert family activity planner. Extract up to 10 time-limited events happening STRICTLY this weekend (between 2026-08-07 and 2026-08-09) in Vaughan from the text below.\nOutput JSON now. Use EXACT schema:\n{\"transient_events\": [{\"name\": \"str\", \"location\": \"str\", \"target_ages\": \"str\", \"price\": \"str\", \"start_date\": \"str\", \"end_date\": \"str\", \"duration\": \"str\", \"weather\": \"str\", \"day\": \"str\", \"description\": \"str\"}]}\nRules for every field:\n- Suggest up to 10 specific weekend activities. Do NOT stop after just 1 or 2 events. Find as many as you can.\n- Only extract events that occur within or overlap with the dates 2026-08-07 to 2026-08-09. Discard events from past or future weekends.\n- Copy values from the source text. NEVER invent one.\n\nSearch results:\nEvent 1: Summer Rib Fest at Vaughan Park. August 7 2026. Kids all ages. Free.\nEvent 2: Fall Fair at Markham. August 8 2026. Kids 5-10. $10.\nEvent 3: Food Truck Festival at Toronto. August 9 2026. All ages. Free.\nEvent 4: Magic Show at Vaughan Library. August 7 2026. Kids 4-8. Free.\nEvent 5: Future Festival at Vaughan Park. August 14 2026. All ages. Free.\nOutput ONLY JSON.",
            vec![
                Check::Contains("transient_events".to_string()),
                Check::Contains("Summer Rib Fest".to_string()),
                Check::Contains("Magic Show".to_string()),
                Check::JsonArrayLen("transient_events".to_string(), 2),
            ],
        ),
        EvalTask::new(
            "Twitter Summarizer (Markdown formatting)",
            "Summarize these tweets into a markdown report. Use ## headers and - bullet points.\nTweets:\n- \"New Rust version 1.75 released!\"\n- \"I had a great sandwich today.\"\n- \"Learn about lifetime elision in Rust.\"",
            vec![
                Check::Contains("##".to_string()),
                Check::ContainsAny(vec!["- ".to_string(), "* ".to_string()]),
                Check::ContainsLower("rust".to_string()),
                Check::NotContainsLower("```html".to_string()),
            ],
        ),
        EvalTask::new(
            "Image Renamer (Constraint adherence)",
            "Analyze this image description and output a snake_case filename. End with .jpg.\nDescription: A red sports car parked on a sunny beach.\nRules: Output ONLY the filename. No markdown, no conversational text.",
            vec![
                Check::Contains(".jpg".to_string()),
                Check::Contains("_".to_string()),
                Check::NotContains(" ".to_string()),
                Check::NotContainsLower("here is".to_string()),
            ],
        ),
        EvalTask::new(
            "Twitter Summarizer (Factual Consistency)",
            "Summarize this tweet timeline:\nTweet 1: @john_doe (2026-08-01): Just launched the new API!\nTweet 2: @jane_smith (2026-08-02): The new API is incredibly fast.",
            vec![
                Check::NotContains("@elonmusk".to_string()),
                Check::NotContains("@realDonaldTrump".to_string()),
                Check::ContainsAny(vec!["john_doe".to_string(), "@john_doe".to_string()]),
                Check::ContainsAny(vec!["jane_smith".to_string(), "@jane_smith".to_string()]),
                Check::ContainsAny(vec!["2026-08".to_string(), "August".to_string()]),
                Check::NotContains("2025".to_string()),
                Check::NotContains("2024".to_string()),
            ],
        ),
        EvalTask::new(
            "File Summary (Content detail)",
            "Read the file list below and give one-line summary for each file.\n\nCRITICAL: Rely ONLY on provided content context. DO NOT infer functionality from file names, words, or puns. Describe what each file DOES.\n- Bad: \"a python library\" (infers from .py extension)\n- Good: \"parses web content and extracts metadata\"\n\nFiles:\n- lib/parser.py\n- lib/validator.py\n- lib/fetcher.py\n- lib/reporter.py\n\nOutput a JSON array of {\"path\": \"...\", \"desc\": \"...\"} objects.",
            vec![Check::FileSummary(50)],
        ),
    ]
}
