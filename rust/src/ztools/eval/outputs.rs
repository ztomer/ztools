//! Keep what the model actually said, not just the score it earned.
//!
//! Ported from `eval/outputs.py`. The eval records a score, a failure reason
//! and a one-line evidence string; without the raw output every question about
//! a SCORER is unanswerable without re-running the model. That cost a full day
//! once: deciding whether a failing metric was the models or the scoring
//! needed one look at what a model had written, and the only way back was
//! another sweep -- hours of GPU on a machine that runs one model at a time.
//!
//! A few KB per task removes that entire category of dead end, so this is on
//! by default on the production path (`record_signals`); prompts are fixtures,
//! so nothing saved here is user data.

use std::path::{Path, PathBuf};

/// Enough to diagnose a scorer; short of a model that emits a novel of reasoning.
pub fn max_saved_chars() -> usize {
    std::env::var("EVAL_MAX_SAVED_OUTPUT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200_000)
}

/// On unless explicitly disabled, because the failure mode is silent loss.
pub fn outputs_enabled() -> bool {
    match std::env::var("EVAL_SAVE_OUTPUTS") {
        Ok(v) => !matches!(v.as_str(), "0" | "false" | "no"),
        Err(_) => true,
    }
}

/// Where saved outputs live. Overridable so tests never touch the real one.
pub fn outputs_dir(eval_dir: Option<&Path>) -> PathBuf {
    if let Ok(override_dir) = std::env::var("EVAL_OUTPUT_DIR") {
        return PathBuf::from(override_dir);
    }
    let base = match eval_dir {
        Some(d) => d.to_path_buf(),
        None => crate::ztools::eval::report::default_eval_dir(),
    };
    base.join("outputs")
}

/// Model identifiers carry dots and slashes; keep them out of the path.
pub fn safe(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect();
    let trimmed = cleaned.trim_matches(|c| c == '.' || c == '_');
    if trimmed.is_empty() {
        "unnamed".to_string()
    } else {
        trimmed.to_string()
    }
}

/// One attempt's evidence, grouped so this signature cannot grow another
/// argument per knob.
#[derive(Debug, Clone, Copy)]
pub struct OutputRecord<'a> {
    pub model: &'a str,
    pub task: &'a str,
    pub content: &'a str,
    pub reasoning: &'a str,
    pub error: Option<&'a str>,
    pub score: u8,
    pub failure_reason: &'a str,
}

/// Write one model's raw answer for one task, with its verdict in the header.
///
/// Returns the path written, or None when saving is off or there was nothing
/// to save. Never fails the run: losing an output is bad, but ending a
/// ten-hour eval over a full disk would be worse.
pub fn save_output(record: &OutputRecord, eval_dir: Option<&Path>) -> Option<PathBuf> {
    if !outputs_enabled() {
        return None;
    }
    let model = record.model;
    let task = record.task;
    let content = record.content;
    let reasoning = record.reasoning;
    let error = record.error.unwrap_or("");
    let score = record.score;
    let failure_reason = record.failure_reason;
    if content.is_empty() && reasoning.is_empty() && error.is_empty() {
        return None;
    }

    let target = outputs_dir(eval_dir).join(safe(model));
    std::fs::create_dir_all(&target).ok()?;
    let path = target.join(format!("{}.txt", safe(task)));
    let max = max_saved_chars();
    let mut header = format!(
        "model: {model}\ntask: {task}\nscore: {score}\nfailure: {failure_reason}\nerror: {error}\nchars: {}\n",
        content.chars().count()
    );
    // Kept separate: for thinking models the visible answer is often short and
    // the reasoning is where a format failure is explained.
    if !reasoning.is_empty() {
        header.push_str(&format!("reasoning_chars: {}\n", reasoning.chars().count()));
    }
    let mut text = format!("{header}---\n{}", &content[..content.len().min(max)]);
    if !reasoning.is_empty() {
        text.push_str("\n--- reasoning ---\n");
        text.push_str(&reasoning[..reasoning.len().min(max)]);
    }
    std::fs::write(&path, text).ok()?;
    Some(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn guard(key: &str, value: Option<&std::ffi::OsStr>) -> Option<std::ffi::OsString> {
        let prev = std::env::var_os(key);
        match value {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
        prev
    }

    #[test]
    fn safe_names_strip_path_shaped_characters() {
        assert_eq!(safe("qwen3.8-27b-8bit"), "qwen3.8-27b-8bit");
        assert_eq!(safe("org/model"), "org_model");
        assert_eq!(safe("../../etc/passwd"), "etc_passwd", "leading dots strip away like Python's .strip(\"._-\")");
        assert_eq!(safe("..."), "unnamed");
        assert_eq!(safe("___"), "unnamed");
    }

    #[test]
    #[serial_test::serial]
    fn saving_is_on_by_default_and_env_can_disable_it() {
        let prev = guard("EVAL_SAVE_OUTPUTS", None);
        assert!(outputs_enabled());
        let _ = guard("EVAL_SAVE_OUTPUTS", Some(std::ffi::OsStr::new("0")));
        assert!(!outputs_enabled());
        let _ = guard("EVAL_SAVE_OUTPUTS", Some(std::ffi::OsStr::new("no")));
        assert!(!outputs_enabled());
        let _ = guard("EVAL_SAVE_OUTPUTS", Some(std::ffi::OsStr::new("1")));
        assert!(outputs_enabled());
        let _ = guard("EVAL_SAVE_OUTPUTS", prev.as_deref());
    }

    #[test]
    #[serial_test::serial]
    fn save_output_writes_header_body_and_reasoning_to_the_seamed_dir() {
        let dir = tempfile::tempdir().unwrap();
        let prev_out = guard("EVAL_OUTPUT_DIR", Some(dir.path().as_os_str()));
        let prev_en = guard("EVAL_SAVE_OUTPUTS", None);

        let path = save_output(
            &OutputRecord {
                model: "m/1",
                task: "task x",
                content: "the answer",
                reasoning: "chain of thought",
                error: None,
                score: 100,
                failure_reason: "",
            },
            None,
        );

        let _ = guard("EVAL_OUTPUT_DIR", prev_out.as_deref());
        let _ = guard("EVAL_SAVE_OUTPUTS", prev_en.as_deref());

        let path = path.expect("saved");
        assert!(path.starts_with(dir.path()));
        assert_eq!(path.parent().unwrap().file_name().unwrap(), "m_1");
        assert_eq!(path.file_name().unwrap(), "task_x.txt");
        let text = std::fs::read_to_string(path).unwrap();
        assert!(text.starts_with("model: m/1\ntask: task x\nscore: 100\n"), "{text}");
        assert!(text.contains("\nchars: 10\n"));
        assert!(text.contains("reasoning_chars: 16\n"));
        assert!(text.contains("---\nthe answer"));
        assert!(text.ends_with("--- reasoning ---\nchain of thought"));
    }

    fn rec<'a>(
        model: &'a str,
        task: &'a str,
        content: &'a str,
        reasoning: &'a str,
        error: Option<&'a str>,
        score: u8,
        failure_reason: &'a str,
    ) -> OutputRecord<'a> {
        OutputRecord {
            model,
            task,
            content,
            reasoning,
            error,
            score,
            failure_reason,
        }
    }

    impl OutputRecord<'_> {
        fn save(&self) -> Option<PathBuf> {
            save_output(self, None)
        }
    }

    #[test]
    #[serial_test::serial]
    fn nothing_to_save_and_disabled_both_return_none_without_touching_disk() {
        let dir = tempfile::tempdir().unwrap();
        let prev_out = guard("EVAL_OUTPUT_DIR", Some(dir.path().as_os_str()));

        let prev_en = guard("EVAL_SAVE_OUTPUTS", None);
        let none = rec("m", "t", "", "", None, 0, "").save();
        let _ = guard("EVAL_SAVE_OUTPUTS", prev_en.as_deref());
        assert!(none.is_none(), "empty everything -> nothing to save");

        let prev_en = guard("EVAL_SAVE_OUTPUTS", Some(std::ffi::OsStr::new("0")));
        let disabled = rec("m", "t", "content", "", None, 0, "").save();
        let _ = guard("EVAL_SAVE_OUTPUTS", prev_en.as_deref());
        assert!(disabled.is_none(), "disabled -> not saved");

        // An ERROR alone is worth keeping even with no content.
        let prev_en = guard("EVAL_SAVE_OUTPUTS", None);
        let err_only = rec("m", "t", "", "", Some("Timeout"), 0, "").save();
        let _ = guard("EVAL_SAVE_OUTPUTS", prev_en.as_deref());
        assert!(err_only.is_some(), "error-only output is evidence too");
        let _ = guard("EVAL_OUTPUT_DIR", prev_out.as_deref());
    }
}
