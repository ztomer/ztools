//! Context refusal: a task whose prompt certainly cannot fit the model's
//! context window is NOT MEASURED, never scored.
//!
//! `foundation` has a fixed 4096-token window covering prompt AND output
//! (`conf/models/foundation.toml`). Once `file_summary` started carrying the
//! files' contents its prompt grew to ~22.6 KB, and nothing compared a prompt
//! against a window: the request went out, could not succeed, and the task
//! landed in the table and the history as a 0 -- "this model summarises files
//! badly", when the model was never shown the files.
//!
//! WHY A LOWER BOUND, AND WHY THE PROMPT ALONE. The refusal has to fire only
//! when failure is certain, or it stops measuring tasks that work: `foundation`
//! completes `summarize` with a ~4.6 KB prompt and a 3000-token output cap,
//! which `prompt + max_tokens > window` at the transport's 3 chars/token
//! ([`super::transport::CHARS_PER_TOKEN`]) would refuse. `max_tokens` is a CAP,
//! not a reservation, so only the prompt is certain to be spent; and the token
//! count is bounded from BELOW with a generous [`MAX_CHARS_PER_TOKEN`], so an
//! estimate that errs can only err towards measuring.
//!
//! A model with no documented window is never refused: server models report no
//! window here, and guessing one would be the unforced error this module exists
//! to prevent in the other direction.

use crate::ztools::eval::model_resolve::documented_context_window;

/// More characters per token than any tokenizer averages over English or code
/// (~4 for English prose, fewer for paths and code). Dividing by it gives a
/// token count no real tokenizer comes in under.
pub const MAX_CHARS_PER_TOKEN: u64 = 5;

/// Does a prompt of `prompt_bytes` certainly overflow a `window`-token context?
#[must_use]
pub const fn certainly_overflows(prompt_bytes: u64, window: u64) -> bool {
    prompt_bytes.div_ceil(MAX_CHARS_PER_TOKEN) >= window
}

/// Why `model` must not be sent a `prompt_bytes`-byte prompt, or `None`.
#[must_use]
pub fn context_refusal(model: &str, prompt_bytes: usize) -> Option<String> {
    let window = documented_context_window(model)?;
    let bytes = u64::try_from(prompt_bytes).unwrap_or(u64::MAX);
    certainly_overflows(bytes, window).then(|| {
        format!(
            "prompt does not fit: {bytes} bytes is at least {} tokens, and {model}'s whole \
             context window is {window}",
            bytes.div_ceil(MAX_CHARS_PER_TOKEN)
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_env::TestEnv;
    use serial_test::serial;

    /// The two prompts that decided the bound, at foundation's 4096 window.
    #[test]
    fn the_file_summary_prompt_overflows_and_the_summarize_prompt_does_not() {
        assert!(certainly_overflows(22_600, 4096), "~22.6 KB file_summary");
        assert!(
            !certainly_overflows(4_617, 4096),
            "~4.6 KB summarize, which foundation completes, must stay measured"
        );
    }

    /// The edge is exact: one token's worth of bytes either side of the window.
    #[test]
    fn the_bound_is_inclusive_at_the_window() {
        let at = 4096 * MAX_CHARS_PER_TOKEN;
        assert!(certainly_overflows(at, 4096));
        assert!(certainly_overflows(at - MAX_CHARS_PER_TOKEN + 1, 4096));
        assert!(!certainly_overflows(at - MAX_CHARS_PER_TOKEN, 4096));
    }

    /// A documented window refuses and says why; an undocumented model is
    /// never refused, however large the prompt.
    #[test]
    #[serial]
    fn only_a_documented_window_refuses() {
        let env = TestEnv::new();
        let models = env.root().join("conf").join("models");
        std::fs::create_dir_all(&models).unwrap();
        std::fs::write(models.join("foundation.toml"), "context_window = 4096\n").unwrap();

        let why = context_refusal("foundation", 22_600).expect("22.6 KB cannot fit 4096");
        assert!(why.contains("4520 tokens"), "{why}");
        assert!(why.contains("4096"), "{why}");
        assert_eq!(context_refusal("foundation", 4_617), None);
        assert_eq!(context_refusal("qwen3.8-27b-jang_6d", 10_000_000), None);
        drop(env);
    }

    /// The runner never SENDS a prompt that cannot fit: the small task goes
    /// out and meets the closed port (`INFRA`), the large one is refused before
    /// any request (`CONTEXT`), and neither is a measured row.
    #[test]
    #[serial]
    fn the_runner_refuses_an_overflowing_task_without_sending_it() {
        use crate::ztools::eval::task_loader::EvalTask;
        use crate::ztools::eval::{FAIL_CONTEXT, FAIL_INFRA, RunnerConfig, run_eval};

        let env = TestEnv::new();
        let models = env.root().join("conf").join("models");
        std::fs::create_dir_all(&models).unwrap();
        std::fs::write(models.join("foundation.toml"), "context_window = 4096\n").unwrap();

        let tasks = vec![
            EvalTask::new("small", "hi", Vec::new()),
            EvalTask::new("big", "x".repeat(22_600), Vec::new()),
        ];
        let cfg = RunnerConfig {
            port: 1,
            timeout_secs: 1,
            max_retries: 0,
            allow_model_substitution: false,
            ..Default::default()
        };
        let outcomes = run_eval("foundation", &tasks, &cfg);
        let categories: Vec<&str> = outcomes
            .iter()
            .map(|o| o.failure_category.as_str())
            .collect();
        assert_eq!(categories, [FAIL_INFRA, FAIL_CONTEXT], "{outcomes:?}");
        assert!(
            outcomes[1]
                .error
                .as_deref()
                .is_some_and(|e| e.contains("does not fit"))
        );
        assert!(outcomes.iter().all(|o| !o.was_measured()));
        drop(env);
    }
}
