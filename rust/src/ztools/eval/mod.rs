//! Eval machinery: output cleaning (`clean.rs`) and task validators
//! (`validate.rs`). Ported from `lib/content_processing.py` and
//! `eval/validate.py` so the Rust eval path judges the same cleaned, parsed
//! text the Python eval does.

pub mod clean;
pub mod budgets;
pub mod discrimination;
pub mod failures;
pub mod model_resolve;
pub mod oversize;
pub mod prompts;
pub mod quirks;
pub mod gpu_lock;
pub mod prefill;
pub mod runner;
pub mod signals;
pub mod samples;
pub mod task_loader;
pub mod transport;
pub mod validate;
pub mod validators;
pub mod watchdog;

pub use budgets::max_tokens_for_task;
pub use failures::{classify_failure, reasoning_retry_budget, Diagnosis, FAIL_CONTENT, FAIL_INFRA, FAIL_PARSE, FAIL_REASONING, FAIL_TIMEOUT};
pub use oversize::{estimate_model_memory_gb, is_thrashing, model_disk_bytes, oversize_refusal, OVERSIZE_OVERRIDE_ENV};
pub use clean::{clean_model_output, extract_content_from_code_blocks, extract_json};
pub use discrimination::{classify, distinct_values, disagreements, is_gate, ranking_mean, ranking_tasks};
// Generated from references/eval/tasks_prompts.py by tools/gen_rust_prompts.py;
// byte-parity gated by references/tests/test_rust_prompt_parity.py.
pub use prompts::{
    CONTRADICTION_PHRASE, FALSEHOOD_PHRASES, FILENAME_INJECTION_KEYWORDS,
    FILENAME_INJECTION_MARKERS, FILENAME_INJECTION_PROMPT, FILE_SUMMARY_PROMPT,
    FILE_SUMMARY_PROMPT_MIXED, IMAGE_RENAME_PROMPT, IMAGE_RENAME_PROMPT_MIXED,
    KEY_FACTS, MISATTRIBUTION_TIMELINE, RENAME_PROMPT, RENAME_PROMPT_MIXED,
    TWITTER_PROMPT, TWITTER_PROMPT_ACCURACY, TWITTER_PROMPT_CONTRADICTION,
    TWITTER_PROMPT_MISATTRIBUTION, TWITTER_PROMPT_MIXED, WEEKEND_USR_FIXED,
    WEEKEND_USR_FIXED_MIXED, WEEKEND_USR_TRANSIENT, WEEKEND_USR_TRANSIENT_MIXED,
};
pub use gpu_lock::{
    foreign_holder, lock_dir, GpuLockGuard, DEFAULT_LOCK_DIR, DEFAULT_MAX_IDLE_SECS,
};
pub use samples::{add_sample, clean_estimate, estimate_from, median, Sample, SAMPLE_WINDOW};
pub use task_loader::{
    get_built_in_smoke_tasks, load_all_eval_tasks, load_taxes_tasks_from_dir, run_check,
    ChatMessage, Check, EvalTask,
};
pub use validate::validate_file_summary;
pub use validators::*;
pub use runner::{run_eval, run_eval_with_signals, RunnerConfig, TaskOutcome};
pub use model_resolve::{
    default_fallback_chain, disk_corroborated, fetch_roster, is_generative_model,
    is_missing_model_error, parameter_billions, substitute_model, RosterEntry,
    MISSING_MODEL_MARKERS,
};
pub use quirks::apply_model_quirks;
pub use prefill::{measure_prefill_rate, record_prefill_rate, MAX_PLAUSIBLE_PREFILL_RATE};
pub use signals::{
    default_eval_timeout, derived_timeout, effective_timeout, load_signals,
    machine_is_uncontended, memory_pressure, record_capability_sample, record_signal,
    save_signals, signals_path, SignalStore, MAX_EVAL_TIMEOUT,
};
pub use watchdog::{is_stalled, model_stall_duration, stalled_for, DEFAULT_MODEL_STALL_SECONDS};
