//! The pieces of `models` that need no process: the `[best_models]` table the
//! check reads, and how a report becomes an exit status.

use super::*;
use crate::cli::Exit;

#[test]
fn best_models_are_the_five_slots_of_the_running_config() {
    let _env = crate::test_env::TestEnv::new();
    let table = best_models(&ZtoolsConfig::default());
    assert_eq!(
        table.keys().map(String::as_str).collect::<Vec<_>>(),
        ["filename", "json", "summarize", "think", "vlm"]
    );
    assert_eq!(table["think"], ZtoolsConfig::default().think_model);
}

#[test]
fn a_report_exits_by_its_status() {
    let mut report = Report::default();
    assert!(finish(&report).is_ok());
    report.errors = 1;
    report.repaired = 1;
    let code = |r: &Report| {
        finish(r)
            .unwrap_err()
            .downcast_ref::<Exit>()
            .map(|e| e.code)
    };
    assert_eq!(code(&report), Some(3));
    report.repaired = 0;
    report.unresolved = 1;
    assert_eq!(code(&report), Some(1));
}
