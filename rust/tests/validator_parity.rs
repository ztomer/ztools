//! The CI drift gate between the Rust and Python validator stacks.
//!
//! For every fixture answer in `tests/fixtures/validator_parity/`, prints one
//! `PARITY <task>|<score>|<reason>` line computed by the RUST validators. The
//! pytest side (`references/tests/test_rust_validator_parity.py`) computes the
//! same verdicts with the PYTHON validators and asserts the two stacks agree
//! byte-for-byte. If this test's output format changes, update the parser —
//! a silent format change would read as "nothing to compare" and green-light
//! exactly the drift this gate exists to catch.

use serde_json::Value;
use ztools::eval::validators::{
    validate_taxes_anomalies, validate_taxes_audit_readiness, validate_taxes_qa,
    validate_taxes_slip_qa, validate_taxes_synthesis, validate_taxes_yoy_narrative,
};

fn fixture_path(task: &str) -> String {
    let manifest = env!("CARGO_MANIFEST_DIR");
    format!("{manifest}/../tests/fixtures/validator_parity/taxes_{task}.txt")
}

#[test]
fn print_rust_verdicts_for_python_comparison() {
    for task in [
        "anomalies",
        "audit_readiness",
        "synthesis",
        "qa",
        "slip_qa",
        "yoy_narrative",
    ] {
        let text = std::fs::read_to_string(fixture_path(task))
            .unwrap_or_else(|e| panic!("fixture for {task}: {e}"));
        let v = Value::String(text);
        let (score, reason) = match task {
            "anomalies" => validate_taxes_anomalies(&v),
            "audit_readiness" => validate_taxes_audit_readiness(&v),
            "synthesis" => validate_taxes_synthesis(&v),
            "qa" => validate_taxes_qa(&v, None),
            "slip_qa" => validate_taxes_slip_qa(&v, None),
            "yoy_narrative" => validate_taxes_yoy_narrative(&v, None),
            _ => unreachable!(),
        };
        println!("PARITY taxes_{task}|{score}|{reason}");
    }
}
