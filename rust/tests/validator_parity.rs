//! Validator parity: the Rust taxes validators must still produce the verdicts
//! the retired Python validators produced.
//!
//! Until 2026-09-13 this was a two-process gate: this file PRINTED
//! `PARITY <task>|<score>|<reason>` lines and a pytest half recomputed the
//! same verdicts with the Python stack and diffed them. The Python stack is
//! gone; the byte-agreement property survives as a golden. On the last run of
//! the Python validators their verdicts over every fixture answer in
//! `tests/fixtures/validator_parity/` were frozen into
//! `expected_python_verdicts.json`, and this test asserts the Rust validators
//! reproduce each `(score, reason)` byte for byte.
//!
//! A change to a validator that moves one of these numbers is a change to the
//! reference behaviour; if that is intended, regenerate the golden from the
//! new Rust output deliberately and say so in the diff.

use serde_json::Value;
use std::collections::BTreeMap;
use ztools::eval::validators::{
    validate_taxes_anomalies, validate_taxes_audit_readiness, validate_taxes_qa,
    validate_taxes_slip_qa, validate_taxes_synthesis, validate_taxes_yoy_narrative,
};

fn fixtures_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("tests/fixtures/validator_parity")
}

#[derive(serde::Deserialize, Debug, PartialEq, Eq)]
struct Verdict {
    score: i64,
    reason: String,
}

fn expected() -> BTreeMap<String, Verdict> {
    let raw = std::fs::read_to_string(fixtures_dir().join("expected_python_verdicts.json"))
        .expect("expected_python_verdicts.json fixture");
    serde_json::from_str(&raw).expect("expected verdicts json")
}

fn rust_verdict(task: &str, text: String) -> Verdict {
    let v = Value::String(text);
    let (score, reason) = match task {
        "taxes_anomalies" => validate_taxes_anomalies(&v),
        "taxes_audit_readiness" => validate_taxes_audit_readiness(&v),
        "taxes_synthesis" => validate_taxes_synthesis(&v),
        "taxes_qa" => validate_taxes_qa(&v, None),
        "taxes_slip_qa" => validate_taxes_slip_qa(&v, None),
        "taxes_yoy_narrative" => validate_taxes_yoy_narrative(&v, None),
        other => panic!("no Rust validator wired for fixture {other}"),
    };
    Verdict { score, reason }
}

#[test]
fn rust_validators_reproduce_the_frozen_python_verdicts_byte_for_byte() {
    let expected = expected();
    assert!(
        expected.len() >= 6,
        "expected all six taxes fixtures, golden holds {}",
        expected.len()
    );
    let mut compared = 0;
    for (task, want) in &expected {
        let text = std::fs::read_to_string(fixtures_dir().join(format!("{task}.txt")))
            .unwrap_or_else(|e| panic!("fixture answer for {task}: {e}"));
        let got = rust_verdict(task, text);
        assert_eq!(
            &got, want,
            "{task}: Rust verdict drifted from the frozen Python verdict"
        );
        compared += 1;
    }
    assert_eq!(compared, expected.len());
}

#[test]
fn every_fixture_answer_has_a_golden_and_vice_versa() {
    // A fixture without a golden is an answer nobody checks; a golden without
    // a fixture is a check nobody can run. Both are silent holes.
    let expected = expected();
    let mut on_disk: Vec<String> = std::fs::read_dir(fixtures_dir())
        .unwrap()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            name.strip_suffix(".txt").map(str::to_owned)
        })
        .collect();
    on_disk.sort();
    let golden: Vec<String> = expected.keys().cloned().collect();
    assert_eq!(on_disk, golden);
}

#[test]
fn the_golden_can_fail() {
    // Calibration: the golden must be reachable from a changed answer, so a
    // green run above is evidence rather than tautology. The rubric grounds
    // the answer on expected signals (`t1135`, `form 106`, `box 38`, ...);
    // renumbering the T4 box drops a signal and moves the grounding band.
    let text = std::fs::read_to_string(fixtures_dir().join("taxes_anomalies.txt")).unwrap();
    let base = rust_verdict("taxes_anomalies", text.clone());
    let mutated = text.replace("box 38", "box 99");
    assert_ne!(mutated, text, "mutation was a no-op");
    let moved = rust_verdict("taxes_anomalies", mutated);
    assert_ne!(
        base, moved,
        "dropping a grounding signal must change the verdict"
    );
}
