//! Every failure shape the multi-engine search helper can produce, without
//! needing a Python interpreter.
//!
//! In a sibling file for the house 500-line cap, following the convention
//! `weekend_tests.rs` already set. These exist because each of these outcomes
//! used to collapse into an empty `Vec`, which downstream is indistinguishable
//! from a search that legitimately found nothing — the bug that made a weekend
//! plan report "no events found" when nothing had searched.

use crate::ztools::weekend::classify_helper_output;
use std::os::unix::process::ExitStatusExt;
use std::process::{ExitStatus, Output};

/// A synthetic process result. `from_raw(code << 8)` is how a wait-status
/// encodes a normal exit on Unix, so `status.code()` reads back as `code`.
fn output(code: i32, stdout: &[u8], stderr: &str) -> std::io::Result<Output> {
    Ok(Output {
        status: ExitStatus::from_raw(code << 8),
        stdout: stdout.to_vec(),
        stderr: stderr.as_bytes().to_vec(),
    })
}

#[test]
fn a_successful_helper_yields_its_snippets() {
    let got = classify_helper_output(output(0, br#"["one: a", "two: b"]"#, "")).unwrap();
    assert_eq!(got, vec!["one: a".to_string(), "two: b".to_string()]);
}

/// An empty result is a legitimate answer and must stay distinguishable from
/// every failure below — `Ok(vec![])`, never `Err`.
#[test]
fn a_search_that_genuinely_found_nothing_is_an_empty_ok() {
    assert_eq!(
        classify_helper_output(output(0, b"[]", "")).unwrap(),
        Vec::<String>::new()
    );
}

#[test]
fn a_nonzero_exit_reports_the_code_and_the_last_thing_it_said() {
    let err = classify_helper_output(output(
        1,
        b"",
        "Traceback (most recent call last):\nModuleNotFoundError: No module named 'ddgs'\n",
    ))
    .unwrap_err();
    assert!(err.contains("exited Some(1)"), "{err}");
    assert!(
        err.contains("No module named 'ddgs'"),
        "the LAST line carries the cause, not the first: {err}"
    );
}

/// A helper that dies saying nothing still has to produce a reason, not an
/// empty sentence the operator cannot act on.
#[test]
fn a_silent_nonzero_exit_still_states_something() {
    let err = classify_helper_output(output(3, b"", "   \n\n")).unwrap_err();
    assert!(err.contains("exited Some(3)"), "{err}");
    assert!(err.contains("no output"), "{err}");
}

#[test]
fn output_that_is_not_utf8_is_a_stated_failure() {
    let err = classify_helper_output(output(0, &[0xff, 0xfe, 0x00], "")).unwrap_err();
    assert!(err.contains("non-UTF-8"), "{err}");
}

#[test]
fn output_that_is_not_the_promised_json_array_is_a_stated_failure() {
    let err = classify_helper_output(output(0, b"not json", "")).unwrap_err();
    assert!(err.contains("unparseable JSON"), "{err}");
    // Valid JSON of the wrong SHAPE is just as wrong as invalid JSON.
    let err = classify_helper_output(output(0, br#"{"results": []}"#, "")).unwrap_err();
    assert!(err.contains("unparseable JSON"), "{err}");
}

#[test]
fn a_helper_that_never_started_says_so() {
    let err = classify_helper_output(Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "no such file or directory",
    )))
    .unwrap_err();
    assert!(err.contains("could not start"), "{err}");
    assert!(
        err.contains("no such file"),
        "it carries the OS reason: {err}"
    );
}
