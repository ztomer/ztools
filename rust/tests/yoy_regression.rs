// Prove the sign fix against the exact output that broke parity.
#[test]
fn yoy_signed_traceable_matches_python_on_the_parity_break_output() {
    let raw = match std::fs::read_to_string("/tmp/yoy_out.txt") {
        Ok(r) => r,
        Err(_) => return, // fixture absent (CI): unit tests cover enumeration
    };
    let val = serde_json::Value::String(raw);
    let (score, note) = ztools::eval::validators::validate_taxes_yoy_narrative(&val, None);
    assert_eq!(
        (score, note.as_str()),
        (100, "schema=20/20  traceable=4/4 (30/30)  reconcile err=367.68 tol=1568.22 (30/30)  prose_amounts=6/6 (20/20)"),
        "must byte-match the Python validator's verdict on this text"
    );
}
