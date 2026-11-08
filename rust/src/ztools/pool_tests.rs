use super::*;

fn ok_answer() -> Value {
    json!({"status": "ok", "text": " ## Summary\n- a ", "tier": "opencode/glm-5.3-flash-free",
           "detail": "", "attempts": []})
}

#[test]
fn structured_ok_is_an_answer() {
    let a = read_answer(&json!({"content": [], "structuredContent": ok_answer()})).unwrap();
    assert_eq!(a.text, "## Summary\n- a");
    assert_eq!(a.tier, "opencode/glm-5.3-flash-free");
}

#[test]
fn unverified_with_text_is_an_answer_wrapped_or_as_text() {
    let mut v = ok_answer();
    v["status"] = json!("unverified");
    let wrapped = json!({"structuredContent": {"result": v.clone()}});
    assert_eq!(read_answer(&wrapped).unwrap().text, "## Summary\n- a");
    let as_text = json!({"content": [{"type": "text", "text": v.to_string()}]});
    assert_eq!(
        read_answer(&as_text).unwrap().tier,
        "opencode/glm-5.3-flash-free"
    );
}

#[test]
fn escalation_is_a_refusal_naming_the_tiers() {
    let v = json!({"structuredContent": {"status": "escalate_to_caller", "text": "",
        "detail": "every tier cooled", "attempts": [{"tier": "opencode/x", "status": "cooling"}]}});
    let e = read_answer(&v).unwrap_err().to_string();
    assert!(e.contains("escalate_to_caller"), "{e}");
    assert!(e.contains("every tier cooled"), "{e}");
    assert!(e.contains("opencode/x cooling"), "{e}");
}

#[test]
fn ok_with_empty_text_is_not_an_answer() {
    let v = json!({"structuredContent": {"status": "ok", "text": "  "}});
    assert!(
        read_answer(&v)
            .unwrap_err()
            .to_string()
            .contains("ok with no usable text")
    );
}

#[test]
fn is_error_and_unreadable_results_are_errors() {
    let v = json!({"isError": true, "content": [{"type": "text", "text": "bad class"}]});
    assert!(
        read_answer(&v)
            .unwrap_err()
            .to_string()
            .contains("bad class")
    );
    let v = json!({"content": [{"type": "text", "text": "not json"}]});
    assert!(
        read_answer(&v)
            .unwrap_err()
            .to_string()
            .contains("no readable result")
    );
    let v = json!({"structuredContent": {}});
    assert!(
        read_answer(&v)
            .unwrap_err()
            .to_string()
            .contains("no status")
    );
}

#[test]
fn arguments_are_never_private() {
    let a = ask_arguments("bulk_read", "p");
    assert_eq!(a["task_class"], "bulk_read");
    assert_eq!(a["prompt"], "p");
    assert_eq!(a["private"], false);
}

/// A stdio MCP server in python3 that answers the handshake and `ask` with
/// `reply` (a JSON object literal), and refuses every other request.
const FAKE: &str = r#"
import json, sys
reply = json.loads(sys.argv[1])
for line in sys.stdin:
    msg = json.loads(line)
    if "id" not in msg:
        continue
    m = msg.get("method")
    if m == "initialize":
        res = {"protocolVersion": msg["params"]["protocolVersion"], "capabilities": {"tools": {}},
               "serverInfo": {"name": "fake-pool", "version": "0"}}
    elif m == "tools/call":
        res = {"content": [{"type": "text", "text": json.dumps(reply)}], "structuredContent": reply,
               "isError": False}
    elif m == "sleep":
        continue
    else:
        out = {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": m}}
        print(json.dumps(out), flush=True)
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": res}), flush=True)
"#;

pub fn fake_command(dir: &Path, reply: &Value) -> Vec<String> {
    let script = dir.join("fake_pool.py");
    std::fs::write(&script, FAKE).unwrap();
    vec![
        "python3".to_string(),
        script.display().to_string(),
        reply.to_string(),
    ]
}

#[test]
fn ask_speaks_mcp_to_the_server_and_reads_its_answer() {
    let dir = tempfile::tempdir().unwrap();
    let cmd = fake_command(dir.path(), &ok_answer());
    let a = ask(&cmd, SUMMARY_CLASS, "summarise", Duration::from_secs(30)).unwrap();
    assert_eq!(a.text, "## Summary\n- a");
}

#[test]
fn a_refusal_from_the_server_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let cmd = fake_command(
        dir.path(),
        &json!({"status": "refused", "detail": "private"}),
    );
    let e = ask(&cmd, SUMMARY_CLASS, "p", Duration::from_secs(30)).unwrap_err();
    assert!(format!("{e:#}").contains("refused"), "{e:#}");
}

#[test]
fn a_server_that_cannot_start_names_its_command_and_stderr() {
    let e = ask(&[], SUMMARY_CLASS, "p", Duration::from_secs(5)).unwrap_err();
    assert!(e.to_string().contains("no pool command"));
    let e = ask(
        &["/nonexistent/pool-mcp".to_string()],
        SUMMARY_CLASS,
        "p",
        Duration::from_secs(5),
    )
    .unwrap_err();
    assert!(format!("{e:#}").contains("/nonexistent/pool-mcp"), "{e:#}");
    // A server that dies saying why: the session never opens, and its words
    // are quoted.
    let cmd = vec![
        "sh".to_string(),
        "-c".to_string(),
        "echo uv: venv missing >&2; exit 3".to_string(),
    ];
    let e = ask(&cmd, SUMMARY_CLASS, "p", Duration::from_secs(10)).unwrap_err();
    let msg = format!("{e:#}");
    assert!(msg.contains("venv missing"), "{msg}");
}

#[test]
fn a_silent_server_times_out() {
    let cmd = vec!["sleep".to_string(), "30".to_string()];
    let e = ask(&cmd, SUMMARY_CLASS, "p", Duration::from_secs(1)).unwrap_err();
    assert!(e.to_string().contains("within 1s"), "{e:#}");
}
