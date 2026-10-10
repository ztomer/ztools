//! W3: a line-per-row answer cut at the token limit keeps its whole rows.
//!
//! On 2026-10-10 17:17 the draft prompt (29,490 chars) made the model list
//! every candidate instead of 10; the answer hit `max_tokens`, `llm::chat`
//! refused it as cut -- rightly, for a summary or a JSON object -- and
//! `call_llm_text` turned that refusal into "no answer" twice, so the plan
//! came from the monolithic fallback. Replayed, the same prompt gave 7,253
//! chars of complete rows before the cut.

use std::io::{Read, Write};

use super::*;

const CUT_ROWS: &str = "Fall Fair | Woodbridge | Oct 10 | Free | all ages | rides\n\
Pumpkinfest | Downsview Park | Oct 9-12 | unknown | all ages | corn maze\n\
Robotics Work";

/// A server whose every answer is `content` stopped by the token limit.
fn cut_server(content: &'static str) -> String {
    cut_server_recording(content).0
}

/// [`cut_server`], also handing back every request it was sent.
fn cut_server_recording(content: &'static str) -> (String, std::sync::mpsc::Receiver<String>) {
    let (tx, rx) = std::sync::mpsc::channel();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        while let Ok((mut stream, _)) = listener.accept() {
            let mut buf = vec![0u8; 65536];
            let n = stream.read(&mut buf).unwrap_or(0);
            let _ = tx.send(String::from_utf8_lossy(&buf[..n]).into_owned());
            let body = serde_json::json!({"choices": [
                {"message": {"content": content}, "finish_reason": "length"}
            ]})
            .to_string();
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(resp.as_bytes());
        }
    });
    (format!("http://{addr}"), rx)
}

#[test]
fn a_cut_line_answer_keeps_its_complete_rows() {
    let _env = crate::test_env::TestEnv::new();
    let config = crate::config::ZtoolsConfig {
        osaurus_url: cut_server(CUT_ROWS),
        ..Default::default()
    };
    let answer = call_llm_text("list the events", &config).expect("the whole rows survive the cut");
    assert_eq!(
        answer,
        "Fall Fair | Woodbridge | Oct 10 | Free | all ages | rides\n\
Pumpkinfest | Downsview Park | Oct 9-12 | unknown | all ages | corn maze"
    );
}

/// A cut answer with no complete row is still no answer.
#[test]
fn a_cut_answer_with_no_whole_row_is_no_answer() {
    let _env = crate::test_env::TestEnv::new();
    let config = crate::config::ZtoolsConfig {
        osaurus_url: cut_server("Robotics Work"),
        ..Default::default()
    };
    assert_eq!(call_llm_text("list the events", &config), None);
}

/// The planner's text phases decode on their own policy: they used to borrow
/// the Twitter summary's call, so `TWITTER_TEMPERATURE` (or any change to the
/// summary's decoding) moved the weekend plan too.
#[test]
#[serial_test::serial]
fn the_weekend_text_calls_never_take_the_summary_decoding() {
    let env = crate::test_env::TestEnv::new();
    env.set(crate::ztools::twitter::TEMPERATURE_ENV, "1.5");
    let (url, sent) = cut_server_recording(CUT_ROWS);
    let config = crate::config::ZtoolsConfig {
        osaurus_url: url,
        ..Default::default()
    };
    let _ = call_llm_text("list the events", &config);
    let request = sent.recv().expect("the call reached the server");
    let body = &request[request.find("\r\n\r\n").expect("a body") + 4..];
    let payload: serde_json::Value = serde_json::from_str(body).expect("a JSON body");
    assert_eq!(
        payload["temperature"], TEXT_SAMPLING.temperature,
        "{payload}"
    );
    drop(env);
}
