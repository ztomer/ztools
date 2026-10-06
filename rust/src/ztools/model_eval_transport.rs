//! The smoke path's one request, classified: an answer to score, or a refusal
//! with its reason.
//!
//! Split out of `model_eval.rs` for the house 500-line cap at the seam the defect
//! lived in. `model_eval.rs` holds what a run DOES with an answer; this holds how
//! one request is judged to have produced one. That is the whole fix — a request
//! that never reached the model must arrive as a refusal rather than as an empty
//! answer, and the two are one edit apart only if they live in one file.

use reqwest::blocking::Client;

/// What one request to the model produced.
pub(super) enum Answer {
    /// The model answered; here is what it said, and it gets scored.
    Content(String),
    /// Nothing was measured. `reason` is the one-line verdict that goes in the
    /// table; `detail` is the full error chain, for stderr.
    Refused { reason: String, detail: String },
}

/// The innermost cause of a reqwest error, in one line.
///
/// `{e:?}` on a refused connection is four hundred characters of
/// `hyper_util::client::legacy::Error(Connect, ConnectError("tcp connect error",
/// 127.0.0.1:54055, Os { code: 61, … }))`, and the table printed it once per
/// task — five near-identical rows nobody can read. The innermost `Display` is
/// the sentence an operator acts on ("tcp connect error: Connection refused"),
/// and the whole chain still goes to stderr where nothing has to fit it in a
/// column.
///
/// Bounded at [`CAUSE_DEPTH`] hops: this walks a foreign error chain, and a
/// source that returned itself would otherwise hang the run on the very path
/// that exists to report a broken one.
const CAUSE_DEPTH: usize = 8;

fn cause_line(e: &dyn std::error::Error) -> String {
    let mut last = e.to_string();
    let mut src = e.source();
    for _ in 0..CAUSE_DEPTH {
        match src {
            Some(inner) => {
                last = inner.to_string();
                src = inner.source();
            }
            None => break,
        }
    }
    last
}

/// One chat completion, classified.
///
/// THE SWALLOW THIS REPLACES: `if let Ok(r) = resp && r.status().is_success()`
/// left every other outcome — a refused connection, a 503, a body that is not
/// JSON — as an EMPTY STRING, which the loop then scored as `0/total` and filed
/// as `"failed"`. A dead inference server therefore produced five rows of 0.0%
/// and an exit code of 0, indistinguishable from a model that answered every
/// task wrongly. The distinction is the whole function: a request that never
/// reached the model has no score, and the reason travels out of here so it can
/// be printed.
pub(super) fn ask(client: &Client, url: &str, payload: &serde_json::Value) -> Answer {
    match client.post(url).json(payload).send() {
        Ok(r) if r.status().is_success() => match r.json::<serde_json::Value>() {
            Ok(json) => Answer::Content(
                json["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
            ),
            Err(e) => {
                let reason = format!("the answer from {url} is not JSON: {}", cause_line(&e));
                Answer::Refused {
                    detail: format!("{url} answered 200 with a body that is not JSON: {e:?}"),
                    reason,
                }
            }
        },
        Ok(r) => {
            let reason = format!("{url} answered HTTP {}", r.status());
            Answer::Refused {
                detail: reason.clone(),
                reason,
            }
        }
        Err(e) => {
            let reason = format!("{url} gave no answer: {}", cause_line(&e));
            Answer::Refused {
                detail: format!("{reason}: {e:?}"),
                reason,
            }
        }
    }
}
