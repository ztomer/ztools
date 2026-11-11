//! ZT.EMBEDMODEL: the embedding model is the one the server serves, a refused
//! name is a visible reason, and the threshold is calibrated on recorded
//! vectors of the served model.

use super::super::*;
use super::{mock_tweet, sandboxed_config};

/// Osaurus 1.0.1's refusal, verbatim (probed 2026-11-11).
const REFUSAL: &str = r#"{"error":{"type":"invalid_request_error","code":"unsupported_embedding_model","param":"model","message":"Embedding model 'nomic-embed-text' is not supported by this endpoint. Use 'potion-base-4M' or 'minishlab\/potion-base-4M'. No embeddings were generated."}}"#;

/// A loopback server whose answer depends on the requested model: `answer`
/// gets the request body and returns the status line and the body.
fn serve_by(answer: fn(&str) -> (&'static str, String)) -> String {
    use std::io::{Read, Write};
    use std::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        while let Ok((mut stream, _)) = listener.accept() {
            let mut buf = vec![0u8; 1 << 16];
            let n = stream.read(&mut buf).unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]).to_string();
            let body = req.split("\r\n\r\n").nth(1).unwrap_or("");
            let (status, reply) = answer(body);
            let resp = format!(
                "{status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
                reply.len()
            );
            let _ = stream.write_all(resp.as_bytes());
        }
    });
    format!("http://{addr}")
}

/// Serves `potion-base-4M` only, refusing every other name the way Osaurus does.
fn osaurus_like(body: &str) -> (&'static str, String) {
    let v: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
    let n = v["input"].as_array().map_or(0, Vec::len);
    if v["model"] == "potion-base-4M" {
        let data: Vec<String> = (0..n)
            .map(|_| r#"{"embedding":[1.0,0.0]}"#.to_string())
            .collect();
        let reply = format!(
            r#"{{"model":"potion-base-4M","data":[{}]}}"#,
            data.join(",")
        );
        return ("HTTP/1.1 200 OK", reply);
    }
    ("HTTP/1.1 400 Bad Request", REFUSAL.to_string())
}

/// Refuses everything and names only a model nothing is calibrated for.
fn names_an_uncalibrated_model(_: &str) -> (&'static str, String) {
    (
        "HTTP/1.1 400 Bad Request",
        REFUSAL.replace("potion-base-4M", "mystery-embed"),
    )
}

#[test]
fn a_refused_configured_model_is_named_and_the_served_one_clusters() {
    let url = serve_by(osaurus_like);
    let (env, mut config) = sandboxed_config();
    config.twitter_embedding_model = Some("nomic-embed-text".into());
    let tweets = vec![mock_tweet("a"), mock_tweet("b")];

    let got = cluster_tweets(&tweets, &url, &config);

    let Outcome::Clustered { model, notice, .. } = &got.outcome else {
        panic!(
            "expected clustering by the served model, got {:?}",
            got.outcome
        );
    };
    assert_eq!(model, "potion-base-4M");
    let notice = notice
        .as_deref()
        .expect("a refused configured name must be said");
    assert!(
        notice.contains("'nomic-embed-text'") && notice.contains("refused"),
        "{notice}"
    );
    assert!(got.document_note().unwrap().contains("nomic-embed-text"));
    assert!(got.status_line().starts_with('⚠'), "{}", got.status_line());
    assert_eq!(got.clusters.len(), 1, "identical vectors merge");
    drop(env);
}

#[test]
fn with_nothing_configured_the_server_is_asked_and_no_notice_is_raised() {
    let url = serve_by(osaurus_like);
    let (env, config) = sandboxed_config();
    assert_eq!(config.twitter_embedding_model, None);

    let got = cluster_tweets(&[mock_tweet("a"), mock_tweet("b")], &url, &config);

    assert!(
        matches!(&got.outcome, Outcome::Clustered { model, notice: None, .. } if model == "potion-base-4M"),
        "{:?}",
        got.outcome
    );
    assert_eq!(got.document_note(), None);
    drop(env);
}

#[test]
fn a_refusal_with_no_calibrated_model_degrades_with_the_reason_in_the_output() {
    let url = serve_by(names_an_uncalibrated_model);
    let (env, mut config) = sandboxed_config();
    config.twitter_embedding_model = Some("nomic-embed-text".into());
    let tweets = vec![mock_tweet("a"), mock_tweet("b")];

    let got = cluster_tweets(&tweets, &url, &config);

    let Outcome::Off { reason } = &got.outcome else {
        panic!("expected clustering off, got {:?}", got.outcome);
    };
    assert!(
        reason.contains("nomic-embed-text") && reason.contains("mystery-embed"),
        "{reason}"
    );
    assert!(got.status_line().contains(reason.as_str()));
    assert!(got.document_note().unwrap().contains(reason.as_str()));
    assert_eq!(got.clusters.len(), 2);
    drop(env);
}

#[test]
fn an_unreachable_server_says_so_rather_than_clustering_silently() {
    let (env, config) = sandboxed_config();
    let got = cluster_tweets(&[mock_tweet("a")], "http://127.0.0.1:59999", &config);
    assert!(
        matches!(&got.outcome, Outcome::Off { reason } if reason.contains("unreachable")),
        "{:?}",
        got.outcome
    );
    drop(env);
}

#[test]
fn the_refusal_names_what_the_server_serves() {
    let msg = "Embedding model 'x' is not supported by this endpoint. Use 'potion-base-4M' or 'minishlab/potion-base-4M'. No embeddings were generated.";
    assert_eq!(
        served_names(msg),
        ["potion-base-4M", "minishlab/potion-base-4M"]
    );
    assert_empty!(served_names("Embedding model 'x' is not supported."));
}

#[test]
fn calibration_matches_with_and_without_the_organisation() {
    assert!(calibration_for("minishlab/potion-base-4M").is_some());
    assert!(calibration_for("POTION-BASE-4M").is_some());
    assert_eq!(calibration_for("nomic-embed-text"), None);
}

#[test]
fn a_latin_only_model_does_not_read_hebrew_or_letterless_text() {
    let cal = calibration_for("potion-base-4M").unwrap();
    assert!(cal.can_read("Bank of Canada cuts rates"));
    assert!(!cal.can_read("הרכבת לא תפעל בסוף השבוע"));
    assert!(!cal.can_read("123 456 !!!"));
}

// MARK: - The calibration, on recorded vectors of the served model

#[derive(serde::Deserialize)]
struct Recorded {
    model: String,
    tweets: Vec<RecordedTweet>,
}

#[derive(serde::Deserialize)]
struct RecordedTweet {
    group: String,
    text: String,
    embedding: Vec<f32>,
}

fn recorded() -> Recorded {
    serde_json::from_str(include_str!(
        "../../../tests/fixtures/embeddings/potion_base_4m.json"
    ))
    .expect("the committed fixture parses")
}

/// The groups `threshold` produces on the fixture, as sorted group-name lists.
fn groups_at(threshold: f32) -> Vec<Vec<String>> {
    let fx = recorded();
    let cal = calibration_for(&fx.model).expect("the fixture's model is calibrated");
    let vectors: Vec<Vec<f32>> = fx.tweets.iter().map(|t| t.embedding.clone()).collect();
    let eligible: Vec<bool> = fx.tweets.iter().map(|t| cal.can_read(&t.text)).collect();
    cluster_indices(&vectors, &eligible, threshold)
        .into_iter()
        .map(|c| c.into_iter().map(|i| fx.tweets[i].group.clone()).collect())
        .collect()
}

/// Near-duplicates (two accounts, one event) share a cluster; nothing else
/// does. Bites at a blind threshold: 0.0 merges every Latin tweet, 1.0 merges
/// none, and the old 0.85 misses the rate-cut pair (cosine 0.785).
fn assert_calibrated(threshold: f32) {
    let groups = groups_at(threshold);
    for cluster in &groups {
        assert!(
            cluster.iter().all(|g| *g == cluster[0]),
            "unrelated tweets merged at {threshold}: {cluster:?}"
        );
    }
    for pair in ["transit", "rates", "release"] {
        assert!(
            groups
                .iter()
                .any(|c| c.iter().filter(|g| *g == pair).count() == 2),
            "the {pair} near-duplicates did not cluster at {threshold}: {groups:?}"
        );
    }
}

#[test]
fn the_calibrated_threshold_clusters_near_duplicates_and_nothing_else() {
    let fx = recorded();
    assert_eq!(fx.model, "potion-base-4M");
    assert_calibrated(calibration_for(&fx.model).unwrap().threshold);
}

#[test]
fn a_blind_threshold_fails_the_calibration() {
    for blind in [0.0_f32, 1.0, 0.85] {
        let verdict = std::panic::catch_unwind(|| assert_calibrated(blind));
        assert!(
            verdict.is_err(),
            "threshold {blind} passed a calibration it cannot pass"
        );
    }
}
