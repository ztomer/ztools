use super::*;

#[test]
fn test_cosine_similarity() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    assert_eq!(cosine_similarity(&a, &b), 0.0);

    let c = vec![1.0, 0.0, 0.0];
    assert_eq!(cosine_similarity(&a, &c), 1.0);

    let d = vec![0.5, 0.5, 0.0];
    assert!(cosine_similarity(&a, &d) > 0.7);
}

#[test]
fn test_cluster_tweets_fallback() {
    let tweets = vec![
        Tweet {
            screen_name: "u1".into(),
            text: "t1".into(),
            created_at: "now".into(),
            favorite_count: 0,
            retweet_count: 0,
            reply_to: None,
        },
        Tweet {
            screen_name: "u2".into(),
            text: "t2".into(),
            created_at: "now".into(),
            favorite_count: 0,
            retweet_count: 0,
            reply_to: None,
        },
    ];

    let config = crate::config::ZtoolsConfig::default();

    // Using a fake URL so it triggers fallback
    let result = cluster_tweets(&tweets, "http://127.0.0.1:59999", &config).unwrap();

    // In fallback mode, it puts every tweet in its own cluster
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].len(), 1);
    assert_eq!(result[1].len(), 1);
}

#[test]
fn test_cluster_tweets_empty() {
    let tweets: Vec<Tweet> = vec![];
    let config = crate::config::ZtoolsConfig::default();
    let result = cluster_tweets(&tweets, "http://127.0.0.1:59999", &config).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_cosine_similarity_zero_vector() {
    let a = vec![0.0, 0.0, 0.0];
    let b = vec![1.0, 2.0, 3.0];
    assert_eq!(cosine_similarity(&a, &b), 0.0);
    assert_eq!(cosine_similarity(&b, &a), 0.0);
}

#[test]
fn test_cosine_similarity_opposite() {
    let a = vec![1.0, 0.0];
    let b = vec![-1.0, 0.0];
    assert!(cosine_similarity(&a, &b) < 0.9);
    let sim = cosine_similarity(&a, &b);
    assert!(sim < 0.0, "opposite vectors should be negative, got {sim}");
}

fn mock_tweet(name: &str) -> Tweet {
    Tweet {
        screen_name: name.into(),
        text: format!("text of {name}"),
        created_at: "now".into(),
        favorite_count: 0,
        retweet_count: 0,
        reply_to: None,
    }
}

/// Local HTTP mock that answers every request with `status_line` + `body`.
/// Loopback only: no live endpoint is ever contacted.
fn serve(status_line: &'static str, body: &'static str) -> String {
    use std::io::{Read, Write};
    use std::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        while let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 8192];
            let _ = stream.read(&mut buf);
            let resp = format!(
                "{}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                status_line,
                body.len(),
                body
            );
            if stream.write_all(resp.as_bytes()).is_err() {
                break;
            }
        }
    });
    format!("http://{addr}")
}

#[test]
fn similar_tweets_merge_into_one_cluster_and_distant_ones_start_their_own() {
    // t0 and t1 point the same way (cosine ~0.999 >= 0.85 threshold); t2 is
    // orthogonal to t0 (cosine 0) so it must open a second cluster.
    let body =
        r#"{"data":[{"embedding":[1.0,1.0]},{"embedding":[1.0,0.9]},{"embedding":[0.0,1.0]}]}"#;
    let url = serve("HTTP/1.1 200 OK", body);
    let tweets = vec![mock_tweet("t0"), mock_tweet("t1"), mock_tweet("t2")];
    let config = crate::config::ZtoolsConfig::default();

    let result = cluster_tweets(&tweets, &url, &config).unwrap();

    assert_eq!(result.len(), 2, "one merged cluster plus one singleton");
    assert_eq!(
        result[0].len(),
        2,
        "{:?}",
        result[0].iter().map(|t| &t.screen_name).collect::<Vec<_>>()
    );
    assert_eq!(result[0][0].screen_name, "t0");
    assert_eq!(result[0][1].screen_name, "t1");
    assert_eq!(result[1].len(), 1);
    assert_eq!(result[1][0].screen_name, "t2");
}

#[test]
fn an_http_error_status_falls_back_to_singletons() {
    let url = serve("HTTP/1.1 500 Internal Server Error", "{}");
    let tweets = vec![mock_tweet("u1"), mock_tweet("u2"), mock_tweet("u3")];
    let config = crate::config::ZtoolsConfig::default();

    let result = cluster_tweets(&tweets, &url, &config).unwrap();

    assert_eq!(
        result.len(),
        3,
        "a failed embedding call must not merge anything"
    );
    assert!(result.iter().all(|c| c.len() == 1));
}

#[test]
fn an_embedding_count_mismatch_falls_back_to_singletons() {
    // Two tweets but only one vector back: clustering would index out of sync
    // with the input, so the code must bail to singletons instead.
    let body = r#"{"data":[{"embedding":[1.0,0.0]}]}"#;
    let url = serve("HTTP/1.1 200 OK", body);
    let tweets = vec![mock_tweet("u1"), mock_tweet("u2"), mock_tweet("u3")];
    let config = crate::config::ZtoolsConfig::default();

    let result = cluster_tweets(&tweets, &url, &config).unwrap();

    assert_eq!(result.len(), 3);
    assert!(result.iter().all(|c| c.len() == 1));
}
