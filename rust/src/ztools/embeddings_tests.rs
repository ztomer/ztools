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
