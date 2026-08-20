use crate::ztools::twitter::Tweet;
use anyhow::Result;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: Vec<&'a str>,
}

#[derive(Deserialize)]
struct EmbedData {
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Use Ollama API to semantically cluster tweets that discuss the same event.
pub fn cluster_tweets(
    tweets: &[Tweet],
    base_url: &str,
    config: &crate::config::ZtoolsConfig,
) -> Result<Vec<Vec<Tweet>>> {
    if tweets.is_empty() {
        return Ok(Vec::new());
    }

    let client = Client::builder()
        .timeout(Duration::from_secs(config.llm_extended_timeout_secs))
        .build()?;

    let url = format!("{}/v1/embeddings", base_url.trim_end_matches('/'));

    let texts: Vec<&str> = tweets.iter().map(|t| t.text.as_str()).collect();
    // Assuming 'nomic-embed-text' is available, or use the default model
    let req = EmbedRequest {
        model: "nomic-embed-text",
        input: texts,
    };

    let resp = client.post(&url).json(&req).send();

    // Fallback if embeddings fail: just return each tweet in its own cluster
    let Ok(resp) = resp else {
        return Ok(tweets.iter().map(|t| vec![t.clone()]).collect());
    };

    let Ok(embed_resp) = resp.json::<EmbedResponse>() else {
        return Ok(tweets.iter().map(|t| vec![t.clone()]).collect());
    };

    let embeddings = embed_resp
        .data
        .into_iter()
        .map(|d| d.embedding)
        .collect::<Vec<_>>();
    if embeddings.len() != tweets.len() {
        return Ok(tweets.iter().map(|t| vec![t.clone()]).collect());
    }

    let mut clusters: Vec<Vec<(usize, Tweet)>> = Vec::new();
    let threshold = 0.85; // Similarity threshold for clustering

    for (i, tweet) in tweets.iter().enumerate() {
        let emb = &embeddings[i];
        let mut matched = false;

        for cluster in &mut clusters {
            // Compare against the first item in the cluster (centroid approximation)
            let center_idx = cluster[0].0;
            let center_emb = &embeddings[center_idx];
            if cosine_similarity(emb, center_emb) >= threshold {
                cluster.push((i, tweet.clone()));
                matched = true;
                break;
            }
        }

        if !matched {
            clusters.push(vec![(i, tweet.clone())]);
        }
    }

    Ok(clusters
        .into_iter()
        .map(|c| c.into_iter().map(|(_, t)| t).collect())
        .collect())
}

#[cfg(test)]
#[path = "embeddings_tests.rs"]
mod tests;
