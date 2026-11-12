//! Cluster tweets that discuss the same event, by the embeddings Osaurus serves.
//!
//! OUTSIDE the residency seam's load and release on purpose (ZT.EMBEDSEAM, measured 2026-11-11
//! against Osaurus 1.0.1 under the GPU lock): `/v1/embeddings` serves a static
//! embedding model (`potion-base-4M`) that answers in ~10 ms and never appears
//! in `/health`'s `resident_models` or `loaded`, not right after the call and
//! not 40 s later, so there is no weight to load and none to release.
//!
//! WHICH MODEL (ZT.EMBEDMODEL). The name used to be hard-coded as
//! `nomic-embed-text`, which Osaurus refuses with 400
//! `unsupported_embedding_model`; the refusal was swallowed into "every tweet
//! its own cluster", so clustering never happened and nothing said so. The
//! server is now asked: Osaurus does not list embedding models in
//! `/v1/models`, it names them in that refusal ("Use 'potion-base-4M' or
//! ..."), so a request with the configured name (or none) either answers or
//! says what it serves, and the served model that has a calibrated threshold
//! is used. Every way clustering can be off is an [`Outcome::Off`] with its
//! reason, which the run prints and the summary document carries.

use crate::ztools::twitter::Tweet;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[path = "embeddings_calibration.rs"]
mod calibration;
pub use calibration::{Calibration, calibration_for};

#[path = "embeddings_hebrew.rs"]
pub mod hebrew;
pub use hebrew::Hebrew;

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
    #[serde(default)]
    model: Option<String>,
}

#[derive(Deserialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Deserialize)]
struct ErrorDetail {
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    message: Option<String>,
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

/// What clustering did, or why it did not.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// The tweets were embedded by `model` and grouped at `threshold`.
    Clustered {
        model: String,
        threshold: f32,
        /// Tweets the model cannot represent, left out of the comparison.
        unembeddable: usize,
        /// The Hebrew ones among them: paired by the local model, or why not.
        hebrew: Hebrew,
        /// Set when the run did not get what was configured (a refused name).
        notice: Option<String>,
    },
    /// No clustering: every tweet is its own cluster, for `reason`.
    Off { reason: String },
}

/// The clusters, in first-member order, and how they came to be.
#[derive(Debug, Clone)]
pub struct Clustering {
    pub clusters: Vec<Vec<Tweet>>,
    pub outcome: Outcome,
}

impl Clustering {
    /// The run's line for the terminal.
    #[must_use]
    pub fn status_line(&self) -> String {
        match &self.outcome {
            Outcome::Clustered {
                model,
                threshold,
                unembeddable,
                hebrew,
                notice,
            } => {
                let mut line = format!(
                    "→ clustering: {} tweets into {} clusters by {model} at {threshold}",
                    self.clusters.iter().map(Vec::len).sum::<usize>(),
                    self.clusters.len()
                );
                if let Some(clause) = hebrew.clause() {
                    line.push_str("; ");
                    line.push_str(&clause);
                }
                let other = unembeddable.saturating_sub(hebrew.tweets());
                if other > 0 {
                    line.push_str("; ");
                    line.push_str(&other.to_string());
                    line.push_str(" tweets in a script the model cannot read kept apart");
                }
                if let Some(n) = notice {
                    line = format!("⚠ {n}\n{line}");
                }
                line
            }
            Outcome::Off { reason } => format!("⚠ {}", off_sentence(reason)),
        }
    }

    /// The document's line, when the run is not the plain clustered case.
    #[must_use]
    pub fn document_note(&self) -> Option<String> {
        match &self.outcome {
            Outcome::Clustered {
                notice,
                hebrew: hebrew @ Hebrew::Apart { fault: true, .. },
                ..
            } => {
                let lead = notice.as_ref().map_or_default(|n| format!("> ⚠ {n}\n"));
                Some(format!("{lead}> ⚠ {}", hebrew.clause().unwrap_or_default()))
            }
            Outcome::Clustered { notice, .. } => notice.as_ref().map(|n| format!("> ⚠ {n}")),
            Outcome::Off { reason } => Some(format!("> ⚠ {}", off_sentence(reason))),
        }
    }
}

fn off_sentence(reason: &str) -> String {
    format!("Clustering off, every tweet is its own cluster: {reason}")
}

fn singletons(tweets: &[Tweet], reason: String) -> Clustering {
    Clustering {
        clusters: tweets.iter().map(|t| vec![t.clone()]).collect(),
        outcome: Outcome::Off { reason },
    }
}

/// Group `vectors` greedily against each cluster's first member. Only the
/// `eligible` ones are compared; the rest stay alone. Indices, first-member
/// order.
#[must_use]
pub fn cluster_indices(vectors: &[Vec<f32>], eligible: &[bool], threshold: f32) -> Vec<Vec<usize>> {
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    for (i, emb) in vectors.iter().enumerate() {
        let home = if eligible.get(i).copied().unwrap_or(false) {
            clusters
                .iter_mut()
                .find(|c| eligible[c[0]] && cosine_similarity(emb, &vectors[c[0]]) >= threshold)
        } else {
            None
        };
        match home {
            Some(c) => c.push(i),
            None => clusters.push(vec![i]),
        }
    }
    clusters
}

/// One request's answer.
enum Answer {
    Vectors {
        model: Option<String>,
        vectors: Vec<Vec<f32>>,
    },
    /// 400 `unsupported_embedding_model`: the names the server says it serves.
    Refused {
        served: Vec<String>,
    },
    Failed(String),
}

/// The quoted names after "Use" in Osaurus's refusal message.
fn served_names(message: &str) -> Vec<String> {
    let Some(at) = message.find("Use ") else {
        return Vec::new();
    };
    message[at..]
        .split('\'')
        .skip(1)
        .step_by(2)
        .map(str::to_string)
        .filter(|s| !s.is_empty())
        .collect()
}

fn ask(client: &Client, url: &str, model: &str, texts: &[&str]) -> Answer {
    let req = EmbedRequest {
        model,
        input: texts.to_vec(),
    };
    let resp = match client.post(url).json(&req).send() {
        Ok(r) => r,
        Err(e) => return Answer::Failed(format!("the embeddings endpoint is unreachable ({e})")),
    };
    let status = resp.status();
    let body = resp.text().unwrap_or_default();
    if status.is_success() {
        return match serde_json::from_str::<EmbedResponse>(&body) {
            Ok(r) => Answer::Vectors {
                model: r.model,
                vectors: r.data.into_iter().map(|d| d.embedding).collect(),
            },
            Err(e) => Answer::Failed(format!(
                "the embeddings answer is not the expected shape ({e})"
            )),
        };
    }
    let detail = serde_json::from_str::<ErrorBody>(&body)
        .ok()
        .map(|b| b.error);
    if let Some(d) = &detail
        && d.code.as_deref() == Some("unsupported_embedding_model")
    {
        return Answer::Refused {
            served: served_names(d.message.as_deref().unwrap_or("")),
        };
    }
    Answer::Failed(format!(
        "the embeddings endpoint answered HTTP {}{}",
        status.as_u16(),
        detail
            .and_then(|d| d.code)
            .map_or_default(|c| format!(" {c}"))
    ))
}

/// Cluster `tweets` by the embedding model the server serves.
///
/// Infallible by design: every failure is an [`Outcome::Off`] whose reason the
/// caller prints, never an error swallowed into singletons.
#[must_use]
pub fn cluster_tweets(
    tweets: &[Tweet],
    base_url: &str,
    config: &crate::config::ZtoolsConfig,
) -> Clustering {
    if tweets.is_empty() {
        return Clustering {
            clusters: Vec::new(),
            outcome: Outcome::Off {
                reason: "no tweets".to_string(),
            },
        };
    }
    // No weight to load, but still a request to the server: made under the
    // machine-wide GPU lock like every other (ZT.TWGPULOCK).
    let clustering = match crate::ztools::residency::locked(
        "ztools embeddings (cluster tweets)",
        crate::ztools::residency::Timing::default(),
        || cluster_locked(tweets, base_url, config),
    ) {
        Ok(clustering) => clustering,
        Err(e) => return singletons(tweets, format!("{e:#}")),
    };
    with_hebrew(
        clustering,
        base_url,
        &config.twitter_model,
        hebrew::CALIBRATED_JUDGES,
        crate::ztools::residency::Timing::default(),
    )
}

/// The Hebrew phase: the tweets the embedding model could not read, paired by
/// the summarize slot's `model` ([`hebrew`]). A clustering that is off stays
/// off.
#[must_use]
pub fn with_hebrew(
    clustering: Clustering,
    base_url: &str,
    model: &str,
    calibrated: &[&str],
    timing: crate::ztools::residency::Timing,
) -> Clustering {
    let Outcome::Clustered {
        model: embedder,
        threshold,
        unembeddable,
        notice,
        ..
    } = clustering.outcome
    else {
        return clustering;
    };
    let unreadable = |t: &Tweet| calibration_for(&embedder).is_some_and(|c| !c.can_read(&t.text));
    let (clusters, hebrew) = hebrew::pair_hebrew(
        clustering.clusters,
        unreadable,
        base_url,
        model,
        calibrated,
        timing,
    );
    Clustering {
        clusters,
        outcome: Outcome::Clustered {
            model: embedder,
            threshold,
            unembeddable,
            hebrew,
            notice,
        },
    }
}

fn cluster_locked(
    tweets: &[Tweet],
    base_url: &str,
    config: &crate::config::ZtoolsConfig,
) -> Clustering {
    let client = match Client::builder().timeout(Duration::from_secs(3)).build() {
        Ok(c) => c,
        Err(e) => return singletons(tweets, format!("no HTTP client ({e})")),
    };
    let url = format!("{}/v1/embeddings", base_url.trim_end_matches('/'));
    let texts: Vec<&str> = tweets.iter().map(|t| t.text.as_str()).collect();

    let configured = config
        .twitter_embedding_model
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let mut notice = None;
    let (model, vectors) = match ask(&client, &url, configured.unwrap_or(""), &texts) {
        Answer::Vectors { model, vectors } => {
            (model.or_else(|| configured.map(str::to_string)), vectors)
        }
        Answer::Refused { served } => {
            let Some(pick) = served.iter().find(|m| calibration_for(m).is_some()) else {
                return singletons(
                    tweets,
                    format!(
                        "the server refuses embedding model '{}' and serves [{}], none with a calibrated similarity threshold",
                        configured.unwrap_or(""),
                        served.join(", ")
                    ),
                );
            };
            if let Some(c) = configured {
                notice = Some(format!(
                    "configured embedding model '{c}' is refused by the server (unsupported_embedding_model); clustered with the served '{pick}' instead"
                ));
            }
            match ask(&client, &url, pick, &texts) {
                Answer::Vectors { vectors, .. } => (Some(pick.clone()), vectors),
                Answer::Refused { .. } => {
                    return singletons(
                        tweets,
                        format!("the server refuses '{pick}', the model it named"),
                    );
                }
                Answer::Failed(r) => return singletons(tweets, r),
            }
        }
        Answer::Failed(r) => return singletons(tweets, r),
    };
    let Some(model) = model else {
        return singletons(
            tweets,
            "the server did not say which embedding model answered".to_string(),
        );
    };
    let Some(cal) = calibration_for(&model) else {
        return singletons(
            tweets,
            format!("no similarity threshold is calibrated for embedding model '{model}'"),
        );
    };
    if vectors.len() != tweets.len() {
        return singletons(
            tweets,
            format!(
                "{} embeddings came back for {} tweets",
                vectors.len(),
                tweets.len()
            ),
        );
    }
    let eligible: Vec<bool> = tweets.iter().map(|t| cal.can_read(&t.text)).collect();
    let unembeddable = eligible.iter().filter(|e| !**e).count();
    let clusters = cluster_indices(&vectors, &eligible, cal.threshold)
        .into_iter()
        .map(|c| c.into_iter().map(|i| tweets[i].clone()).collect())
        .collect();
    Clustering {
        clusters,
        outcome: Outcome::Clustered {
            model,
            threshold: cal.threshold,
            unembeddable,
            hebrew: Hebrew::None,
            notice,
        },
    }
}

#[cfg(test)]
#[path = "embeddings_tests.rs"]
mod tests;
