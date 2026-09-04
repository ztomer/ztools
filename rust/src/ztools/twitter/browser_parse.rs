//! Parsing of x.com's HomeTimeline GraphQL payloads into plain Tweet structs.
//!
//! Port of `twitter/browser_parse.py`. Pure data transformation: no browser, no network.
use super::Tweet;

pub const TWITTER_TYPENAME_VISIBILITY: &str = "TweetWithVisibilityResults";
pub const TWITTER_ITEM_TIMELINE_TWEET: &str = "TimelineTweet";
pub const TWITTER_TYPE_TIMELINE_ADD_ENTRIES: &str = "TimelineAddEntries";
pub const UNKNOWN_USER: &str = "unknown";

/// Format: "Wed Aug 19 14:32:00 +0000 2026"
pub const TWITTER_DATE_FORMAT: &str = "%a %b %d %H:%M:%S %z %Y";

/// Parse tweets from x.com GraphQL JSON payload.
pub fn parse_tweets_from_response(data: &serde_json::Value) -> Vec<Tweet> {
    let mut tweets = Vec::new();

    let instructions = match data
        .get("data")
        .and_then(|d| d.get("home"))
        .and_then(|h| h.get("home_timeline_urt"))
        .and_then(|u| u.get("instructions"))
        .and_then(|i| i.as_array())
    {
        Some(i) => i,
        None => return tweets,
    };

    for instruction in instructions {
        if instruction.get("type").and_then(|t| t.as_str())
            != Some(TWITTER_TYPE_TIMELINE_ADD_ENTRIES)
        {
            continue;
        }

        let entries = match instruction.get("entries").and_then(|e| e.as_array()) {
            Some(e) => e,
            None => continue,
        };

        for entry in entries {
            let item_content = match entry.get("content").and_then(|c| c.get("itemContent")) {
                Some(ic) => ic,
                None => continue,
            };

            if item_content.get("itemType").and_then(|t| t.as_str())
                != Some(TWITTER_ITEM_TIMELINE_TWEET)
            {
                continue;
            }

            let mut tweet_result = match item_content
                .get("tweet_results")
                .and_then(|tr| tr.get("result"))
            {
                Some(r) => r,
                None => continue,
            };

            if tweet_result.get("__typename").and_then(|t| t.as_str())
                == Some(TWITTER_TYPENAME_VISIBILITY)
            {
                if let Some(inner) = tweet_result.get("tweet") {
                    tweet_result = inner;
                }
            }

            let legacy = match tweet_result.get("legacy") {
                Some(l) => l,
                None => continue,
            };

            let full_text = match legacy.get("full_text").and_then(|t| t.as_str()) {
                Some(t) if !t.is_empty() => t,
                _ => continue,
            };

            let created_at_str = match legacy.get("created_at").and_then(|t| t.as_str()) {
                Some(t) if !t.is_empty() => t,
                _ => continue,
            };

            let user_result = tweet_result
                .get("core")
                .and_then(|c| c.get("user_results"))
                .and_then(|ur| ur.get("result"));

            let screen_name = user_result
                .and_then(|u| {
                    u.get("core")
                        .and_then(|c| c.get("screen_name"))
                        .or_else(|| u.get("legacy").and_then(|l| l.get("screen_name")))
                })
                .and_then(|s| s.as_str())
                .unwrap_or(UNKNOWN_USER);

            let favorite_count = legacy
                .get("favorite_count")
                .and_then(|f| f.as_u64())
                .unwrap_or(0);
            let retweet_count = legacy
                .get("retweet_count")
                .and_then(|r| r.as_u64())
                .unwrap_or(0);
            let in_reply_to = legacy
                .get("in_reply_to_screen_name")
                .and_then(|s| s.as_str())
                .map(|s| s.to_string());

            tweets.push(Tweet {
                screen_name: screen_name.to_string(),
                text: full_text.to_string(),
                created_at: created_at_str.to_string(),
                favorite_count,
                retweet_count,
                reply_to: in_reply_to,
            });
        }
    }

    tweets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tweets_from_mock_graphql() {
        let payload = serde_json::json!({
            "data": {
                "home": {
                    "home_timeline_urt": {
                        "instructions": [
                            {
                                "type": "TimelineAddEntries",
                                "entries": [
                                    {
                                        "content": {
                                            "itemContent": {
                                                "itemType": "TimelineTweet",
                                                "tweet_results": {
                                                    "result": {
                                                        "__typename": "Tweet",
                                                        "core": {
                                                            "user_results": {
                                                                "result": {
                                                                    "core": {
                                                                        "screen_name": "rustlang"
                                                                    }
                                                                }
                                                            }
                                                        },
                                                        "legacy": {
                                                            "full_text": "Rust 1.80 is released!",
                                                            "created_at": "Thu Aug 20 12:00:00 +0000 2026",
                                                            "favorite_count": 1500,
                                                            "retweet_count": 350
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    },
                                    {
                                        "content": {
                                            "itemContent": {
                                                "itemType": "TimelineTweet",
                                                "tweet_results": {
                                                    "result": {
                                                        "__typename": "TweetWithVisibilityResults",
                                                        "tweet": {
                                                            "core": {
                                                                "user_results": {
                                                                    "result": {
                                                                        "legacy": {
                                                                            "screen_name": "developer"
                                                                        }
                                                                    }
                                                                }
                                                            },
                                                            "legacy": {
                                                                "full_text": "Shipping the Rust rewrite today!",
                                                                "created_at": "Thu Aug 20 12:05:00 +0000 2026",
                                                                "favorite_count": 42,
                                                                "retweet_count": 5,
                                                                "in_reply_to_screen_name": "rustlang"
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                ]
                            }
                        ]
                    }
                }
            }
        });

        let tweets = parse_tweets_from_response(&payload);
        assert_eq!(tweets.len(), 2);
        assert_eq!(tweets[0].screen_name, "rustlang");
        assert_eq!(tweets[0].text, "Rust 1.80 is released!");
        assert_eq!(tweets[0].favorite_count, 1500);
        assert_eq!(tweets[0].reply_to, None);

        assert_eq!(tweets[1].screen_name, "developer");
        assert_eq!(tweets[1].text, "Shipping the Rust rewrite today!");
        assert_eq!(tweets[1].reply_to.as_deref(), Some("rustlang"));
    }

    #[test]
    fn test_parse_tweets_handles_empty_or_malformed_json() {
        let empty = serde_json::json!({});
        assert_eq!(parse_tweets_from_response(&empty).len(), 0);

        let null_val = serde_json::Value::Null;
        assert_eq!(parse_tweets_from_response(&null_val).len(), 0);
    }
}
