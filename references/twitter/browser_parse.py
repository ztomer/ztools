#!/usr/bin/env python3
"""Parsing of x.com's HomeTimeline GraphQL payloads into plain tweet dicts.

Split out of `twitter.browser` to keep that module under the project's
500-line ceiling. Pure data transformation: no browser, no network.
"""

import os
from datetime import datetime, timezone

# Constants to eliminate magic numbers/strings (Mitchell Hashimoto design)
TWITTER_TYPENAME_VISIBILITY = "TweetWithVisibilityResults"
TWITTER_ITEM_TIMELINE_TWEET = "TimelineTweet"
TWITTER_TYPE_TIMELINE_ADD_ENTRIES = "TimelineAddEntries"
UNKNOWN_USER = "unknown"
TWITTER_DATE_FORMAT = "%a %b %d %H:%M:%S +0000 %Y"


def parse_tweets_from_response(data: dict) -> list[dict]:
    tweets = []
    try:
        instructions = (
            data.get("data", {})
            .get("home", {})
            .get("home_timeline_urt", {})
            .get("instructions", [])
        )
        for instruction in instructions:
            if instruction.get("type") != TWITTER_TYPE_TIMELINE_ADD_ENTRIES:
                continue
            for entry in instruction.get("entries", []):
                content = entry.get("content", {})
                item_content = content.get("itemContent", {})
                if item_content.get("itemType") != TWITTER_ITEM_TIMELINE_TWEET:
                    continue
                tweet_result = item_content.get("tweet_results", {}).get("result", {})
                if tweet_result.get("__typename") == TWITTER_TYPENAME_VISIBILITY:
                    tweet_result = tweet_result.get("tweet", tweet_result)

                legacy = tweet_result.get("legacy", {})
                user_result = tweet_result.get("core", {}).get("user_results", {}).get("result", {})
                user_core = user_result.get("core", {})
                user_legacy = user_result.get("legacy", {})
                full_text = legacy.get("full_text", "")
                created_at_str = legacy.get("created_at", "")
                screen_name = (
                    user_core.get("screen_name") or user_legacy.get("screen_name") or UNKNOWN_USER
                )
                favorite_count = legacy.get("favorite_count", 0)
                retweet_count = legacy.get("retweet_count", 0)
                reply_count = legacy.get("reply_count", 0)
                tweet_id = legacy.get("id_str", "")
                in_reply_to = legacy.get("in_reply_to_screen_name", "")

                if not full_text or not created_at_str:
                    continue

                try:
                    created_at = datetime.strptime(created_at_str, TWITTER_DATE_FORMAT)
                    created_at = created_at.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue

                tweet = {
                    "screen_name": screen_name,
                    "text": full_text,
                    "created_at": created_at,
                    "favorite_count": favorite_count,
                    "retweet_count": retweet_count,
                    "reply_count": reply_count,
                }
                if tweet_id:
                    tweet["id_str"] = tweet_id
                if in_reply_to:
                    tweet["in_reply_to_screen_name"] = in_reply_to

                tweets.append(tweet)
    except Exception as e:
        if os.environ.get("DEBUG"):
            print(f"Error: {e}")
    return tweets
