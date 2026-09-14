//! x.com GraphQL endpoint markers, read from data, not code.
//!
//! Port of the `[endpoints]` table in `conf/twitter.toml`, the single source
//! both collectors read: the timeline URL markers plus which one is the
//! Following (reverse-chron) feed. Comparing ID sets across feeds is
//! meaningless, so a run that never sees the Following endpoint must fail
//! loudly rather than produce a plausible-but-wrong set.

use anyhow::Result;

/// Timeline URL markers with the Following designation, as read from
/// `conf/twitter.toml [endpoints]`. No literals live here: every string
/// below arrives from [`load_endpoint_markers`].
pub struct EndpointMarkers {
    pub(crate) timeline: Vec<String>,
    pub(crate) following: String,
}

impl EndpointMarkers {
    /// Whether a response URL carries timeline tweets at all.
    #[must_use]
    pub fn is_timeline_url(&self, url: &str) -> bool {
        self.timeline.iter().any(|m| url.contains(m))
    }

    /// Whether a response URL is the Following timeline endpoint.
    #[must_use]
    pub fn is_following_url(&self, url: &str) -> bool {
        !self.following.is_empty() && url.contains(&self.following)
    }
}

/// Load endpoint markers from the first candidate file carrying an
/// `[endpoints]` table with a non-empty `following` marker.
///
/// # Errors
///
/// When no candidate yields a usable table. An empty marker set would match
/// nothing and fail every run with the wrong-feed message, so the absence is
/// reported here, where the cause is visible, instead.
pub fn load_endpoint_markers(paths: &[String]) -> Result<EndpointMarkers> {
    for raw in paths {
        let path = crate::manifest::expand_tilde(raw);
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(val) = toml::from_str::<toml::Value>(&content) else {
            continue;
        };
        let Some(table) = val.get("endpoints") else {
            continue;
        };
        let following = table
            .get("following")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .unwrap_or_default();
        if following.is_empty() {
            continue;
        }
        let timeline: Vec<String> = table
            .get("timeline")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default();
        return Ok(EndpointMarkers {
            timeline,
            following: following.to_string(),
        });
    }
    anyhow::bail!("no twitter.toml [endpoints] table found with a following marker")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    fn write_conf(dir: &std::path::Path, body: &str) -> String {
        let path = dir.join("twitter.toml");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(body.as_bytes()).unwrap();
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn loads_markers_from_the_first_file_carrying_the_table() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir
            .path()
            .join("absent.toml")
            .to_string_lossy()
            .into_owned();
        let good = write_conf(
            dir.path(),
            "[endpoints]\ntimeline = [\"HomeTimeline\", \"HomeLatestTimeline\"]\nfollowing = \"HomeLatestTimeline\"\n",
        );
        let markers =
            load_endpoint_markers(&[missing, good]).expect("shipped-shaped table must load");
        assert!(markers.is_timeline_url("https://x.com/api/graphql/HomeTimeline"));
        assert!(markers.is_timeline_url("https://x.com/api/graphql/HomeLatestTimeline"));
        assert!(!markers.is_timeline_url("https://x.com/api/graphql/UserByScreenName"));
        assert!(markers.is_following_url("https://x.com/api/graphql/HomeLatestTimeline?a=1"));
        assert!(!markers.is_following_url("https://x.com/api/graphql/HomeTimeline?a=1"));
    }

    #[test]
    fn missing_table_is_an_error_not_an_empty_matcher() {
        // An empty marker set would match nothing and fail every run with the
        // wrong-feed message. Failing HERE names the real cause instead.
        let dir = tempfile::tempdir().unwrap();
        let path = write_conf(dir.path(), "[location]\ncity = \"Vaughan\"\n");
        assert!(load_endpoint_markers(&[path]).is_err());
        assert!(load_endpoint_markers(&[]).is_err());
    }

    #[test]
    fn shipped_config_loads() {
        let manifest = env!("CARGO_MANIFEST_DIR");
        let path = std::path::Path::new(manifest)
            .parent()
            .unwrap()
            .join("conf/twitter.toml");
        let markers =
            load_endpoint_markers(&[path.to_string_lossy().into_owned()]).expect("shipped conf");
        assert!(markers.is_following_url("https://x.com/api/graphql/HomeLatestTimeline"));
    }
}
