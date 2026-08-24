//! Scoring weights, thresholds and the shared field/stopword tables.
//!
//! Split out of json_validator.rs for the 500-line production cap. Data only --
//! no logic -- so the numbers that define the scoring contract sit in one place
//! rather than beside the code that happens to apply them first.

use std::collections::HashSet;
use std::sync::LazyLock;

pub const MAX_SCORE: i64 = 100;
pub const MIN_ITEMS_GOOD: usize = 8;
pub const MIN_ITEMS_OK: usize = 5;
pub const JSON_STRUCTURE_WEIGHT: i64 = 20;
pub const JSON_COUNT_GOOD: i64 = 25;
pub const JSON_COUNT_OK: i64 = 15;
pub const JSON_VALIDITY_WEIGHT: i64 = 30;
pub const JSON_VALIDITY_THRESHOLD: f64 = 0.7;
pub const JSON_SOURCE_WEIGHT: i64 = 25;

pub const DETAILED_STRUCTURE_WEIGHT: i64 = 15;
pub const DETAILED_COUNT_GOOD: i64 = 15;
pub const DETAILED_COUNT_OK: i64 = 10;
pub const DETAILED_QUALITY_WEIGHT: i64 = 40;
pub const DETAILED_SOURCE_WEIGHT: i64 = 30;
pub const JSON_QUALITY_WEIGHT: i64 = 25;
pub const DETAIL_REQUIRED_FIELDS: usize = 3;

pub const SOURCE_THRESHOLD_HIGH: f64 = 0.8;
pub const SOURCE_THRESHOLD_MED: f64 = 0.5;
pub const SOURCE_THRESHOLD_LOW: f64 = 0.2;
pub const MAX_SCORE_HIGH_SOURCE: i64 = 100;
pub const MAX_SCORE_MED_SOURCE: i64 = 85;
pub const MAX_SCORE_LOW_SOURCE: i64 = 70;
pub const MAX_SCORE_NO_SOURCE: i64 = 50;

pub(super) static STOPWORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "the", "and", "for", "with", "this", "that", "from", "are", "was", "has", "have", "but",
        "not", "you", "all", "can", "her", "his", "had", "they", "been", "will", "would", "could",
        "what", "when", "where", "who", "which", "why", "how",
    ]
    .into_iter()
    .collect()
});

pub const DETAIL_FIELDS: &[&str] = &[
    "name",
    "event",
    "title",
    "activity",
    "place",
    "location",
    "venue",
    "address",
    "where",
    "day",
    "date",
    "when",
    "time",
    "duration",
    "target_ages",
    "age_group",
    "ages",
    "audience",
    "who",
    "price",
    "cost",
    "pricing",
    "weather",
    "type",
    "indoor_outdoor",
    "setting",
    "desc",
    "description",
];
