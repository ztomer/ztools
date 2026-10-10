//! Where an activity happens: the weather label, and the kind of place it
//! follows from.
//!
//! The structure phase names the KIND of place a row's text gives ("hotel",
//! "beach", "art gallery") -- reading the text is the model's judgement. That
//! a hotel is a building and a beach is not is a fact about the noun, so the
//! label is looked up here rather than asked for: on the 2026-10-10 batches the
//! model named "hotel" correctly and then answered "unknown" for its weather.
//! A kind this table does not know (a "market" can be either) keeps the
//! model's own label; a kind of "unknown" keeps none, because a label with no
//! place behind it is a guess.

/// Kinds of place in the open.
const OUTDOOR_KINDS: &[&str] = &[
    "park",
    "beach",
    "farm",
    "fairground",
    "fair",
    "trail",
    "street",
    "garden",
    "zoo",
    "forest",
    "waterfront",
    "orchard",
    "pumpkin patch",
    "corn maze",
    "campground",
    "conservation area",
    "lake",
    "amusement park",
    "theme park",
];

/// Kinds of place inside a building. Checked before the outdoor list on the
/// whole kind, so "trampoline park" is a building and not a park.
const INDOOR_KINDS: &[&str] = &[
    "hotel",
    "library",
    "museum",
    "gallery",
    "mall",
    "shopping centre",
    "shopping center",
    "studio",
    "theatre",
    "theater",
    "arena",
    "cinema",
    "aquarium",
    "community centre",
    "community center",
    "recreation centre",
    "recreation center",
    "science centre",
    "science center",
    "planetarium",
    "trampoline park",
    "play centre",
    "play center",
    "indoor playground",
    "bowling alley",
    "arcade",
    "castle",
    "restaurant",
    "school",
    "church",
];

/// The weather label as the plan stores it: `indoor`, `outdoor`, `both` or `""`.
///
/// Anything else -- the model's "unknown", a stray word, nothing -- is `""`.
/// Everything downstream (the score, the report, the inversion check) reads
/// only these four, so the set is closed where the model's answer comes in.
#[must_use]
pub fn weather_label(raw: &str) -> &'static str {
    match raw.trim().to_lowercase().as_str() {
        "indoor" | "indoors" => "indoor",
        "outdoor" | "outdoors" => "outdoor",
        "both" => "both",
        _ => "",
    }
}

/// The label a kind of place implies, if the tables know it: the whole kind
/// first, then its last word (the head noun: "art gallery" is a gallery).
fn kind_label(kind: &str) -> Option<&'static str> {
    let kind = kind.trim().to_lowercase();
    let head = kind.rsplit(' ').next().unwrap_or_default();
    for candidate in [kind.as_str(), head] {
        if INDOOR_KINDS.contains(&candidate) {
            return Some("indoor");
        }
        if OUTDOOR_KINDS.contains(&candidate) {
            return Some("outdoor");
        }
    }
    None
}

/// A row's label from the kind of place the model named and the label it gave.
#[must_use]
pub fn label_for(kind: &str, said: &str) -> &'static str {
    if kind.trim().eq_ignore_ascii_case("unknown") {
        return "";
    }
    kind_label(kind).unwrap_or_else(|| weather_label(said))
}
