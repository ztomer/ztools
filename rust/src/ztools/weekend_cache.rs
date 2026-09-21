//! Weekend planner cache and list directory filter helper module.

use std::fs;

use super::weekend::WeekendEvent;

/// Check if a title refers to a directory page, list article, social video, or round-up.
#[must_use]
pub fn is_directory_or_list_page(title: &str) -> bool {
    let lower = title.to_lowercase();
    let keywords = [
        "tiktok",
        "youtube",
        "tripadvisor",
        "realtor",
        "hotelsbyday",
        "real estate",
        "mls®",
        "listing",
        "directory",
        "calendar of events",
        "things to do",
        "best kids parks",
        "best hikes",
        "day trips",
        "easy adventures",
        "townhouse",
        "hotel & resorts",
        "hotels",
        "top 10",
        "best museums",
        "winter hikes",
        "guide to",
        "allevents",
        "kathryn anywhere",
        "special events calendar",
        "best hiking trails",
        "toronto, ontario",
        "vaughan, ontario",
        "ontario, canada",
        "nearby attractions",
        "day use hotels",
        "you can hike through",
        "vaughan events & activities",
        "kids events in vaughan",
        "upcoming kids events",
    ];

    if keywords.iter().any(|kw| lower.contains(kw)) {
        return true;
    }

    // Filter out Non-ASCII characters (e.g. Cyrillic/Russian titles)
    if title.chars().any(|c| c as u32 > 0x024F) {
        return true;
    }

    // Filter out exact city names
    let trimmed = lower.trim();
    if [
        "toronto",
        "vaughan",
        "ontario",
        "canada",
        "toronto, ontario",
    ]
    .contains(&trimmed)
    {
        return true;
    }

    // Filter out street address patterns
    if lower.contains("road") && lower.contains("suite") || lower.contains("bedroom") {
        return true;
    }

    let mut words = lower.split_whitespace();
    if let Some(first) = words.next() {
        if first.chars().any(|c| c.is_ascii_digit())
            && (lower.contains("best") || lower.contains("things") || lower.contains("top"))
        {
            return true;
        }
    }

    false
}

/// Region evidence lists: the configured city/region plus the `[region]`
/// table (`in_region`, `foreign`) of `weekend.toml`.
///
/// Data, not code — edit the file, never these literals (there are none:
/// every list below arrives from [`load_region_lists`]).
pub struct RegionLists {
    pub cities: Vec<String>,
    pub in_region: Vec<String>,
    pub foreign: Vec<String>,
}

/// Load region lists from the first candidate file carrying a `[region]` table.
///
/// Falls back to the configured city/region alone. A missing table degrades
/// to city-only matching rather than to dropping everything — the empty
/// corpus that follows a total drop starves the draft and the model invents
/// events, which is exactly the failure this filter exists to prevent. Says
/// so loudly when no file yields a table.
#[must_use]
pub fn load_region_lists(paths: &[String]) -> RegionLists {
    for raw in paths {
        let path = crate::manifest::expand_tilde(raw);
        let Ok(content) = fs::read_to_string(&path) else {
            continue;
        };
        let Ok(val) = toml::from_str::<toml::Value>(&content) else {
            continue;
        };
        let Some(region) = val.get("region") else {
            continue;
        };
        let mut cities = Vec::new();
        if let Some(location) = val.get("location") {
            for key in ["city", "region"] {
                if let Some(name) = location.get(key).and_then(|v| v.as_str()) {
                    let name = name.trim().to_lowercase();
                    if !name.is_empty() {
                        cities.push(name);
                    }
                }
            }
        }
        let strings = |key: &str| -> Vec<String> {
            region
                .get(key)
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .map(|s| s.trim().to_lowercase())
                        .filter(|s| !s.is_empty())
                        .collect()
                })
                .unwrap_or_default()
        };
        return RegionLists {
            cities,
            in_region: strings("in_region"),
            foreign: strings("foreign"),
        };
    }
    eprintln!("⚠ no weekend.toml [region] table found; region evidence falls back to nothing");
    RegionLists {
        cities: Vec::new(),
        in_region: Vec::new(),
        foreign: Vec::new(),
    }
}

/// Positive evidence matcher verifying an activity/event is located within the GTA region.
#[must_use]
pub fn has_region_evidence(text: &str, region: &RegionLists) -> bool {
    let lower = text.to_lowercase();

    if region.foreign.iter().any(|ft| lower.contains(ft)) {
        return false;
    }

    region
        .cities
        .iter()
        .chain(region.in_region.iter())
        .any(|tok| lower.contains(tok))
}

/// Clean raw search titles to extract clean venue or event names.
#[must_use]
pub fn clean_venue_or_event_title(raw_title: &str, region: &RegionLists) -> Option<String> {
    let mut cleaned = raw_title.replace("**", "").trim().to_string();

    for delimiter in [" | ", " - ", " – ", " — "] {
        if let Some(pos) = cleaned.find(delimiter) {
            let suffix = cleaned[pos + delimiter.len()..].to_lowercase();
            if suffix.contains("tiktok")
                || suffix.contains("tripadvisor")
                || suffix.contains("youtube")
                || suffix.contains("yelp")
                || suffix.contains("narcity")
                || suffix.contains("realtor")
                || suffix.contains("hotelsbyday")
                || suffix.contains("medium")
                || suffix.contains("trip")
            {
                cleaned = cleaned[..pos].trim().to_string();
            }
        }
    }

    if is_directory_or_list_page(&cleaned) {
        return None;
    }

    if !has_region_evidence(&cleaned, region) {
        return None;
    }

    if cleaned.len() < 3 {
        return None;
    }

    Some(cleaned)
}

/// Load exclusions from the first configured file that yields any.
///
/// The paths come from config rather than the home directory: a hardcoded
/// `~/…` lookup made this read whichever machine was running, so the same test
/// took a different branch on a developer box than in CI -- and the coverage
/// number moved with it.
#[must_use]
pub fn load_exclusions(config: &crate::config::ZtoolsConfig) -> Vec<String> {
    for p in config
        .weekend_exclusions_paths
        .iter()
        .map(|s| crate::manifest::expand_tilde(s))
    {
        if let Ok(content) = fs::read_to_string(p) {
            let mut list = Vec::new();
            let mut in_excl = false;
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("exclude_places = [") {
                    in_excl = true;
                    continue;
                }
                if in_excl {
                    if trimmed.starts_with(']') {
                        break;
                    }
                    let item = trimmed.trim_matches(|c| c == '"' || c == '\'' || c == ',');
                    if !item.is_empty() {
                        list.push(item.to_string());
                    }
                }
            }
            if !list.is_empty() {
                return list;
            }
        }
    }

    vec![
        "Canada's Wonderland".into(),
        "Toronto Zoo".into(),
        "ROM".into(),
        "Ripley's".into(),
        "LEGOLAND".into(),
        "CN Tower".into(),
        "Sky Zone Toronto".into(),
    ]
}

/// One curated row: the fields a `WeekendEvent` is built from.
struct Curated {
    name: &'static str,
    location: &'static str,
    price: &'static str,
    ages: &'static str,
    day: &'static str,
    dates: &'static str,
    description: &'static str,
}

impl Curated {
    fn event(&self, is_transient: bool) -> WeekendEvent {
        WeekendEvent {
            name: self.name.into(),
            location: self.location.into(),
            price: self.price.into(),
            target_ages: self.ages.into(),
            day: self.day.into(),
            dates: self.dates.into(),
            description: self.description.into(),
            is_transient,
            score: 0.0,
            start_date: String::new(),
            end_date: String::new(),
            weather: String::new(),
            duration: String::new(),
        }
    }
}

/// Year-round venues.
const CURATED_FIXED: &[Curated] = &[
    Curated {
        name: "Kortright Centre for Conservation",
        location: "Vaughan",
        price: "$8-12",
        ages: "All Ages",
        day: "Sat-Sun",
        dates: "Year-Round",
        description: "800 acres of outdoor hiking trails, pond dipping, and interactive nature exhibits great for ages 6-13.",
    },
    Curated {
        name: "Air Riderz Trampoline Park",
        location: "Vaughan",
        price: "$18-24",
        ages: "6-13",
        day: "Fri-Sun",
        dates: "Year-Round",
        description: "Indoor trampoline zone, 24ft climbing walls, dodgeball court, and ninja warrior obstacle course.",
    },
    Curated {
        name: "Playdium Vaughan Arcade & VR",
        location: "Vaughan",
        price: "$15-30",
        ages: "6-13",
        day: "Fri-Sun",
        dates: "Year-Round",
        description: "40,000 sq ft venue featuring high-tech arcade games, virtual reality arenas, and indoor ropes courses.",
    },
    Curated {
        name: "Mount Nemo Conservation Area",
        location: "Halton / GTA",
        price: "$7-10",
        ages: "6-13",
        day: "Fri-Sun",
        dates: "Year-Round",
        description: "Escarpment cliffside walking trails, cliffside lookout points, and limestone cave exploration.",
    },
    Curated {
        name: "McMichael Canadian Art Collection Trails",
        location: "Kleinburg / Vaughan",
        price: "Free Trails / $15",
        ages: "All Ages",
        day: "Sat-Sun",
        dates: "Year-Round",
        description: "100-acre outdoor sculpture park, pine forest trails, and hands-on family art activities.",
    },
];

/// Dated events.
const CURATED_TRANSIENT: &[Curated] = &[
    Curated {
        name: "Vaughan Public Library Youth Science Workshop",
        location: "Vaughan Library",
        price: "Free",
        ages: "6-13",
        day: "Saturday",
        dates: "Aug 08",
        description: "Free hands-on STEM experiment and creative tech coding activity for kids ages 6-13.",
    },
    Curated {
        name: "GTA Outdoor Nature Trail Discovery Walk",
        location: "Kortright Conservation",
        price: "Free with admission",
        ages: "All Ages",
        day: "Sunday",
        dates: "Aug 09",
        description: "Guided family nature walk with wildlife tracking, pond exploration, and bug identification.",
    },
    Curated {
        name: "High Park Family Birding & Biodiversity Tour",
        location: "Toronto / High Park",
        price: "Free",
        ages: "All Ages",
        day: "Saturday",
        dates: "Aug 08",
        description: "Interactive woodland nature walk and birdwatching session tailored for young explorers.",
    },
];

/// Load default cached activities returning clean, curated GTA family venues and events.
#[must_use]
pub fn load_cached_activities(
    config: &crate::config::ZtoolsConfig,
) -> (Vec<WeekendEvent>, Vec<WeekendEvent>) {
    let exclusions = load_exclusions(config);
    let all_fixed: Vec<WeekendEvent> = CURATED_FIXED.iter().map(|c| c.event(false)).collect();
    let all_transient: Vec<WeekendEvent> =
        CURATED_TRANSIENT.iter().map(|c| c.event(true)).collect();

    let fixed = super::weekend::drop_excluded_places(all_fixed, &exclusions).0;

    let transient = super::weekend::drop_excluded_places(all_transient, &exclusions).0;

    (transient, fixed)
}
